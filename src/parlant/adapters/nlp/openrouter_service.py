# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from functools import cached_property
import time
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncClient,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from typing import Any, Callable, Literal, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import (
    EmbedderHints,
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService,
    NoModeration,
)
from parlant.core.tracer import Tracer

RATE_LIMIT_ERROR_MESSAGE = """\
OpenRouter API rate limit exceeded. Possible reasons:
1. Your account may have insufficient API credits.
2. You may be using a free-tier account with limited request capacity.
3. You might have exceeded the requests-per-minute limit for your account.

Recommended actions:
- Check your OpenRouter account balance and billing status.
- Review your API usage limits in OpenRouter's dashboard.
- For more details on rate limits and usage tiers, visit:
    https://openrouter.ai/docs/api-reference/limits
"""


class OpenRouterEmptyEmbeddingResponseError(Exception):
    """Raised when OpenRouter returns an embedding response with no vectors."""


def _create_openrouter_client() -> AsyncClient:
    """Create an OpenAI-compatible client for the OpenRouter API, including the
    optional attribution headers OpenRouter supports.

    OPENROUTER_BASE_URL redirects requests to an OpenRouter-compatible gateway
    (e.g. a proxy that forwards to OpenRouter)."""
    extra_headers: dict[str, str] = {}
    if "OPENROUTER_HTTP_REFERER" in os.environ:
        extra_headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
    if "OPENROUTER_SITE_NAME" in os.environ:
        extra_headers["X-Title"] = os.environ["OPENROUTER_SITE_NAME"]

    return AsyncClient(
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers=extra_headers if extra_headers else None,
    )


def _extract_cached_input_tokens(usage: CompletionUsage | None) -> int:
    """Extract the number of cache-hit prompt tokens from a usage payload.

    OpenRouter normalizes cache reporting to the OpenAI-compatible
    usage.prompt_tokens_details.cached_tokens field; some providers (e.g.
    DeepSeek) additionally attach a non-standard prompt_cache_hit_tokens field.
    """
    if usage is None:
        return 0

    details = getattr(usage, "prompt_tokens_details", None)
    detail_cached = getattr(details, "cached_tokens", None) if details is not None else None
    if isinstance(detail_cached, int):
        return detail_cached

    legacy_cached = getattr(usage, "prompt_cache_hit_tokens", None)
    return legacy_cached if isinstance(legacy_cached, int) else 0


class OpenRouterEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # Use the gpt-4o encoding as a default approximation for token estimation
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class OpenRouterSchematicGenerator(BaseSchematicGenerator[T]):
    supported_openrouter_params = ["temperature", "max_tokens"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._client = _create_openrouter_client()
        self._tokenizer = OpenRouterEstimatingTokenizer(model_name=self.model_name)

        # The strongest response-format mode known to work for this model.
        # Demoted (json_schema -> json_object -> plain) when a provider rejects
        # the mode, so failing modes are not retried on every request.
        self._response_format_mode: Literal["json_schema", "json_object", "plain"] = "json_schema"

        # Optional completion-token cap, forwarded to the API as max_tokens.
        # A max_tokens hint always takes precedence over this value.
        completion_max_tokens = os.environ.get("OPENROUTER_COMPLETION_MAX_TOKENS")
        self._completion_max_tokens = int(completion_max_tokens) if completion_max_tokens else None

    @property
    @override
    def id(self) -> str:
        return f"openrouter/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenRouterEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        # Default implementation - should be overridden by subclasses
        return 8192

    @cached_property
    def _schema_json(self) -> dict[str, Any]:
        """The JSON Schema of the target schema, computed once per instance.

        Strict structured-output providers (e.g. OpenAI) require object nodes to
        carry additionalProperties=false and every property to be required, so the
        schema is transformed with the OpenAI SDK's strict converter. When the
        transformation is unavailable or rejects the schema, the raw Pydantic
        schema is sent instead; providers that then refuse it trigger the
        json_object fallback in _create_completion.
        """
        try:
            from openai.lib._pydantic import to_strict_json_schema

            return dict(to_strict_json_schema(self.schema))
        except Exception:
            return self.schema.model_json_schema()

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    ConflictError,
                    RateLimitError,
                    APIResponseValidationError,
                    OpenRouterEmptyEmbeddingResponseError,
                ),
            ),
            retry(InternalServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"OpenRouter LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    @staticmethod
    def _is_structured_outputs_rejection(error: Exception) -> bool:
        """Heuristically detect provider errors caused by the json_schema
        response_format / structured-outputs requirement."""
        error_str = str(error).lower()
        return any(
            marker in error_str
            for marker in (
                "json_schema",
                "json schema",
                # "json mode" deliberately overlaps with _is_json_mode_rejection:
                # a JSON-mode rejection while in json_schema mode demotes to
                # json_object first, and only then (if it fails again) to plain.
                "json mode",
                "structured output",
                "structured_outputs",
                "response_format",
                "require_parameters",
                "no endpoints found",
            )
        )

    @staticmethod
    def _is_json_mode_rejection(error: Exception) -> bool:
        """Heuristically detect provider errors caused by JSON mode."""
        error_str = str(error)
        return "JSON mode" in error_str or "json_object" in error_str.lower()

    async def _create_completion(
        self,
        prompt: str,
        api_arguments: Mapping[str, Any],
    ) -> ChatCompletion:
        """Issue a chat-completion request, transmitting the schema with the
        strongest response-format mode the model supports.

        Modes demote monotonically (json_schema -> json_object -> plain) when a
        provider rejects one, and the working mode is memoized per instance.
        """
        while True:
            mode = self._response_format_mode
            try:
                response: ChatCompletion

                if mode == "json_schema":
                    response = await self._client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model_name,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": self.schema.__name__,
                                "strict": True,
                                "schema": self._schema_json,
                            },
                        },
                        extra_body={"provider": {"require_parameters": True}},
                        **api_arguments,
                    )
                elif mode == "json_object":
                    response = await self._client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model_name,
                        response_format={"type": "json_object"},
                        **api_arguments,
                    )
                else:
                    # Last resort: instruct the model to emit JSON via a system
                    # message, without any response_format enforcement.
                    json_instruction = (
                        "IMPORTANT: You must respond with ONLY valid JSON. "
                        "No explanatory text before or after the JSON. "
                        "The response must be a valid JSON object."
                    )
                    response = await self._client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": json_instruction},
                            {"role": "user", "content": prompt},
                        ],
                        model=self.model_name,
                        **api_arguments,
                    )

                return response
            except (BadRequestError, NotFoundError) as e:
                if mode == "json_schema" and self._is_structured_outputs_rejection(e):
                    self.logger.warning(
                        f"Model '{self.model_name}' rejected json_schema structured outputs"
                        f" ({type(e).__name__}: {e}).\n"
                        f"Falling back to JSON mode for this generator instance."
                    )
                    self._response_format_mode = "json_object"
                    continue

                if mode == "json_object" and self._is_json_mode_rejection(e):
                    self.logger.warning(
                        f"Model '{self.model_name}' does not support JSON mode"
                        f" ({type(e).__name__}: {e}).\n"
                        f"Please consider switching to a model that supports JSON mode"
                        f" (e.g., 'openai/gpt-4o', 'anthropic/claude-sonnet-4.6').\n"
                        f"Falling back to plain JSON instructions, but results may be"
                        f" less reliable."
                    )
                    self._response_format_mode = "plain"
                    continue

                self.logger.error(f"OpenRouter API {type(e).__name__}: {e}")
                raise
            except RateLimitError:
                self.logger.error(
                    f"\nRate limit exceeded for model '{self.model_name}'.\n"
                    f"{RATE_LIMIT_ERROR_MESSAGE}\n"
                    f"Consider:\n"
                    f"  - Using a different model\n"
                    f"  - Waiting a moment before retrying\n"
                    f"  - Adding your own API key for higher limits\n"
                )
                raise
            except Exception as e:
                self.logger.error(
                    f"\nOpenRouter API error with model '{self.model_name}': {type(e).__name__}\n"
                    f"{e}\n"
                    f"Consider switching to a more compatible model.\n"
                )
                raise

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        openrouter_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_openrouter_params
        }

        if "max_tokens" not in openrouter_api_arguments and self._completion_max_tokens is not None:
            openrouter_api_arguments["max_tokens"] = self._completion_max_tokens

        t_start = time.time()
        response = await self._create_completion(prompt, openrouter_api_arguments)
        t_end = time.time()

        if response.usage:
            self.logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError as decode_error:
            self.logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content}")
            try:
                json_content = jsonfinder.only_json(raw_content)[2]
                self.logger.warning("Found JSON content within model response; continuing...")
            except Exception:
                self.logger.error(
                    f"Could not extract valid JSON from the response of '{self.model_name}':\n"
                    f"{raw_content}"
                )
                # Re-raise the original decoding error so that
                # BaseSchematicGenerator.generate() can retry the generation.
                raise decode_error

        try:
            content = self.schema.model_validate(json_content)
        except ValidationError as e:
            self.logger.error(
                f"\nJSON content returned by '{self.model_name}' does not match expected schema.\n"
                f"Schema: {self.schema.__name__}\n"
                f"Raw response: {raw_content}\n"
                f"Parsed JSON: {json.dumps(json_content, indent=2) if json_content else 'Empty'}\n"
                f"Validation errors: {str(e)}\n"
            )
            raise

        usage = response.usage
        if usage is None:
            self.logger.warning(
                f"OpenRouter response for '{self.model_name}' did not include usage data;"
                f" recording zero token counts."
            )

        input_tokens = (usage.prompt_tokens or 0) if usage else 0
        output_tokens = (usage.completion_tokens or 0) if usage else 0
        cached_input_tokens = _extract_cached_input_tokens(usage)

        await record_llm_metrics(
            self.meter,
            self.model_name,
            schema_name=self.schema.__name__,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

        return SchematicGenerationResult(
            content=content,
            info=GenerationInfo(
                schema_name=self.schema.__name__,
                model=self.id,
                duration=(t_end - t_start),
                usage=UsageInfo(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    extra={"cached_input_tokens": cached_input_tokens},
                ),
            ),
        )


class OpenRouterGPT4O(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="openai/gpt-4o-2024-11-20", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class OpenRouterGPT4OMini(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="openai/gpt-4o-mini-2024-07-18", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class OpenRouterClaudeSonnet46(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="anthropic/claude-sonnet-4.6", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1_000_000


class OpenRouterLlama33_70B(OpenRouterSchematicGenerator[T]):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="meta-llama/llama-3.3-70b-instruct",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 128 * 1024


class OpenRouterEmbedder(BaseEmbedder):
    supported_arguments = ["dimensions"]

    # Known embedding model dimensions
    _KNOWN_DIMENSIONS: dict[str, int] = {
        "openai/text-embedding-3-large": 3072,
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-ada-002": 1536,
        "qwen/qwen3-embedding-8b": 4096,
        "qwen/qwen-embedding-v2": 1536,
    }

    def __init__(self, model_name: str, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(logger, tracer, meter, model_name)

        self._client = _create_openrouter_client()
        self._tokenizer = OpenRouterEstimatingTokenizer(model_name=self.model_name)
        # Cache dimensions after first API call if not known
        self._cached_dimensions: int | None = None

    @property
    @override
    def id(self) -> str:
        return f"openrouter/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OpenRouterEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        # Default max tokens for embedding models
        return 8192

    @property
    @override
    def dimensions(self) -> int:
        # Check environment variable override first
        if "OPENROUTER_EMBEDDER_DIMENSIONS" in os.environ:
            return int(os.environ["OPENROUTER_EMBEDDER_DIMENSIONS"])

        # Return cached dimensions if available
        if self._cached_dimensions is not None:
            return self._cached_dimensions

        # Check known dimensions lookup
        for model_key, dims in self._KNOWN_DIMENSIONS.items():
            if model_key in self.model_name:
                return dims

        # Default fallback - most embedding models use 1536 or 3072
        # This will be updated after first API call
        return 1536

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    ConflictError,
                    RateLimitError,
                    APIResponseValidationError,
                    OpenRouterEmptyEmbeddingResponseError,
                ),
            ),
            retry(InternalServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        filtered_hints = {k: v for k, v in hints.items() if k in self.supported_arguments}
        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=texts,
                **filtered_hints,
            )
        except ValueError as exc:
            if "No embedding data received" in str(exc):
                raise OpenRouterEmptyEmbeddingResponseError(str(exc)) from exc
            raise
        except RateLimitError:
            self.logger.error(
                f"\nRate limit exceeded for embedder model '{self.model_name}'.\n"
                f"{RATE_LIMIT_ERROR_MESSAGE}\n"
                f"Consider:\n"
                f"  - Using a different embedder model\n"
                f"  - Waiting a moment before retrying\n"
                f"  - Adding your own API key for higher limits\n"
            )
            raise

        if not response.data:
            raise OpenRouterEmptyEmbeddingResponseError("No embedding data received")

        vectors = [data_point.embedding for data_point in response.data]

        # Cache dimensions from first response if not already cached and not in known list
        if self._cached_dimensions is None and vectors:
            actual_dims = len(vectors[0])
            # Only cache if different from default or if not found in known dimensions
            if actual_dims != 1536 or not any(
                key in self.model_name for key in self._KNOWN_DIMENSIONS
            ):
                self._cached_dimensions = actual_dims
                self.logger.debug(
                    f"Detected embedding dimensions for '{self.model_name}': {actual_dims}"
                )

        return EmbeddingResult(vectors=vectors)


class OpenRouterTextEmbedding3Large(OpenRouterEmbedder):
    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        super().__init__(
            model_name="openai/text-embedding-3-large", logger=logger, tracer=tracer, meter=meter
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def dimensions(self) -> int:
        return 3072


class OpenRouterService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("OPENROUTER_API_KEY"):
            return """\
You're using the OpenRouter NLP service, but OPENROUTER_API_KEY is not set.
Please set OPENROUTER_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._logger.info("Initialized OpenRouterService")
        # Get model_name from environment variable
        self.model_name = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
        # Get embedder_model_name from environment variable
        self.embedder_model_name = os.environ.get(
            "OPENROUTER_EMBEDDER_MODEL", "openai/text-embedding-3-large"
        )
        self._logger.info(f"OpenRouter model name: {self.model_name}")
        self._logger.info(f"OpenRouter embedder model name: {self.embedder_model_name}")

        # Create dynamic embedder class that can be resolved from the container
        # This captures embedder_model_name in a closure so the container can resolve it
        embedder_model = self.embedder_model_name

        class DynamicOpenRouterEmbedder(OpenRouterEmbedder):
            def __init__(self, logger: Logger, tracer: Tracer, meter: Meter):
                super().__init__(
                    model_name=embedder_model, logger=logger, tracer=tracer, meter=meter
                )

        self._dynamic_embedder_class = DynamicOpenRouterEmbedder

    @property
    @override
    def supports_streaming(self) -> bool:
        return False

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        raise NotImplementedError("Streaming is not supported. Check supports_streaming first.")

    def _get_specialized_generator_class(
        self,
        model_name: str,
        t: type[T],
    ) -> Callable[[Logger, Tracer, Meter], OpenRouterSchematicGenerator[T]]:
        """
        Returns the specialized generator class for known models.
        For unknown models, creates a dynamic generator that works with any OpenRouter model.
        """
        model_mapping: dict[
            str, Callable[[Logger, Tracer, Meter], OpenRouterSchematicGenerator[T]]
        ] = {
            "openai/gpt-4o": lambda logger, tracer, meter: OpenRouterGPT4O[t](  # type: ignore
                logger, tracer, meter
            ),
            "openai/gpt-4o-mini": lambda logger, tracer, meter: OpenRouterGPT4OMini[t](  # type: ignore
                logger, tracer, meter
            ),
            "anthropic/claude-sonnet-4.6": lambda logger, tracer, meter: OpenRouterClaudeSonnet46[
                t  # type: ignore
            ](logger, tracer, meter),
            "meta-llama/llama-3.3-70b-instruct": lambda logger, tracer, meter: (
                OpenRouterLlama33_70B[t](  # type: ignore
                    logger, tracer, meter
                )
            ),
        }

        # Check if we have a predefined generator for this model
        if generator_factory := model_mapping.get(model_name):
            return generator_factory

        # Create a dynamic generator for any OpenRouter model
        # Get max_tokens from environment variable or use sensible defaults based on model name
        max_tokens_str = os.environ.get("OPENROUTER_MAX_TOKENS")
        if max_tokens_str:
            max_tokens = int(max_tokens_str)
        else:
            # Ordered (first match wins) context-window defaults by model family.
            family_context_windows = (
                ("gpt-4", 128 * 1024),
                ("claude", 200 * 1024),
                ("llama-3", 128 * 1024),  # must precede the generic "llama" entry
                ("llama", 8192),
                ("gemma", 8192),
            )
            max_tokens = next(
                (
                    context_window
                    for family, context_window in family_context_windows
                    if family in model_name
                ),
                8192,  # Safe default for unknown models
            )

        # Create dynamic generator class with the specific max_tokens
        final_max_tokens = max_tokens

        class DynamicOpenRouterGenerator(OpenRouterSchematicGenerator[T]):
            def __init__(self, logger: Logger, tracer: Tracer, meter: Meter):
                super().__init__(model_name=model_name, logger=logger, tracer=tracer, meter=meter)

            @property
            @override
            def max_tokens(self) -> int:
                return final_max_tokens

        # Return a factory function that creates the properly typed instance
        def create_generator(
            logger: Logger, tracer: Tracer, meter: Meter
        ) -> OpenRouterSchematicGenerator[T]:
            return DynamicOpenRouterGenerator[t](logger, tracer, meter)  # type: ignore

        return create_generator

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> OpenRouterSchematicGenerator[T]:
        generator_factory = self._get_specialized_generator_class(self.model_name, t)
        return generator_factory(self._logger, self._tracer, self._meter)

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        # Use OpenRouter embedder with the configured embedder model name
        # Default to text-embedding-3-large if not specified
        if self.embedder_model_name == "openai/text-embedding-3-large":
            return OpenRouterTextEmbedding3Large(
                logger=self._logger, tracer=self._tracer, meter=self._meter
            )
        else:
            # Return instance of dynamic embedder class that can be resolved from container
            return self._dynamic_embedder_class(
                logger=self._logger, tracer=self._tracer, meter=self._meter
            )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
