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
from typing import Any, Callable, ClassVar, Literal, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import (
    BuiltInSection,
    PromptBuilder,
    render_section,
)
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


_ResponseFormatMode = Literal["json_schema", "json_object", "plain"]


# A chat message's content is either a plain string or a list of content blocks;
# the block form carries a per-block cache_control breakpoint for prompt caching.
_ContentBlock = dict[str, Any]

# Minimum stable-prefix size (characters) worth a cache breakpoint. Below the
# common provider cache minimum (~1024 tokens ≈ 4096 chars) a breakpoint just adds
# overhead without ever producing a cache hit.
_MIN_CACHE_PREFIX_CHARS = 4096

# Once-per-process-per-model log dedup (mirrors the _cache_blocks_unsupported
# memoization pattern). A prefix divergence means caching silently degrades to a
# flat prompt on every turn — that must be visible to operators, but exactly
# once, not per generation. The two INFO sets give a positive deployment signal
# that the split fired and that the provider actually reported cache hits.
_split_divergence_warned: set[str] = set()
_cache_split_logged: set[str] = set()
_cache_hit_logged: set[str] = set()

# PromptBuilder sections whose rendered content can change from one turn to the
# next. The cacheable prefix is the contiguous run of sections *before* the first
# of these. Only AGENT_IDENTITY / CUSTOMER_IDENTITY and custom (string-keyed)
# instruction sections are treated as stable; every BuiltInSection member must be
# placed in exactly one of the two sets below — the asserts after them enforce
# coverage at import time.
_DYNAMIC_SECTIONS: frozenset[BuiltInSection] = frozenset(
    {
        BuiltInSection.CONTEXT_VARIABLES,
        BuiltInSection.GLOSSARY,
        BuiltInSection.CAPABILITIES,
        BuiltInSection.GUIDELINE_DESCRIPTIONS,
        BuiltInSection.GUIDELINES,
        BuiltInSection.JOURNEYS,
        BuiltInSection.OBSERVATIONS,
        BuiltInSection.INTERACTION_HISTORY,
        BuiltInSection.STAGED_EVENTS,
    }
)

# The complement of _DYNAMIC_SECTIONS: built-in sections whose rendered content is
# stable across turns and may safely sit inside the cached prefix.
_STABLE_BUILTIN_SECTIONS: frozenset[BuiltInSection] = frozenset(
    {
        BuiltInSection.AGENT_IDENTITY,
        BuiltInSection.CUSTOMER_IDENTITY,
    }
)

# Force a deliberate classification of every BuiltInSection. A new member added to
# the enum without being placed in exactly one of the two sets above trips this at
# import time, rather than silently defaulting to "stable" and risking a per-turn
# section being served from a stale cache.
assert _DYNAMIC_SECTIONS | _STABLE_BUILTIN_SECTIONS == frozenset(BuiltInSection), (
    "Every BuiltInSection must be classified as dynamic or stable for prompt caching; "
    f"unclassified: {frozenset(BuiltInSection) - (_DYNAMIC_SECTIONS | _STABLE_BUILTIN_SECTIONS)}"
)
assert not (_DYNAMIC_SECTIONS & _STABLE_BUILTIN_SECTIONS), (
    "A BuiltInSection cannot be both dynamic and stable: "
    f"{_DYNAMIC_SECTIONS & _STABLE_BUILTIN_SECTIONS}"
)


def _split_prompt_for_caching(
    prompt: PromptBuilder,
    built: str,
    *,
    ttl: str | None,
    min_prefix_chars: int,
    logger: Logger | None = None,
    model_name: str | None = None,
) -> list[_ContentBlock] | None:
    """Split an already-built prompt into a cached stable-prefix block and an
    uncached variable-tail block, or return None when no worthwhile (or safe) split
    exists.

    *built* is ``prompt.build()``'s output, passed in so the full prompt is
    rendered only once. The breakpoint is placed before the first per-turn
    (dynamic) section; only the leading prefix sections are re-rendered, to locate
    the cut point. The returned block texts concatenate to exactly *built*, so the
    prompt the model receives is byte-for-byte unchanged. When the dynamic tail is
    empty this turn the whole prompt is returned as a single cached block; when the
    re-rendered prefix cannot be reconciled with *built*, None is returned so the
    caller falls back to a flat prompt rather than risk corrupting it.
    """
    items = list(prompt.sections.items())

    first_dynamic = next(
        (i for i, (name, _) in enumerate(items) if name in _DYNAMIC_SECTIONS),
        None,
    )
    # No dynamic section (nothing to keep out of the cache), or the very first
    # section is already dynamic (no stable prefix) -> nothing worthwhile to cache.
    if first_dynamic is None or first_dynamic == 0:
        return None

    # Re-render only the prefix sections to find where the cacheable prefix ends in
    # `built`. render_section is the same function build() uses, so the prefix is
    # byte-identical to its slice of `built` by construction. build() concatenates
    # every section as `render + "\n\n"` then strips the whole prompt; this
    # lstripped join mirrors that buffer for the prefix. The reconciliation below
    # handles the cases where build()'s trailing strip makes the two diverge.
    prefix_text = (
        "\n\n".join(
            render_section(section.template, section.props) for _, section in items[:first_dynamic]
        )
        + "\n\n"
    ).lstrip()
    if len(prefix_text) < min_prefix_chars:
        return None

    # Normally `built` starts with the reconstructed prefix and the tail is the rest.
    # Two cases break that, and neither may corrupt the prompt or crash the turn:
    #
    #  1. Every section from the first dynamic one onward rendered to whitespace. Then
    #     build()'s trailing .strip() trimmed the prefix's own trailing "\n\n", so
    #     `built` equals the whitespace-stripped prefix. There is no per-turn content
    #     this turn, so the whole prompt is stable and is cached as a single block.
    #  2. A section rendered non-deterministically (different bytes than during
    #     build()). The reconstructed prefix is then unrelated to `built`; degrade to
    #     a flat prompt rather than slice `built` at a meaningless offset. Caching is
    #     an optimization, never worth corrupting a prompt or crashing a turn.
    if built.startswith(prefix_text):
        suffix_text = built[len(prefix_text) :]
    elif built == prefix_text.rstrip():
        prefix_text = built
        suffix_text = ""
    else:
        # Divergence means a permanent silent cache miss for this prompt shape;
        # surface it once per model so operators can tell "caching off" apart
        # from "caching working".
        if logger is not None and (model_name or "") not in _split_divergence_warned:
            _split_divergence_warned.add(model_name or "")
            logger.warning(
                f"Prompt cache: stable-prefix re-render diverged from built prompt"
                f" for model '{model_name}'; falling back to a flat (uncached)"
                f" prompt. This indicates a non-deterministically rendering"
                f" section. Warned once; subsequent divergences are silent."
            )
        return None

    # {"type": "ephemeral"} is the standard breakpoint; the optional "ttl" (e.g.
    # "1h") is honored by OpenRouter and Anthropic's extended-TTL cache. Providers
    # that don't support caching ignore the whole key.
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl

    # The prefix carries the cache breakpoint; the variable tail, when present, is a
    # second uncached block. The block texts concatenate to exactly `built`.
    blocks: list[_ContentBlock] = [
        {"type": "text", "text": prefix_text, "cache_control": cache_control},
    ]
    if suffix_text:
        blocks.append({"type": "text", "text": suffix_text})
    return blocks


class OpenRouterSchematicGenerator(BaseSchematicGenerator[T]):
    supported_openrouter_params = ["temperature", "max_tokens"]

    # Process-wide cache of the strongest response-format mode known to work per
    # model. Demoted (json_schema -> json_object -> plain) when a provider rejects
    # a mode, so neither this instance nor newly created generators for the same
    # model waste a round-trip on an already-rejected mode.
    _response_format_modes: ClassVar[dict[str, _ResponseFormatMode]] = {}

    # Models observed to reject the content-block / cache_control message shape.
    # Such models fall back to a plain-string prompt and skip the cache breakpoint
    # for the rest of the process, avoiding a wasted round-trip on every request.
    _cache_blocks_unsupported: ClassVar[set[str]] = set()

    # Safe defaults so a subclass that overrides __init__ without chaining up
    # (e.g. to swap the client base URL) still reads these without AttributeError.
    # __init__ overwrites both with the env-driven values; the defaults only ever
    # govern a bypassing subclass, which degrades to caching-off, never a crash.
    # Plain instance annotations (not ClassVar): unlike the genuinely class-shared
    # _response_format_modes / _cache_blocks_unsupported above, these are per-instance
    # config, so __init__'s `self.` assignment must not trip "assign to ClassVar".
    _prompt_caching_enabled: bool = False
    _cache_ttl: str | None = None

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

        # Optional completion-token cap, forwarded to the API as max_tokens.
        # A max_tokens hint always takes precedence over this value.
        completion_max_tokens = os.environ.get("OPENROUTER_COMPLETION_MAX_TOKENS")
        self._completion_max_tokens = int(completion_max_tokens) if completion_max_tokens else None

        # Prompt caching: split the prompt so its large, stable prefix carries a
        # cache_control breakpoint (honored by Gemini, required by Anthropic Claude,
        # ignored by others). On by default; OPENROUTER_PROMPT_CACHE=false restores
        # the legacy flat-string request.
        cache_flag = os.environ.get("OPENROUTER_PROMPT_CACHE", "true").strip().lower()
        self._prompt_caching_enabled = cache_flag not in ("0", "false", "no", "off")
        self._cache_ttl = os.environ.get("OPENROUTER_PROMPT_CACHE_TTL", "").strip() or None

    @property
    def _response_format_mode(self) -> _ResponseFormatMode:
        return self._response_format_modes.get(self.model_name, "json_schema")

    @_response_format_mode.setter
    def _response_format_mode(self, mode: _ResponseFormatMode) -> None:
        self._response_format_modes[self.model_name] = mode

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
        response_format / structured-outputs requirement.

        String matching is used because OpenRouter forwards upstream provider
        errors verbatim (often wrapped) without stable machine-readable codes.
        A false positive only costs one extra request in a weaker mode, and the
        demotion is memoized per model.
        """
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

    @staticmethod
    def _is_content_block_rejection(error: Exception) -> bool:
        """Heuristically detect provider errors caused by the content-block /
        cache_control message shape, as opposed to an unrelated 400 (bad model
        slug, malformed schema, quota error).

        Like the response-format heuristics above, this matches the forwarded
        upstream error text. The guard must be *positive*: an unrelated 400 that
        merely happens while sending content blocks must not be mistaken for a
        shape rejection, or the model would be permanently (per-process) memoized
        as cache-incompatible. A false negative only fails the current turn loudly
        (and is recoverable via OPENROUTER_PROMPT_CACHE=false); a false positive
        silently disables caching for the model, so we err toward the former.
        """
        error_str = str(error).lower()
        return any(
            marker in error_str
            for marker in (
                "cache_control",
                "cache control",
                "content block",
                "message content",
                "unsupported content",
                "content must be a string",
                "invalid type for 'content'",
            )
        )

    async def _create_completion(
        self,
        prompt: str,
        api_arguments: Mapping[str, Any],
        *,
        message_content: list[_ContentBlock] | None = None,
    ) -> ChatCompletion:
        """Issue a chat-completion request, transmitting the schema with the
        strongest response-format mode the model supports.

        Modes demote monotonically (json_schema -> json_object -> plain) when a
        provider rejects one, and the working mode is memoized per instance.

        When *message_content* is a content-block list (a prompt-caching split) it
        is used verbatim as the user message content. If a provider rejects the
        content-block shape with a non-format error, the model is memoized as
        block-incompatible and the request is retried with the plain string.
        """
        # Held outside the loop so a content-block rejection can permanently drop
        # back to the plain string for the remainder of this request. Typed Any
        # because the content-block dicts intentionally carry a cache_control key
        # the OpenAI SDK's message-param types reject but pass through to OpenRouter.
        content: Any = message_content if message_content is not None else prompt
        while True:
            mode = self._response_format_mode
            try:
                response: ChatCompletion

                if mode == "json_schema":
                    response = await self._client.chat.completions.create(
                        messages=[{"role": "user", "content": content}],
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
                        messages=[{"role": "user", "content": content}],
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
                            {"role": "user", "content": content},
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

                # Format rejections are handled above; if we failed while sending
                # cache_control content blocks AND the error looks like a shape
                # rejection, the provider doesn't accept that message form. Drop to
                # a plain string, memoize the model, and retry. Unrelated 400s (bad
                # slug, malformed schema, quota) fall through to the raise below so
                # they aren't misattributed to caching.
                if isinstance(content, list) and self._is_content_block_rejection(e):
                    self.logger.warning(
                        f"Model '{self.model_name}' rejected cache-control content"
                        f" blocks ({type(e).__name__}: {e}).\n"
                        f"Falling back to a plain-string prompt for this model."
                    )
                    self._cache_blocks_unsupported.add(self.model_name)
                    content = prompt
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
        # Build the prompt once, then derive the cache split from the builder's
        # sections (using the built string to avoid a second full render). Skipped
        # for raw-string prompts and for models known to reject content blocks.
        message_content: list[_ContentBlock] | None = None
        if isinstance(prompt, PromptBuilder):
            builder = prompt
            prompt = builder.build()
            if (
                self._prompt_caching_enabled
                and self.model_name not in self._cache_blocks_unsupported
            ):
                message_content = _split_prompt_for_caching(
                    builder,
                    prompt,
                    ttl=self._cache_ttl,
                    min_prefix_chars=_MIN_CACHE_PREFIX_CHARS,
                    logger=self.logger,
                    model_name=self.model_name,
                )
                if message_content is not None and self.model_name not in _cache_split_logged:
                    _cache_split_logged.add(self.model_name)
                    self.logger.info(
                        f"Prompt cache: first cache split produced for model"
                        f" '{self.model_name}' ({len(message_content[0]['text'])} prefix chars)."
                    )

        openrouter_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_openrouter_params
        }

        if "max_tokens" not in openrouter_api_arguments and self._completion_max_tokens is not None:
            openrouter_api_arguments["max_tokens"] = self._completion_max_tokens

        t_start = time.time()
        response = await self._create_completion(
            prompt, openrouter_api_arguments, message_content=message_content
        )
        t_end = time.time()

        if response.usage:
            self.logger.trace(response.usage.model_dump_json(indent=2))

        if not response.choices[0].message.content:
            self.logger.warning(
                f"Empty response content from '{self.model_name}';"
                f" treating it as an empty JSON object."
            )

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

        if cached_input_tokens > 0 and self.model_name not in _cache_hit_logged:
            _cache_hit_logged.add(self.model_name)
            self.logger.info(
                f"Prompt cache HIT active: model '{self.model_name}' reported"
                f" {cached_input_tokens} cached input tokens."
            )

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
        # Both the canonical short slugs and the versioned slugs the pinned classes
        # use internally resolve to the pinned classes, so users configuring either
        # form get the correct context-window limits.
        model_mapping: dict[
            str, Callable[[Logger, Tracer, Meter], OpenRouterSchematicGenerator[T]]
        ] = {
            "openai/gpt-4o": lambda logger, tracer, meter: OpenRouterGPT4O[t](  # type: ignore
                logger, tracer, meter
            ),
            "openai/gpt-4o-2024-11-20": lambda logger, tracer, meter: OpenRouterGPT4O[t](  # type: ignore
                logger, tracer, meter
            ),
            "openai/gpt-4o-mini": lambda logger, tracer, meter: OpenRouterGPT4OMini[t](  # type: ignore
                logger, tracer, meter
            ),
            "openai/gpt-4o-mini-2024-07-18": lambda logger, tracer, meter: OpenRouterGPT4OMini[
                t  # type: ignore
            ](logger, tracer, meter),
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
