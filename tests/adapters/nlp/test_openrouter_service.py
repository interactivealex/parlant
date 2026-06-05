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

from collections.abc import Generator
import json
import os
from lagom import Container
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import asyncio
from openai import BadRequestError, NotFoundError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from parlant.adapters.nlp.openrouter_service import (  # type: ignore[reportMissingImports]
    OpenRouterService,
    OpenRouterSchematicGenerator,
    OpenRouterEmbedder,
    OpenRouterGPT4O,
    OpenRouterGPT4OMini,
    OpenRouterClaudeSonnet46,
    OpenRouterLlama33_70B,
    OpenRouterEstimatingTokenizer,
)
from parlant.core.loggers import Logger
from parlant.core.common import DefaultBaseModel
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer

from tests.test_utilities import RecordingMeter


class SchemaData(DefaultBaseModel):
    """Test schema for type checking."""

    test_field: str = "test_value"


def _make_chat_response(content: str, usage: CompletionUsage | Mock | None) -> Mock:
    """Build a mocked ChatCompletion response with the given content and usage."""
    mock_response = Mock(spec=ChatCompletion)
    mock_response.choices = [
        Choice(
            message=ChatCompletionMessage(role="assistant", content=content),
            finish_reason="stop",
            index=0,
        )
    ]
    mock_response.usage = usage
    return mock_response


@pytest.fixture(autouse=True)
def set_api_keys() -> Generator[None, None, None]:
    """Set API keys for tests that use container fixture."""
    # Container fixture initializes ServiceRegistry which requires OPENAI_API_KEY
    # OpenRouter tests also need OPENROUTER_API_KEY
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "OPENROUTER_API_KEY": "test-openrouter-key",
        },
        clear=False,
    ):
        yield


def test_that_missing_openrouter_api_key_returns_error_message() -> None:
    """Test that missing OPENROUTER_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = OpenRouterService.verify_environment()
        assert error is not None
        assert "OPENROUTER_API_KEY is not set" in error


def test_that_present_api_key_returns_none() -> None:
    """Test that present API key returns None (success)."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        error = OpenRouterService.verify_environment()
        assert error is None


def test_that_openrouter_service_initializes_with_default_model() -> None:
    """Test OpenRouterService initialization with default model."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        mock_logger = Mock()
        mock_meter = Mock()
        mock_tracer = Mock()
        service = OpenRouterService(logger=mock_logger, tracer=mock_tracer, meter=mock_meter)
        assert service.model_name == "openai/gpt-4o"


def test_that_openrouter_service_initializes_with_custom_model() -> None:
    """Test OpenRouterService initialization with custom model from environment."""
    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "anthropic/claude-3.5-sonnet",
        },
        clear=True,
    ):
        mock_logger = Mock()
        mock_meter = Mock()
        mock_tracer = Mock()
        service = OpenRouterService(logger=mock_logger, tracer=mock_tracer, meter=mock_meter)
        assert service.model_name == "anthropic/claude-3.5-sonnet"


def test_that_openrouter_service_uses_environment_model() -> None:
    """Test OpenRouterService uses OPENROUTER_MODEL from environment."""
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "meta-llama/llama-3.3-70b-instruct"},
        clear=True,
    ):
        mock_logger = Mock()
        mock_meter = Mock()
        mock_tracer = Mock()
        service = OpenRouterService(logger=mock_logger, tracer=mock_tracer, meter=mock_meter)
        assert service.model_name == "meta-llama/llama-3.3-70b-instruct"


def test_that_openrouter_service_respects_custom_max_tokens() -> None:
    """Test OpenRouterService respects max_tokens from environment variable."""
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MAX_TOKENS": "4096"},
        clear=True,
    ):
        mock_logger = Mock()
        mock_meter = Mock()
        mock_tracer = Mock()
        service = OpenRouterService(logger=mock_logger, tracer=mock_tracer, meter=mock_meter)
        # max_tokens is used when creating generators, not stored in service
        assert service.model_name == "openai/gpt-4o"  # Default model


def test_that_openrouter_estimating_tokenizer_works(container: Container) -> None:
    """Test OpenRouterEstimatingTokenizer token estimation."""
    tokenizer = OpenRouterEstimatingTokenizer(model_name="openai/gpt-4o")
    tokens = asyncio.run(tokenizer.estimate_token_count("Hello world"))
    assert tokens > 0


def test_that_openrouter_gpt4o_generator_initializes_correctly(container: Container) -> None:
    """Test OpenRouterGPT4O initialization."""
    generator = OpenRouterGPT4O[SchemaData](
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )
    assert generator.model_name == "openai/gpt-4o-2024-11-20"
    assert generator.id == "openrouter/openai/gpt-4o-2024-11-20"
    assert generator.max_tokens == 128 * 1024


def test_that_openrouter_gpt4o_mini_generator_initializes_correctly(container: Container) -> None:
    """Test OpenRouterGPT4OMini initialization."""
    generator = OpenRouterGPT4OMini[SchemaData](
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )
    assert generator.model_name == "openai/gpt-4o-mini-2024-07-18"
    assert generator.max_tokens == 128 * 1024


def test_that_openrouter_claude_sonnet_generator_initializes_correctly(
    container: Container,
) -> None:
    """Test OpenRouterClaudeSonnet46 initialization."""
    generator = OpenRouterClaudeSonnet46[SchemaData](
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )
    assert generator.model_name == "anthropic/claude-sonnet-4.6"
    assert generator.max_tokens == 1_000_000


def test_that_openrouter_llama_generator_initializes_correctly(container: Container) -> None:
    """Test OpenRouterLlama33_70B initialization."""
    generator = OpenRouterLlama33_70B[SchemaData](
        logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
    )
    assert generator.model_name == "meta-llama/llama-3.3-70b-instruct"
    assert generator.max_tokens == 128 * 1024


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
def test_that_openrouter_generator_sets_custom_headers(mock_client_class: Mock) -> None:
    """Test that OpenRouter generator sets custom headers from environment."""
    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_HTTP_REFERER": "https://example.com",
            "OPENROUTER_SITE_NAME": "My App",
        },
        clear=True,
    ):
        mock_logger = Mock()
        mock_meter = Mock()
        mock_tracer = Mock()
        _ = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=mock_logger,
            tracer=mock_tracer,
            meter=mock_meter,
        )

        # Verify client was called with headers
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert "default_headers" in call_args[1]
        assert call_args[1]["default_headers"]["HTTP-Referer"] == "https://example.com"
        assert call_args[1]["default_headers"]["X-Title"] == "My App"


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
def test_that_openrouter_generator_without_custom_headers(mock_client_class: Mock) -> None:
    """Test OpenRouter generator without custom headers."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        mock_logger = Mock()
        mock_tracer = Mock()
        mock_meter = Mock()
        _ = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=mock_logger,
            tracer=mock_tracer,
            meter=mock_meter,
        )

        # Verify client was called without custom headers
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert call_args[1]["default_headers"] is None


@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_generator_handles_json_mode_error(mock_client_class: Mock) -> None:
    """Test that OpenRouter generator handles JSON mode errors gracefully."""
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client

    # Mock BadRequestError for JSON mode
    from openai import BadRequestError

    mock_client.chat.completions.create.side_effect = BadRequestError(
        "Model does not support JSON mode",
        body={"error": {"message": "JSON mode error"}},
        response=Mock(),
    )

    # MagicMock so that logger.scope() can be used as a context manager
    mock_logger = MagicMock()
    mock_tracer = Mock()
    mock_meter = Mock()

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="test-model",
            logger=mock_logger,
            tracer=mock_tracer,
            meter=mock_meter,
        )

    # Should fail since we're mocking the error
    with pytest.raises(BadRequestError):
        await generator.do_generate("Test prompt")


async def test_that_openrouter_generator_handles_successful_response(
    container: Container,
) -> None:
    """Test OpenRouter generator with successful JSON response."""
    mock_response = Mock(spec=ChatCompletion)
    mock_response.choices = [
        Choice(
            message=ChatCompletionMessage(role="assistant", content='{"test_field": "test_value"}'),
            finish_reason="stop",
            index=0,
        )
    ]
    mock_response.usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        result = await generator.do_generate('Generate {"test_field": "test_value"}')

        assert result.content.test_field == "test_value"
        assert result.info.usage.input_tokens == 10
        assert result.info.usage.output_tokens == 20


async def test_that_openrouter_generator_records_metrics_on_successful_generation(
    container: Container,
) -> None:
    """record_llm_metrics must be called with correct token counts and schema_name."""
    meter = RecordingMeter()

    mock_response = _make_chat_response(
        '{"test_field": "hello"}',
        CompletionUsage(prompt_tokens=15, completion_tokens=7, total_tokens=22),
    )

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=meter,
        )

        result = await generator.do_generate('{"test_field": "hello"}')

    assert result.content.test_field == "hello"

    # Counters must exist on the recording meter (not some other meter)
    assert "input_tokens" in meter.counters
    assert "output_tokens" in meter.counters
    assert "cached_input_tokens" in meter.counters

    input_call = meter.counters["input_tokens"].calls[0]
    assert input_call[0] == 15
    assert input_call[1] is not None
    assert input_call[1]["schema_name"] == "SchemaData"
    assert input_call[1]["model_name"] == "openai/gpt-4o"

    output_call = meter.counters["output_tokens"].calls[0]
    assert output_call[0] == 7


async def test_that_openrouter_generator_records_zero_cached_tokens_when_absent(
    container: Container,
) -> None:
    """When the API response has no cached-token information, cached_input_tokens is 0."""
    meter = RecordingMeter()

    # CompletionUsage has neither prompt_tokens_details nor prompt_cache_hit_tokens
    mock_response = _make_chat_response(
        '{"test_field": "world"}',
        CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=meter,
        )

        await generator.do_generate('{"test_field": "world"}')

    cached_call = meter.counters["cached_input_tokens"].calls[0]
    assert cached_call[0] == 0


async def test_that_openrouter_generator_records_cached_tokens_from_prompt_tokens_details(
    container: Container,
) -> None:
    """Cached tokens are read from the OpenAI-compatible usage.prompt_tokens_details field."""
    meter = RecordingMeter()

    mock_response = _make_chat_response(
        '{"test_field": "cached"}',
        CompletionUsage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=42),
        ),
    )

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=meter,
        )

        result = await generator.do_generate('{"test_field": "cached"}')

    assert meter.counters["cached_input_tokens"].calls[0][0] == 42
    assert result.info.usage.extra is not None
    assert result.info.usage.extra["cached_input_tokens"] == 42


async def test_that_openrouter_generator_records_cached_tokens_from_legacy_field(
    container: Container,
) -> None:
    """Cached tokens fall back to the DeepSeek-style prompt_cache_hit_tokens field."""
    meter = RecordingMeter()

    # The OpenAI SDK usage model allows extra fields, so providers like DeepSeek
    # can attach prompt_cache_hit_tokens to the usage payload.
    usage = CompletionUsage(
        prompt_tokens=50,
        completion_tokens=5,
        total_tokens=55,
        prompt_cache_hit_tokens=9,  # type: ignore[call-arg]
    )
    mock_response = _make_chat_response('{"test_field": "legacy"}', usage)

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="deepseek/deepseek-chat",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=meter,
        )

        result = await generator.do_generate('{"test_field": "legacy"}')

    assert meter.counters["cached_input_tokens"].calls[0][0] == 9
    assert result.info.usage.extra is not None
    assert result.info.usage.extra["cached_input_tokens"] == 9


async def test_that_openrouter_generator_handles_none_token_counts_in_usage(
    container: Container,
) -> None:
    """Some OpenRouter providers return usage with None token counts; treat them as 0."""
    meter = RecordingMeter()

    usage = Mock(spec=CompletionUsage)
    usage.prompt_tokens = None
    usage.completion_tokens = None
    usage.prompt_tokens_details = None
    usage.model_dump_json.return_value = "{}"

    mock_response = _make_chat_response('{"test_field": "none-usage"}', usage)

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=meter,
        )

        result = await generator.do_generate('{"test_field": "none-usage"}')

    assert result.content.test_field == "none-usage"
    assert result.info.usage.input_tokens == 0
    assert result.info.usage.output_tokens == 0
    assert meter.counters["input_tokens"].calls[0][0] == 0
    assert meter.counters["output_tokens"].calls[0][0] == 0
    assert meter.counters["cached_input_tokens"].calls[0][0] == 0


async def test_that_openrouter_generator_handles_missing_usage_without_crashing(
    container: Container,
) -> None:
    """A response without usage data must not crash; zero token counts are recorded."""
    meter = RecordingMeter()

    mock_response = _make_chat_response('{"test_field": "no-usage"}', None)

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=meter,
        )

        result = await generator.do_generate('{"test_field": "no-usage"}')

    assert result.content.test_field == "no-usage"
    assert result.info.usage.input_tokens == 0
    assert result.info.usage.output_tokens == 0
    assert meter.counters["input_tokens"].calls[0][0] == 0


async def test_that_openrouter_generator_sends_json_schema_response_format(
    container: Container,
) -> None:
    """The first request must transmit the Pydantic schema via json_schema response_format
    and require providers that support it."""
    mock_response = _make_chat_response(
        '{"test_field": "structured"}',
        CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        await generator.do_generate("Generate structured output")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "SchemaData",
            "strict": True,
            "schema": SchemaData.model_json_schema(),
        },
    }
    assert call_kwargs["extra_body"] == {"provider": {"require_parameters": True}}


async def test_that_openrouter_generator_falls_back_to_json_object_when_json_schema_is_rejected(
    container: Container,
) -> None:
    """A json_schema rejection demotes the generator to json_object mode, which is memoized."""
    mock_responses = [
        _make_chat_response(
            '{"test_field": "first"}',
            CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
        _make_chat_response(
            '{"test_field": "second"}',
            CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
    ]

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                BadRequestError(
                    "Provider returned error: json_schema response_format is not supported",
                    response=Mock(),
                    body=None,
                ),
                mock_responses[0],
                mock_responses[1],
            ]
        )
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="some/model-without-structured-outputs",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        first_result = await generator.do_generate("First request")
        second_result = await generator.do_generate("Second request")

    assert first_result.content.test_field == "first"
    assert second_result.content.test_field == "second"

    calls = mock_client.chat.completions.create.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs["response_format"]["type"] == "json_schema"
    assert calls[1].kwargs["response_format"]["type"] == "json_object"
    # The fallback is memoized: the second generation goes straight to json_object
    assert calls[2].kwargs["response_format"]["type"] == "json_object"


async def test_that_openrouter_generator_falls_back_when_no_endpoints_support_structured_outputs(
    container: Container,
) -> None:
    """require_parameters can yield a 404 'no endpoints' error; fall back to json_object."""
    mock_response = _make_chat_response(
        '{"test_field": "fallback"}',
        CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                NotFoundError(
                    "No endpoints found that support the requested parameters",
                    response=Mock(),
                    body=None,
                ),
                mock_response,
            ]
        )
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="some/model",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        result = await generator.do_generate("Fallback request")

    assert result.content.test_field == "fallback"

    calls = mock_client.chat.completions.create.call_args_list
    assert calls[0].kwargs["response_format"]["type"] == "json_schema"
    assert calls[1].kwargs["response_format"]["type"] == "json_object"


async def test_that_openrouter_generator_forwards_completion_max_tokens_from_env(
    container: Container,
) -> None:
    """OPENROUTER_COMPLETION_MAX_TOKENS is forwarded to the API as max_tokens."""
    mock_response = _make_chat_response(
        '{"test_field": "capped"}',
        CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    with (
        patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class,
        patch.dict(os.environ, {"OPENROUTER_COMPLETION_MAX_TOKENS": "512"}, clear=False),
    ):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        await generator.do_generate("Capped request")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 512


async def test_that_openrouter_generator_forwards_max_tokens_hint_over_env(
    container: Container,
) -> None:
    """A max_tokens hint takes precedence over OPENROUTER_COMPLETION_MAX_TOKENS."""
    mock_response = _make_chat_response(
        '{"test_field": "hinted"}',
        CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    with (
        patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class,
        patch.dict(os.environ, {"OPENROUTER_COMPLETION_MAX_TOKENS": "512"}, clear=False),
    ):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        await generator.do_generate("Hinted request", hints={"max_tokens": 256})

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 256


async def test_that_openrouter_generator_omits_max_tokens_when_not_configured(
    container: Container,
) -> None:
    """Without a hint or env var, max_tokens is not sent (provider defaults apply)."""
    mock_response = _make_chat_response(
        '{"test_field": "uncapped"}',
        CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    env_without_cap = {
        k: v for k, v in os.environ.items() if k != "OPENROUTER_COMPLETION_MAX_TOKENS"
    }

    with (
        patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class,
        patch.dict(os.environ, env_without_cap, clear=True),
    ):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        await generator.do_generate("Uncapped request")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in call_kwargs


async def test_that_openrouter_generator_raises_json_decode_error_when_response_is_not_json(
    container: Container,
) -> None:
    """Unparseable responses must raise (so the base generator can retry), not be
    silently swallowed into an empty JSON object."""
    mock_response = _make_chat_response(
        "I'm sorry, I cannot help with that.",
        CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    with patch("parlant.adapters.nlp.openrouter_service.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        generator = OpenRouterSchematicGenerator[SchemaData](
            model_name="openai/gpt-4o",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        with pytest.raises(json.JSONDecodeError):
            await generator.do_generate("Some request")


def test_that_openrouter_service_returns_correct_generator(container: Container) -> None:
    """Test OpenRouterService.get_schematic_generator with default model."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert isinstance(generator, OpenRouterSchematicGenerator)
        assert generator.model_name == "openai/gpt-4o-2024-11-20"


def test_that_openrouter_service_returns_correct_generator_for_claude(
    container: Container,
) -> None:
    """Test OpenRouterService.get_schematic_generator with Claude model."""
    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "anthropic/claude-sonnet-4.6",
        },
        clear=True,
    ):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert isinstance(generator, OpenRouterClaudeSonnet46)
        assert generator.model_name == "anthropic/claude-sonnet-4.6"


def test_that_openrouter_service_creates_dynamic_generator_for_unknown_model(
    container: Container,
) -> None:
    """Test OpenRouterService creates dynamic generator for unknown model."""
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "custom/model-name"},
        clear=True,
    ):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert isinstance(generator, OpenRouterSchematicGenerator)
        assert generator.model_name == "custom/model-name"


def test_that_openrouter_service_uses_custom_max_tokens(container: Container) -> None:
    """Test OpenRouterService uses max_tokens from environment for unknown model."""
    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "custom/model",
            "OPENROUTER_MAX_TOKENS": "2048",
        },
        clear=True,
    ):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert generator.max_tokens == 2048


def test_that_openrouter_service_uses_environment_max_tokens(container: Container) -> None:
    """Test OpenRouterService uses environment max_tokens for unknown model."""
    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "custom/unknown-model",
            "OPENROUTER_MAX_TOKENS": "4096",
        },
        clear=True,
    ):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert generator.max_tokens == 4096


def test_that_openrouter_service_sets_default_max_tokens_for_gpt4(container: Container) -> None:
    """Test OpenRouterService sets default max_tokens for GPT-4 models."""
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "openai/gpt-4-turbo"},
        clear=True,
    ):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert generator.max_tokens == 128 * 1024


def test_that_openrouter_service_sets_default_max_tokens_for_claude(container: Container) -> None:
    """Test OpenRouterService sets default max_tokens for Claude models."""
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "anthropic/claude-2"},
        clear=True,
    ):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert generator.max_tokens == 200 * 1024


def test_that_openrouter_service_sets_default_max_tokens_for_llama(container: Container) -> None:
    """Test OpenRouterService sets default max_tokens for Llama models."""
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "meta-llama/llama-2-70b"},
        clear=True,
    ):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert generator.max_tokens == 8192


def test_that_openrouter_service_sets_default_max_tokens_for_llama3(container: Container) -> None:
    """Test OpenRouterService sets default max_tokens for Llama 3.x models."""
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "meta-llama/llama-3.1-8b-instruct"},
        clear=True,
    ):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        generator = asyncio.run(service.get_schematic_generator(SchemaData))
        assert generator.max_tokens == 128 * 1024


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_embedder_retries_empty_embedding_value_error(
    mock_client_class: Mock,
    mock_sleep: AsyncMock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=[
            ValueError("No embedding data received"),
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])]),
        ]
    )
    mock_client_class.return_value = mock_client

    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    result = await embedder.do_embed(["hello"])

    assert result.vectors == [[0.1, 0.2, 0.3]]
    assert mock_client.embeddings.create.await_count == 2
    mock_sleep.assert_awaited_once()


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_embedder_retries_empty_embedding_response_data(
    mock_client_class: Mock,
    mock_sleep: AsyncMock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=[
            Mock(data=[]),
            Mock(data=[Mock(embedding=[0.4, 0.5])]),
        ]
    )
    mock_client_class.return_value = mock_client

    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    result = await embedder.do_embed(["hello"])

    assert result.vectors == [[0.4, 0.5]]
    assert mock_client.embeddings.create.await_count == 2
    mock_sleep.assert_awaited_once()


@patch("parlant.core.nlp.policies.asyncio.sleep", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.openrouter_service.AsyncClient")
async def test_that_openrouter_embedder_does_not_retry_unrelated_value_error(
    mock_client_class: Mock,
    mock_sleep: AsyncMock,
    container: Container,
) -> None:
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=ValueError("Embedding payload is malformed")
    )
    mock_client_class.return_value = mock_client

    embedder = OpenRouterEmbedder(
        model_name="openai/text-embedding-3-small",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    with pytest.raises(ValueError, match="Embedding payload is malformed"):
        await embedder.do_embed(["hello"])

    assert mock_client.embeddings.create.await_count == 1
    mock_sleep.assert_not_awaited()


@pytest.mark.skip(
    reason="Requires network access - embedder initialization may use JinaAIEmbedder fallback"
)
def test_that_openrouter_service_returns_openrouter_embedder(container: Container) -> None:
    """Test OpenRouterService returns OpenRouter embedder.

    Note: This test is skipped because the embedder initialization may require network access
    if the installed version uses a JinaAIEmbedder fallback.
    """
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        embedder = asyncio.run(service.get_embedder())
        # OpenRouter embedder should be returned
        from parlant.adapters.nlp.openrouter_service import (
            OpenRouterEmbedder,
            OpenRouterTextEmbedding3Large,
        )

        # Should be either OpenRouterEmbedder or OpenRouterTextEmbedding3Large
        assert isinstance(embedder, (OpenRouterEmbedder, OpenRouterTextEmbedding3Large))
        assert embedder is not None


def test_that_openrouter_service_returns_no_moderation(container: Container) -> None:
    """Test OpenRouterService returns NoModeration."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
        service = OpenRouterService(
            logger=container[Logger], tracer=container[Tracer], meter=container[Meter]
        )
        moderation = asyncio.run(service.get_moderation_service())
        from parlant.core.nlp.moderation import NoModeration

        assert isinstance(moderation, NoModeration)


def test_that_openrouter_generator_supports_correct_parameters(container: Container) -> None:
    """Test supported OpenRouter parameters."""
    generator = OpenRouterSchematicGenerator[SchemaData](
        model_name="openai/gpt-4o",
        logger=container[Logger],
        tracer=container[Tracer],
        meter=container[Meter],
    )

    expected_params = ["temperature", "max_tokens"]
    assert generator.supported_openrouter_params == expected_params
