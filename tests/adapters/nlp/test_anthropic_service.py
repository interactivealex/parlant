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

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from anthropic import BadRequestError
from anthropic.types import Message, Usage
from anthropic.types.parsed_message import ParsedMessage, ParsedTextBlock

from parlant.adapters.nlp.anthropic_service import (
    AnthropicAISchematicGenerator,
    Claude_Sonnet_3_5,
    Claude_Sonnet_4,
    Claude_Opus_4_1,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleSchema(DefaultBaseModel):
    """Simple test schema."""

    answer: str


def _make_parsed_message(
    parsed: SimpleSchema,
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_read_input_tokens: int | None = None,
) -> ParsedMessage[SimpleSchema]:
    """Build a minimal ParsedMessage mock that matches SDK expectations."""
    text_block: Any = MagicMock(spec=ParsedTextBlock)
    text_block.type = "text"
    text_block.text = parsed.model_dump_json()
    text_block.parsed_output = parsed

    usage_kwargs: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
    }

    raw_msg = MagicMock(spec=Message)
    raw_msg.content = [text_block]
    raw_msg.usage = Usage(**usage_kwargs)
    raw_msg.parsed_output = parsed

    return raw_msg  # type: ignore[return-value]


def _make_generator(
    container: Any,
    model_name: str = "claude-sonnet-4-20250514",
) -> AnthropicAISchematicGenerator[SimpleSchema]:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        return Claude_Sonnet_4[SimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )


# ---------------------------------------------------------------------------
# Tests — structured-output path sends schema
# ---------------------------------------------------------------------------


async def test_that_generate_sends_output_format_schema_to_parse(container: Any) -> None:
    """messages.parse is called with output_format=SimpleSchema (schema transmitted)."""
    expected = SimpleSchema(answer="42")
    parsed_msg = _make_parsed_message(expected)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        generator = _make_generator(container)

        with patch.object(
            generator._client.messages, "parse", new=AsyncMock(return_value=parsed_msg)
        ) as mock_parse:
            result = await generator.do_generate("What is the answer?")

    mock_parse.assert_called_once()
    call_kwargs = mock_parse.call_args.kwargs
    assert call_kwargs.get("output_format") is SimpleSchema
    assert result.content.answer == "42"


# ---------------------------------------------------------------------------
# Tests — parsed result returned and validated
# ---------------------------------------------------------------------------


async def test_that_generate_returns_validated_schema_instance(container: Any) -> None:
    """The returned SchematicGenerationResult.content is a validated SimpleSchema."""
    expected = SimpleSchema(answer="hello")
    parsed_msg = _make_parsed_message(expected)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        generator = _make_generator(container)

        with patch.object(
            generator._client.messages, "parse", new=AsyncMock(return_value=parsed_msg)
        ):
            result = await generator.do_generate("Say hello")

    assert isinstance(result.content, SimpleSchema)
    assert result.content.answer == "hello"
    assert result.info.usage.input_tokens == 10
    assert result.info.usage.output_tokens == 20


# ---------------------------------------------------------------------------
# Tests — schema-rejection 400 falls back to legacy scraping
# ---------------------------------------------------------------------------


async def test_that_schema_rejection_400_falls_back_to_legacy_path(container: Any) -> None:
    """A 400 BadRequestError referencing schema/format causes fallback to legacy scraping."""
    bad_request = BadRequestError(
        message="Invalid schema format",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
        body={"error": {"type": "invalid_request_error", "message": "Invalid schema format"}},
    )

    legacy_message = MagicMock(spec=Message)
    legacy_text_block = MagicMock()
    legacy_text_block.type = "text"
    legacy_text_block.text = '{"answer": "legacy"}'
    legacy_message.content = [legacy_text_block]
    legacy_message.usage = Usage(input_tokens=5, output_tokens=10)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        generator = _make_generator(container)

        with (
            patch.object(
                generator._client.messages, "parse", new=AsyncMock(side_effect=bad_request)
            ),
            patch.object(
                generator._client.messages, "create", new=AsyncMock(return_value=legacy_message)
            ),
        ):
            result = await generator.do_generate("Generate answer")

    assert isinstance(result.content, SimpleSchema)
    assert result.content.answer == "legacy"


async def test_that_schema_rejection_caches_fallback_decision(container: Any) -> None:
    """After the first schema-rejection, subsequent calls skip parse() entirely."""
    bad_request = BadRequestError(
        message="Invalid schema format",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
        body={"error": {"type": "invalid_request_error", "message": "Invalid schema format"}},
    )

    legacy_message = MagicMock(spec=Message)
    legacy_text_block = MagicMock()
    legacy_text_block.type = "text"
    legacy_text_block.text = '{"answer": "legacy"}'
    legacy_message.content = [legacy_text_block]
    legacy_message.usage = Usage(input_tokens=5, output_tokens=10)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        generator = _make_generator(container)

        mock_parse = AsyncMock(side_effect=bad_request)
        mock_create = AsyncMock(return_value=legacy_message)

        with (
            patch.object(generator._client.messages, "parse", new=mock_parse),
            patch.object(generator._client.messages, "create", new=mock_create),
        ):
            await generator.do_generate("First call")
            await generator.do_generate("Second call — should skip parse")

    # parse() called only once; second call goes straight to legacy
    assert mock_parse.call_count == 1
    assert mock_create.call_count == 2


# ---------------------------------------------------------------------------
# Tests — max_tokens
# ---------------------------------------------------------------------------


def test_that_claude_sonnet_35_has_correct_output_max_tokens(container: Any) -> None:
    """Claude-3-5-sonnet uses 8192 output max_tokens."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        gen = Claude_Sonnet_3_5[SimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
    assert gen.output_max_tokens == 8192


def test_that_claude_sonnet_4_has_correct_output_max_tokens(container: Any) -> None:
    """Claude-sonnet-4 uses 32000 output max_tokens."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        gen = Claude_Sonnet_4[SimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
    assert gen.output_max_tokens == 32000


def test_that_claude_opus_4_1_has_correct_output_max_tokens(container: Any) -> None:
    """Claude-opus-4-1 uses 32000 output max_tokens."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        gen = Claude_Opus_4_1[SimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
    assert gen.output_max_tokens == 32000


async def test_that_max_tokens_hint_overrides_default(container: Any) -> None:
    """hints['max_tokens'] is forwarded as max_tokens in the API call."""
    expected = SimpleSchema(answer="ok")
    parsed_msg = _make_parsed_message(expected)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        generator = _make_generator(container)

        with patch.object(
            generator._client.messages, "parse", new=AsyncMock(return_value=parsed_msg)
        ) as mock_parse:
            await generator.do_generate("prompt", hints={"max_tokens": 1024})

    call_kwargs = mock_parse.call_args.kwargs
    assert call_kwargs.get("max_tokens") == 1024


async def test_that_default_output_max_tokens_used_when_no_hint(container: Any) -> None:
    """Without a hint, output_max_tokens is used as max_tokens."""
    expected = SimpleSchema(answer="ok")
    parsed_msg = _make_parsed_message(expected)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        gen35 = Claude_Sonnet_3_5[SimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        with patch.object(
            gen35._client.messages, "parse", new=AsyncMock(return_value=parsed_msg)
        ) as mock_parse:
            await gen35.do_generate("prompt")

    call_kwargs = mock_parse.call_args.kwargs
    assert call_kwargs.get("max_tokens") == 8192


# ---------------------------------------------------------------------------
# Tests — cache_read_input_tokens in metrics and UsageInfo.extra
# ---------------------------------------------------------------------------


async def test_that_cache_read_tokens_appear_in_usage_info_extra(container: Any) -> None:
    """cache_read_input_tokens from response.usage lands in UsageInfo.extra."""
    expected = SimpleSchema(answer="cached")
    parsed_msg = _make_parsed_message(expected, cache_read_input_tokens=300)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        generator = _make_generator(container)

        with patch.object(
            generator._client.messages, "parse", new=AsyncMock(return_value=parsed_msg)
        ):
            result = await generator.do_generate("cached prompt")

    assert result.info.usage.extra is not None
    assert result.info.usage.extra.get("cached_input_tokens") == 300


async def test_that_none_cache_tokens_are_handled_gracefully(container: Any) -> None:
    """When cache_read_input_tokens is None, extra is absent or has 0."""
    expected = SimpleSchema(answer="no-cache")
    parsed_msg = _make_parsed_message(expected, cache_read_input_tokens=None)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        generator = _make_generator(container)

        with patch.object(
            generator._client.messages, "parse", new=AsyncMock(return_value=parsed_msg)
        ):
            result = await generator.do_generate("no-cache prompt")

    # Either no extra key or value is 0 — must not raise
    cached = (result.info.usage.extra or {}).get("cached_input_tokens", 0)
    assert cached == 0


async def test_that_cache_tokens_are_recorded_in_metrics(container: Any) -> None:
    """cached_input_tokens are forwarded to record_llm_metrics."""
    expected = SimpleSchema(answer="metrics")
    parsed_msg = _make_parsed_message(expected, cache_read_input_tokens=150)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        generator = _make_generator(container)

        with (
            patch.object(
                generator._client.messages, "parse", new=AsyncMock(return_value=parsed_msg)
            ),
            patch(
                "parlant.adapters.nlp.anthropic_service.record_llm_metrics",
                new=AsyncMock(),
            ) as mock_metrics,
        ):
            await generator.do_generate("metrics prompt")

    mock_metrics.assert_called_once()
    call_kwargs = mock_metrics.call_args.kwargs
    assert call_kwargs.get("cached_input_tokens") == 150
