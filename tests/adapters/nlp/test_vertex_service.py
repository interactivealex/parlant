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

import asyncio
import os
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parlant.adapters.nlp.vertex_service import (
    VertexGemini20Flash,
    VertexClaudeSonnet4,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import StdoutLogger
from parlant.core.meter import LocalMeter
from parlant.core.tracer import LocalTracer


class SimpleSchema(DefaultBaseModel):
    value: str = "hello"
    count: Optional[int] = None


@pytest.fixture
def logger() -> StdoutLogger:
    tracer = LocalTracer()
    return StdoutLogger(tracer)


@pytest.fixture
def tracer() -> LocalTracer:
    return LocalTracer()


@pytest.fixture
def meter(logger: StdoutLogger) -> LocalMeter:
    return LocalMeter(logger)


def _make_gemini_usage(
    prompt: int = 10,
    candidates: int = 20,
    cached: int = 0,
    thoughts: int = 0,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_token_count = prompt
    usage.candidates_token_count = candidates
    usage.cached_content_token_count = cached
    usage.thoughts_token_count = thoughts
    usage.model_dump_json = MagicMock(return_value="{}")
    return usage


def _make_gemini_response(
    parsed_value: object = None,
    text: Optional[str] = None,
    usage: Optional[MagicMock] = None,
) -> MagicMock:
    response = MagicMock()
    response.parsed = parsed_value
    response.text = text
    response.usage_metadata = usage if usage is not None else _make_gemini_usage()
    return response


@patch("parlant.adapters.nlp.vertex_service.google.genai.Client")
def test_that_vertex_gemini_config_uses_pydantic_class_as_response_schema(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """VertexAIGeminiSchematicGenerator must pass the Pydantic class (not a dict) as response_schema."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="vertex_parsed", count=3)
    mock_response = _make_gemini_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(
        os.environ,
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_REGION": "us-central1"},
        clear=False,
    ):
        generator = VertexGemini20Flash[SimpleSchema](  # type: ignore[misc]
            project_id="test-project",
            region="us-central1",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is SimpleSchema
    assert result.content.value == "vertex_parsed"


@patch("parlant.adapters.nlp.vertex_service.google.genai.Client")
def test_that_vertex_gemini_uses_parsed_when_available(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """When response.parsed is set, use it directly."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="direct_parsed", count=1)
    mock_response = _make_gemini_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(
        os.environ,
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_REGION": "us-central1"},
        clear=False,
    ):
        generator = VertexGemini20Flash[SimpleSchema](  # type: ignore[misc]
            project_id="test-project",
            region="us-central1",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    assert result.content.value == "direct_parsed"
    assert result.content.count == 1


@patch("parlant.adapters.nlp.vertex_service.google.genai.Client")
def test_that_vertex_gemini_falls_back_to_model_validate_json_when_parsed_is_none(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """When response.parsed is None, use model_validate_json(response.text) as fallback."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = _make_gemini_response(
        parsed_value=None,
        text='{"value": "json_fallback", "count": 55}',
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(
        os.environ,
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_REGION": "us-central1"},
        clear=False,
    ):
        generator = VertexGemini20Flash[SimpleSchema](  # type: ignore[misc]
            project_id="test-project",
            region="us-central1",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    assert result.content.value == "json_fallback"
    assert result.content.count == 55


@patch("parlant.adapters.nlp.vertex_service.record_llm_metrics", new_callable=AsyncMock)
@patch("parlant.adapters.nlp.vertex_service.google.genai.Client")
def test_that_vertex_gemini_records_llm_metrics(
    mock_client_class: MagicMock,
    mock_record_metrics: AsyncMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """VertexAIGeminiSchematicGenerator must call record_llm_metrics after successful generation."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="metrics")
    usage = _make_gemini_usage(prompt=30, candidates=40, cached=5)
    mock_response = _make_gemini_response(parsed_value=parsed_instance, usage=usage)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(
        os.environ,
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_REGION": "us-central1"},
        clear=False,
    ):
        generator = VertexGemini20Flash[SimpleSchema](  # type: ignore[misc]
            project_id="test-project",
            region="us-central1",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        asyncio.run(generator.do_generate("test prompt"))

    mock_record_metrics.assert_called_once()
    call_kwargs = mock_record_metrics.call_args.kwargs
    assert call_kwargs["schema_name"] == "SimpleSchema"
    assert call_kwargs["input_tokens"] == 30
    assert call_kwargs["output_tokens"] == 40


@patch("parlant.adapters.nlp.vertex_service.google.genai.Client")
def test_that_vertex_gemini_forwards_max_output_tokens_from_hints(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """max_output_tokens must be set from hints['max_tokens'] when provided."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="tok")
    mock_response = _make_gemini_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(
        os.environ,
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_REGION": "us-central1"},
        clear=False,
    ):
        generator = VertexGemini20Flash[SimpleSchema](  # type: ignore[misc]
            project_id="test-project",
            region="us-central1",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        asyncio.run(generator.do_generate("test prompt", hints={"max_tokens": 1024}))

    config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == 1024


@patch("parlant.adapters.nlp.vertex_service.google.genai.Client")
def test_that_vertex_gemini_does_not_set_max_output_tokens_when_not_hinted(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """max_output_tokens must not appear in config when hints['max_tokens'] is absent."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="notok")
    mock_response = _make_gemini_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(
        os.environ,
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_REGION": "us-central1"},
        clear=False,
    ):
        generator = VertexGemini20Flash[SimpleSchema](  # type: ignore[misc]
            project_id="test-project",
            region="us-central1",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        asyncio.run(generator.do_generate("test prompt"))

    config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens is None


@patch("parlant.adapters.nlp.vertex_service.AsyncAnthropicVertex")
@patch("parlant.adapters.nlp.vertex_service.google.genai.Client")
def test_that_vertex_claude_uses_messages_parse_for_structured_output(
    mock_genai_client_class: MagicMock,
    mock_anthropic_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """VertexAIClaudeSchematicGenerator must use messages.parse with output_format=<PydanticClass>."""
    mock_genai_client = MagicMock()
    mock_genai_client_class.return_value = mock_genai_client

    mock_anthropic_client = MagicMock()
    mock_anthropic_class.return_value = mock_anthropic_client

    parsed_instance = SimpleSchema(value="claude_native", count=8)
    mock_parsed_message = MagicMock()
    mock_parsed_message.parsed_output = parsed_instance
    mock_parsed_message.usage.input_tokens = 15
    mock_parsed_message.usage.output_tokens = 25
    mock_anthropic_client.messages.parse = AsyncMock(return_value=mock_parsed_message)

    with patch.dict(
        os.environ,
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_REGION": "us-east5"},
        clear=False,
    ):
        generator = VertexClaudeSonnet4[SimpleSchema](  # type: ignore[misc]
            project_id="test-project",
            region="us-east5",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    mock_anthropic_client.messages.parse.assert_called_once()
    call_kwargs = mock_anthropic_client.messages.parse.call_args.kwargs
    assert call_kwargs["output_format"] is SimpleSchema
    assert result.content.value == "claude_native"
    assert result.content.count == 8


@patch("parlant.adapters.nlp.vertex_service.AsyncAnthropicVertex")
@patch("parlant.adapters.nlp.vertex_service.google.genai.Client")
def test_that_vertex_claude_uses_legacy_path_when_sdk_lacks_parse(
    mock_genai_client_class: MagicMock,
    mock_anthropic_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """When the SDK has no messages.parse (detected once at init), the legacy
    messages.create path is used."""
    mock_genai_client = MagicMock()
    mock_genai_client_class.return_value = mock_genai_client

    # spec-limited messages object: create exists, parse does not.
    mock_anthropic_client = MagicMock()
    mock_anthropic_class.return_value = mock_anthropic_client
    mock_anthropic_client.messages = MagicMock(spec=["create"])

    mock_legacy_response = MagicMock()
    mock_legacy_response.content = [MagicMock()]
    mock_legacy_response.content[0].text = '{"value": "legacy_claude", "count": 0}'
    mock_legacy_response.usage.input_tokens = 10
    mock_legacy_response.usage.output_tokens = 15
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_legacy_response)

    with patch.dict(
        os.environ,
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_REGION": "us-east5"},
        clear=False,
    ):
        generator = VertexClaudeSonnet4[SimpleSchema](  # type: ignore[misc]
            project_id="test-project",
            region="us-east5",
            logger=logger,
            tracer=tracer,
            meter=meter,
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    assert result.content.value == "legacy_claude"
