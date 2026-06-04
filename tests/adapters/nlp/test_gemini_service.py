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

from parlant.adapters.nlp.gemini_service import Gemini_2_0_Flash
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


def _make_usage(
    prompt: int = 10, candidates: int = 20, cached: int = 0, thoughts: int = 0
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_token_count = prompt
    usage.candidates_token_count = candidates
    usage.cached_content_token_count = cached
    usage.thoughts_token_count = thoughts
    usage.model_dump_json = MagicMock(return_value="{}")
    return usage


def _make_response(
    parsed_value: object = None,
    text: Optional[str] = None,
    usage: Optional[MagicMock] = None,
) -> MagicMock:
    """Build a mock GenerateContentResponse."""
    response = MagicMock()
    response.parsed = parsed_value
    response.text = text
    response.usage_metadata = usage if usage is not None else _make_usage()
    return response


@patch("parlant.adapters.nlp.gemini_service.google.genai.Client")
def test_that_gemini_generator_config_uses_response_mime_type_and_pydantic_schema(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """GenerateContentConfig must carry response_mime_type and the Pydantic class (not a dict)."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="world", count=42)
    mock_response = _make_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        generator = Gemini_2_0_Flash[SimpleSchema](  # type: ignore[misc]
            logger=logger, tracer=tracer, meter=meter
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    mock_client.aio.models.generate_content.assert_called_once()
    config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    # response_schema must be the Pydantic class, not a dict
    assert config.response_schema is SimpleSchema
    assert result.content.value == "world"
    assert result.content.count == 42


@patch("parlant.adapters.nlp.gemini_service.google.genai.Client")
def test_that_gemini_generator_uses_parsed_when_available(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """When response.parsed is populated the generator must use it directly."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="from_parsed", count=7)
    mock_response = _make_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        generator = Gemini_2_0_Flash[SimpleSchema](  # type: ignore[misc]
            logger=logger, tracer=tracer, meter=meter
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    assert result.content.value == "from_parsed"
    assert result.content.count == 7


@patch("parlant.adapters.nlp.gemini_service.google.genai.Client")
def test_that_gemini_generator_falls_back_to_model_validate_json_when_parsed_is_none(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """When response.parsed is None the generator must fall back to model_validate_json(response.text)."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = _make_response(
        parsed_value=None,
        text='{"value": "fallback_value", "count": 99}',
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        generator = Gemini_2_0_Flash[SimpleSchema](  # type: ignore[misc]
            logger=logger, tracer=tracer, meter=meter
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    assert result.content.value == "fallback_value"
    assert result.content.count == 99


@patch("parlant.adapters.nlp.gemini_service.google.genai.Client")
def test_that_gemini_generator_does_not_mutate_schema_class(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """The Pydantic schema class must not be mutated (no _conversion_cache attribute added)."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="clean")
    mock_response = _make_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        generator = Gemini_2_0_Flash[SimpleSchema](  # type: ignore[misc]
            logger=logger, tracer=tracer, meter=meter
        )
        asyncio.run(generator.do_generate("test prompt"))

    # The old tool-forcing code added _conversion_cache to the schema class.
    assert not hasattr(SimpleSchema, "_conversion_cache"), (
        "_conversion_cache must not be set on the schema class after generation"
    )


@patch("parlant.adapters.nlp.gemini_service.google.genai.Client")
def test_that_gemini_generator_forwards_max_output_tokens_from_hints(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """max_output_tokens in GenerateContentConfig must be set from hints['max_tokens'] when provided."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="tok")
    mock_response = _make_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        generator = Gemini_2_0_Flash[SimpleSchema](  # type: ignore[misc]
            logger=logger, tracer=tracer, meter=meter
        )
        asyncio.run(generator.do_generate("test prompt", hints={"max_tokens": 512}))

    config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == 512


@patch("parlant.adapters.nlp.gemini_service.google.genai.Client")
def test_that_gemini_generator_does_not_set_max_output_tokens_when_not_hinted(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """max_output_tokens must not be included in the config when hints['max_tokens'] is absent."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="notok")
    mock_response = _make_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        generator = Gemini_2_0_Flash[SimpleSchema](  # type: ignore[misc]
            logger=logger, tracer=tracer, meter=meter
        )
        asyncio.run(generator.do_generate("test prompt"))

    config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens is None


@patch("parlant.adapters.nlp.gemini_service.google.genai.Client")
def test_that_gemini_generator_includes_thoughts_token_count_in_usage_extra(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """thoughts_token_count from usage_metadata must appear in UsageInfo.extra."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    usage = _make_usage(prompt=10, candidates=20, cached=0, thoughts=5)
    parsed_instance = SimpleSchema(value="think")
    mock_response = _make_response(parsed_value=parsed_instance, usage=usage)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        generator = Gemini_2_0_Flash[SimpleSchema](  # type: ignore[misc]
            logger=logger, tracer=tracer, meter=meter
        )
        result = asyncio.run(generator.do_generate("test prompt"))

    assert result.info.usage.extra is not None
    assert result.info.usage.extra.get("thoughts_token_count") == 5


@patch("parlant.adapters.nlp.gemini_service.google.genai.Client")
def test_that_gemini_generator_no_tool_forcing_machinery(
    mock_client_class: MagicMock,
    logger: StdoutLogger,
    tracer: LocalTracer,
    meter: LocalMeter,
) -> None:
    """No Tool / FunctionDeclaration should be passed in the config."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    parsed_instance = SimpleSchema(value="no_tools")
    mock_response = _make_response(parsed_value=parsed_instance)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        generator = Gemini_2_0_Flash[SimpleSchema](  # type: ignore[misc]
            logger=logger, tracer=tracer, meter=meter
        )
        asyncio.run(generator.do_generate("test prompt"))

    config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
    # tools and tool_config must be absent / None (no fake function-calling machinery)
    assert getattr(config, "tools", None) is None
    assert getattr(config, "tool_config", None) is None
