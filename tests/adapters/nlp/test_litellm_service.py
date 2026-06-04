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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock

from lagom import Container

from parlant.adapters.nlp.litellm_service import (
    LiteLLMEmbedder,
    LiteLLMEstimatingTokenizer,
    LiteLLMService,
    LiteLLM_Default,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import Meter, LocalMeter
from parlant.core.tracer import Tracer, LocalTracer

import pytest


class SampleSchema(DefaultBaseModel):
    value: str = "ok"


@pytest.fixture
def container() -> Container:
    from parlant.core.loggers import StdoutLogger
    from parlant.core.tracer import LocalTracer
    from parlant.core.meter import LocalMeter

    container = Container()
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    container[Logger] = logger
    container[Tracer] = tracer
    container[Meter] = meter

    return container


def test_that_missing_model_name_returns_error_message() -> None:
    with patch.dict(os.environ, {}, clear=True):
        error = LiteLLMService.verify_environment()
        assert error is not None
        assert "LITELLM_PROVIDER_MODEL_NAME" in error


def test_that_verify_environment_returns_none_when_model_name_is_set() -> None:
    with patch.dict(
        os.environ,
        {"LITELLM_PROVIDER_MODEL_NAME": "gpt-4"},
        clear=True,
    ):
        error = LiteLLMService.verify_environment()
        assert error is None


def test_that_service_reads_base_url_from_env(container: Container) -> None:
    with patch.dict(
        os.environ,
        {
            "LITELLM_PROVIDER_MODEL_NAME": "gpt-4",
            "LITELLM_PROVIDER_BASE_URL": "http://localhost:8000",
        },
        clear=False,
    ):
        service = LiteLLMService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        assert service._base_url == "http://localhost:8000"


def test_that_service_reads_embedding_model_name_from_env(container: Container) -> None:
    with patch.dict(
        os.environ,
        {
            "LITELLM_PROVIDER_MODEL_NAME": "gpt-4",
            "LITELLM_EMBEDDING_MODEL_NAME": "text-embedding-3-small",
        },
        clear=False,
    ):
        service = LiteLLMService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        assert service._embedding_model_name == "text-embedding-3-small"


def test_that_get_embedder_returns_litellm_embedder_when_embedding_model_configured(
    container: Container,
) -> None:
    with patch.dict(
        os.environ,
        {
            "LITELLM_PROVIDER_MODEL_NAME": "gpt-4",
            "LITELLM_EMBEDDING_MODEL_NAME": "text-embedding-3-small",
        },
        clear=False,
    ):
        service = LiteLLMService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        embedder = asyncio.run(service.get_embedder())

        assert isinstance(embedder, LiteLLMEmbedder)
        assert embedder.model_name == "text-embedding-3-small"


@patch("parlant.adapters.nlp.litellm_service.JinaAIEmbedder")
def test_that_get_embedder_falls_back_to_jina_when_embedding_model_not_configured(
    mock_jina_embedder: Mock, container: Container
) -> None:
    mock_jina_instance = Mock()
    mock_jina_embedder.return_value = mock_jina_instance

    env = {k: v for k, v in os.environ.items() if k != "LITELLM_EMBEDDING_MODEL_NAME"}
    env["LITELLM_PROVIDER_MODEL_NAME"] = "gpt-4"

    with patch.dict(os.environ, env, clear=True):
        service = LiteLLMService(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        embedder = asyncio.run(service.get_embedder())

        assert embedder is mock_jina_instance
        mock_jina_embedder.assert_called_once()


def test_that_embedder_max_tokens_defaults_to_8192(container: Container) -> None:
    env = {k: v for k, v in os.environ.items() if k != "LITELLM_EMBEDDING_MAX_TOKENS"}
    with patch.dict(os.environ, env, clear=True):
        embedder = LiteLLMEmbedder(
            model_name="text-embedding-3-small",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        assert embedder.max_tokens == 8192


def test_that_embedder_max_tokens_reads_from_env(container: Container) -> None:
    with patch.dict(
        os.environ,
        {"LITELLM_EMBEDDING_MAX_TOKENS": "4096"},
        clear=False,
    ):
        embedder = LiteLLMEmbedder(
            model_name="text-embedding-3-small",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        assert embedder.max_tokens == 4096


def test_that_embedder_dimensions_defaults_to_1536(container: Container) -> None:
    env = {k: v for k, v in os.environ.items() if k != "LITELLM_EMBEDDING_DIMENSIONS"}
    with patch.dict(os.environ, env, clear=True):
        embedder = LiteLLMEmbedder(
            model_name="text-embedding-3-small",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        assert embedder.dimensions == 1536


def test_that_embedder_dimensions_reads_from_env(container: Container) -> None:
    with patch.dict(
        os.environ,
        {"LITELLM_EMBEDDING_DIMENSIONS": "768"},
        clear=False,
    ):
        embedder = LiteLLMEmbedder(
            model_name="text-embedding-3-small",
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        assert embedder.dimensions == 768


def test_that_api_key_is_optional_for_verify_environment() -> None:
    with patch.dict(
        os.environ,
        {"LITELLM_PROVIDER_MODEL_NAME": "gpt-4"},
        clear=True,
    ):
        error = LiteLLMService.verify_environment()
        assert error is None


def test_that_litellm_cached_tokens_are_read_from_usage_sub_attribute() -> None:
    """getattr(response, 'usage.prompt_cache_hit_tokens', 0) always returns 0.
    The correct form reads response.usage.prompt_tokens_details.cached_tokens."""
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    prompt_tokens_details = MagicMock()
    prompt_tokens_details.cached_tokens = 11

    usage = MagicMock()
    usage.prompt_tokens = 20
    usage.completion_tokens = 5
    usage.prompt_tokens_details = prompt_tokens_details
    usage.model_dump_json = MagicMock(return_value="{}")

    message = MagicMock()
    message.content = '{"value": "hello"}'
    choice = MagicMock()
    choice.message = message

    fake_response = MagicMock()
    fake_response.usage = usage
    fake_response.choices = [choice]

    gen: Any = LiteLLM_Default.__new__(LiteLLM_Default)
    gen.logger = logger
    gen.tracer = tracer
    gen.meter = meter
    gen.model_name = "gpt-4"
    gen.base_url = None
    gen._tokenizer = LiteLLMEstimatingTokenizer("gpt-4")
    gen.schema = SampleSchema

    async_acompletion = AsyncMock(return_value=fake_response)
    mock_litellm = MagicMock()
    mock_litellm.acompletion = async_acompletion
    gen._client = mock_litellm

    recorded_calls: list[tuple[Any, ...]] = []

    async def fake_record_llm_metrics(*args: Any, **kwargs: Any) -> None:
        recorded_calls.append((args, kwargs))

    with patch(
        "parlant.adapters.nlp.litellm_service.record_llm_metrics",
        side_effect=fake_record_llm_metrics,
    ):
        result = asyncio.run(gen.do_generate("test prompt"))

    assert recorded_calls, "record_llm_metrics must be called"
    _, kwargs = recorded_calls[0]
    assert kwargs.get("cached_input_tokens") == 11, (
        f"cached_input_tokens should be 11, got {kwargs.get('cached_input_tokens')!r}"
    )
    assert result.info.usage.extra.get("cached_input_tokens") == 11
