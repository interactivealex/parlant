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
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from lagom import Container

from parlant.adapters.nlp.qwen_service import (
    QwenService,
    get_qwen_base_url,
    QWEN_REGION_BASE_URLS,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import Meter, LocalMeter
from parlant.core.tracer import Tracer, LocalTracer


@pytest.fixture
def container() -> Container:
    c = Container()
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)
    c[Logger] = logger
    c[Tracer] = tracer
    c[Meter] = meter
    return c


class QwenSimpleSchema(DefaultBaseModel):
    answer: str


def _make_openai_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    response.usage = usage
    return response


def test_that_missing_api_key_returns_error_message() -> None:
    """Test that missing DASHSCOPE_API_KEY returns error message."""
    with patch.dict(os.environ, {}, clear=True):
        error = QwenService.verify_environment()
        assert error is not None
        assert "DASHSCOPE_API_KEY is not set" in error


def test_that_verify_environment_returns_error_for_invalid_region() -> None:
    """Test that verify_environment returns error for invalid QWEN_REGION."""
    with patch.dict(
        os.environ,
        {"DASHSCOPE_API_KEY": "test-key", "QWEN_REGION": "invalid-region"},
        clear=True,
    ):
        error = QwenService.verify_environment()
        assert error is not None
        assert "Invalid QWEN_REGION 'invalid-region'" in error
        assert "Must be one of: international, domestic" in error


def test_that_get_qwen_base_url_returns_international_by_default() -> None:
    """Test that get_qwen_base_url returns international URL by default."""
    with patch.dict(os.environ, {}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]
        assert url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_that_get_qwen_base_url_returns_domestic_url_when_region_is_domestic() -> None:
    """Test that get_qwen_base_url returns domestic URL when QWEN_REGION is domestic."""
    with patch.dict(os.environ, {"QWEN_REGION": "domestic"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]
        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_that_get_qwen_base_url_returns_international_url_when_region_is_international() -> None:
    """Test that get_qwen_base_url returns international URL when QWEN_REGION is international."""
    with patch.dict(os.environ, {"QWEN_REGION": "international"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]


def test_that_get_qwen_base_url_is_case_insensitive() -> None:
    """Test that QWEN_REGION is case insensitive."""
    with patch.dict(os.environ, {"QWEN_REGION": "DOMESTIC"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]

    with patch.dict(os.environ, {"QWEN_REGION": "Domestic"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["domestic"]

    with patch.dict(os.environ, {"QWEN_REGION": "INTERNATIONAL"}, clear=True):
        url = get_qwen_base_url()
        assert url == QWEN_REGION_BASE_URLS["international"]


def test_that_get_qwen_base_url_raises_error_for_invalid_region() -> None:
    """Test that get_qwen_base_url raises ValueError for invalid region."""
    with patch.dict(os.environ, {"QWEN_REGION": "invalid_region"}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            get_qwen_base_url()
        assert "Invalid QWEN_REGION" in str(exc_info.value)
        assert "international" in str(exc_info.value)
        assert "domestic" in str(exc_info.value)


def test_that_qwen_base_url_env_var_takes_priority() -> None:
    """Test that QWEN_BASE_URL environment variable takes priority over QWEN_REGION."""
    custom_url = "https://custom.api.url/v1"
    with patch.dict(
        os.environ,
        {"QWEN_BASE_URL": custom_url, "QWEN_REGION": "domestic"},
        clear=True,
    ):
        url = get_qwen_base_url()
        assert url == custom_url


def test_that_qwen_base_url_env_var_works_alone() -> None:
    """Test that QWEN_BASE_URL works without QWEN_REGION set."""
    custom_url = "https://custom.api.url/v1"
    with patch.dict(os.environ, {"QWEN_BASE_URL": custom_url}, clear=True):
        url = get_qwen_base_url()
        assert url == custom_url


def test_that_qwen_generator_does_not_raise_with_max_tokens_hint(
    container: Container,
) -> None:
    """Passing hints={'max_tokens': 1000} must not raise TypeError (duplicate kwarg)."""
    from parlant.adapters.nlp.qwen_service import Qwen_Plus

    with patch.dict(
        os.environ,
        {"DASHSCOPE_API_KEY": "test-key"},
        clear=False,
    ):
        # Use bracket syntax so __orig_class__ is set (required for .schema property)
        gen = Qwen_Plus[QwenSimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        mock_response = _make_openai_response('{"answer": "42"}')
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(gen._client.chat.completions, "create", mock_create):
            # Must not raise TypeError: got multiple values for keyword argument 'max_tokens'
            asyncio.run(gen._do_generate("test prompt", hints={"max_tokens": 1000}))

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs.get("max_tokens") == 1000


def test_that_qwen_generator_uses_default_max_tokens_when_not_in_hints(
    container: Container,
) -> None:
    """When hints do not include max_tokens, the call should still include a max_tokens value."""
    from parlant.adapters.nlp.qwen_service import Qwen_Plus

    with patch.dict(
        os.environ,
        {"DASHSCOPE_API_KEY": "test-key"},
        clear=False,
    ):
        # Use bracket syntax so __orig_class__ is set (required for .schema property)
        gen = Qwen_Plus[QwenSimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        mock_response = _make_openai_response('{"answer": "42"}')
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(gen._client.chat.completions, "create", mock_create):
            asyncio.run(gen._do_generate("test prompt", hints={}))

        call_kwargs = mock_create.call_args[1]
        # A default must be present — Qwen/DashScope requires max_tokens
        assert "max_tokens" in call_kwargs
