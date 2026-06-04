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
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from lagom import Container

from parlant.adapters.nlp.glm_service import (
    GLMService,
    GLM_4_5,
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


class GLMSimpleSchema(DefaultBaseModel):
    result: str


def _make_openai_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    response.usage = usage
    return response


def test_that_glm_verify_environment_returns_error_when_api_key_missing() -> None:
    with patch.dict(os.environ, {}, clear=True):
        error = GLMService.verify_environment()
        assert error is not None
        assert "GLM_API_KEY is not set" in error


def test_that_glm_verify_environment_returns_none_when_api_key_present() -> None:
    with patch.dict(os.environ, {"GLM_API_KEY": "test-key"}, clear=True):
        error = GLMService.verify_environment()
        assert error is None


def test_that_glm_generator_does_not_raise_with_max_tokens_hint(
    container: Container,
) -> None:
    """Passing hints={'max_tokens': 1000} must not raise TypeError (duplicate kwarg)."""
    with patch.dict(
        os.environ,
        {"GLM_API_KEY": "test-key"},
        clear=False,
    ):
        # Use bracket syntax so __orig_class__ is set (required for .schema property)
        gen = GLM_4_5[GLMSimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        mock_response = _make_openai_response('{"result": "ok"}')
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(gen._client.chat.completions, "create", mock_create):
            # Must not raise TypeError: got multiple values for keyword argument 'max_tokens'
            asyncio.run(gen._do_generate("test prompt", hints={"max_tokens": 1000}))

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs.get("max_tokens") == 1000


def test_that_glm_generator_uses_default_max_tokens_when_not_in_hints(
    container: Container,
) -> None:
    """When hints do not include max_tokens, the call should still include a max_tokens value."""
    with patch.dict(
        os.environ,
        {"GLM_API_KEY": "test-key"},
        clear=False,
    ):
        # Use bracket syntax so __orig_class__ is set (required for .schema property)
        gen = GLM_4_5[GLMSimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )

        mock_response = _make_openai_response('{"result": "ok"}')
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(gen._client.chat.completions, "create", mock_create):
            asyncio.run(gen._do_generate("test prompt", hints={}))

        call_kwargs = mock_create.call_args[1]
        # A default must be present
        assert "max_tokens" in call_kwargs


def test_that_glm_supported_hints_includes_max_tokens(container: Container) -> None:
    with patch.dict(os.environ, {"GLM_API_KEY": "test-key"}, clear=False):
        # Use bracket syntax so __orig_class__ is set (required for .schema property)
        gen = GLM_4_5[GLMSimpleSchema](
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
        assert "max_tokens" in gen.supported_hints
