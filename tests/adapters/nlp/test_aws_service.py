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

from parlant.adapters.nlp.aws_service import (
    AnthropicBedrockAISchematicGenerator,
    Claude_Sonnet_3_5,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import Meter, LocalMeter
from parlant.core.tracer import Tracer, LocalTracer

AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "test-key-id",
    "AWS_SECRET_ACCESS_KEY": "test-secret",
    "AWS_REGION": "us-east-1",
}


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


class AWSSimpleSchema(DefaultBaseModel):
    answer: str


def _make_anthropic_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    response.usage = usage
    return response


def test_that_aws_generator_uses_default_max_output_tokens_when_not_in_hints(
    container: Container,
) -> None:
    """When hints do not include max_tokens, messages.create must use DEFAULT_MAX_OUTPUT_TOKENS."""
    with patch.dict(os.environ, AWS_ENV, clear=False):
        with patch("parlant.adapters.nlp.aws_service.AsyncAnthropicBedrock"):
            # Use bracket syntax so __orig_class__ is set (required for .schema property)
            gen = Claude_Sonnet_3_5[AWSSimpleSchema](
                logger=container[Logger],
                tracer=container[Tracer],
                meter=container[Meter],
            )

        mock_response = _make_anthropic_response('{"answer": "hello"}')
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(gen._client.messages, "create", mock_create):
            asyncio.run(gen._do_generate("test prompt", hints={}))

        call_kwargs = mock_create.call_args[1]
        assert "max_tokens" in call_kwargs
        assert (
            call_kwargs["max_tokens"]
            == AnthropicBedrockAISchematicGenerator.DEFAULT_MAX_OUTPUT_TOKENS
        )


def test_that_aws_generator_uses_max_tokens_from_hints(
    container: Container,
) -> None:
    """When hints include max_tokens, messages.create uses that value."""
    with patch.dict(os.environ, AWS_ENV, clear=False):
        with patch("parlant.adapters.nlp.aws_service.AsyncAnthropicBedrock"):
            # Use bracket syntax so __orig_class__ is set (required for .schema property)
            gen = Claude_Sonnet_3_5[AWSSimpleSchema](
                logger=container[Logger],
                tracer=container[Tracer],
                meter=container[Meter],
            )

        mock_response = _make_anthropic_response('{"answer": "hello"}')
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(gen._client.messages, "create", mock_create):
            asyncio.run(gen._do_generate("test prompt", hints={"max_tokens": 2048}))

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs.get("max_tokens") == 2048


def test_that_aws_default_max_output_tokens_is_8192() -> None:
    """DEFAULT_MAX_OUTPUT_TOKENS must be 8192 (claude-3-5-sonnet Bedrock max output)."""
    assert AnthropicBedrockAISchematicGenerator.DEFAULT_MAX_OUTPUT_TOKENS == 8192


def test_that_aws_supported_hints_includes_max_tokens(container: Container) -> None:
    with patch.dict(os.environ, AWS_ENV, clear=False):
        with patch("parlant.adapters.nlp.aws_service.AsyncAnthropicBedrock"):
            # Use bracket syntax so __orig_class__ is set (required for .schema property)
            gen = Claude_Sonnet_3_5[AWSSimpleSchema](
                logger=container[Logger],
                tracer=container[Tracer],
                meter=container[Meter],
            )
        assert "max_tokens" in gen.supported_hints


def test_that_aws_max_tokens_property_is_context_window_not_output_limit(
    container: Container,
) -> None:
    """max_tokens property = context window (200k) — distinct from DEFAULT_MAX_OUTPUT_TOKENS."""
    with patch.dict(os.environ, AWS_ENV, clear=False):
        with patch("parlant.adapters.nlp.aws_service.AsyncAnthropicBedrock"):
            # Use bracket syntax so __orig_class__ is set (required for .schema property)
            gen = Claude_Sonnet_3_5[AWSSimpleSchema](
                logger=container[Logger],
                tracer=container[Tracer],
                meter=container[Meter],
            )
        assert gen.max_tokens == 200 * 1024
        assert gen.max_tokens != AnthropicBedrockAISchematicGenerator.DEFAULT_MAX_OUTPUT_TOKENS
