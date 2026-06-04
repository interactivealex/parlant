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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from parlant.adapters.nlp.modelscope_service import (
    ModelScopeSchematicGenerator,
    ModelScopeChat,
    ModelScopeEstimatingTokenizer,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import StdoutLogger
from parlant.core.meter import LocalMeter
from parlant.core.tracer import LocalTracer
from parlant.core.nlp.generation import BaseSchematicGenerator


class SampleSchema(DefaultBaseModel):
    value: str = "ok"


def test_that_modelscope_generator_exposes_do_generate_not_generate() -> None:
    """ModelScope must override do_generate (not generate) so that the
    BaseSchematicGenerator.generate wrapper with tracing/metrics is exercised."""
    # do_generate must be defined directly on the class (not inherited from base)
    # while generate must be the base-class implementation
    assert "do_generate" in ModelScopeSchematicGenerator.__dict__, (
        "ModelScopeSchematicGenerator must override do_generate, not generate"
    )
    assert "generate" not in ModelScopeSchematicGenerator.__dict__, (
        "ModelScopeSchematicGenerator must not override generate directly; "
        "that bypasses the base-class tracing pipeline"
    )


def test_that_modelscope_generate_calls_base_class_pipeline() -> None:
    """Calling generator.generate() must go through BaseSchematicGenerator.generate
    (which records duration histogram), not a direct shortcut."""
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    # Build a fake streaming response (ModelScope uses stream=True)
    async def async_iter() -> Any:  # type: ignore[return]
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = '{"value": "hi"}'
        yield chunk

    fake_stream = async_iter()

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    # litellm/openai streaming doesn't return usage on the stream object directly;
    # ModelScope adapter estimates tokens from the tokenizer so no usage needed

    async_create = AsyncMock(return_value=fake_stream)
    mock_client = MagicMock()
    mock_client.chat.completions.create = async_create

    # Call BaseSchematicGenerator.__init__ to initialize the histogram,
    # but skip ModelScopeSchematicGenerator.__init__ (which needs a real API key)
    gen: Any = ModelScopeChat.__new__(ModelScopeChat)

    BaseSchematicGenerator.__init__(
        gen,
        logger=logger,
        tracer=tracer,
        meter=meter,
        model_name="Qwen/Qwen3-8B",
    )
    gen._tokenizer = ModelScopeEstimatingTokenizer("Qwen/Qwen3-8B")
    gen.schema = SampleSchema
    gen._client = mock_client

    # generate() is the base-class method; if ModelScope overrides it directly
    # it would skip the histogram. We verify it's callable and returns correct type.
    result = asyncio.run(gen.generate("test prompt"))
    assert result.content.value == "hi"
