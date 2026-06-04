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
from unittest.mock import AsyncMock, MagicMock, patch


from parlant.adapters.nlp.cerebras_service import (
    CerebrasService,
    Llama3_3_70B,
    LlamaEstimatingTokenizer,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import StdoutLogger
from parlant.core.meter import LocalMeter
from parlant.core.tracer import LocalTracer


class SampleSchema(DefaultBaseModel):
    value: str = "ok"


def _make_fake_response(content: str) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.model_dump_json = MagicMock(return_value="{}")

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.usage = usage
    response.choices = [choice]
    return response


def test_that_cerebras_verify_environment_message_names_cerebras() -> None:
    """verify_environment error message must say 'Cerebras', not 'OpenAI NLP service'."""
    with patch.dict(os.environ, {}, clear=True):
        error = CerebrasService.verify_environment()
    assert error is not None
    assert "Cerebras" in error, f"Expected 'Cerebras' in error message, got: {error!r}"
    assert "OpenAI NLP service" not in error, "Error message must not mention 'OpenAI NLP service'"


def test_that_cerebras_record_llm_metrics_passes_schema_name() -> None:
    """record_llm_metrics must receive schema_name; missing it causes TypeError."""
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    fake_response = _make_fake_response('{"value": "hello"}')

    gen: Any = Llama3_3_70B.__new__(Llama3_3_70B)
    gen.logger = logger
    gen.tracer = tracer
    gen.meter = meter
    gen.model_name = "llama3.3-70b"
    gen._estimating_tokenizer = LlamaEstimatingTokenizer()
    gen.schema = SampleSchema

    async_create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = async_create
    gen._client = mock_client

    recorded_calls: list[tuple[Any, ...]] = []

    async def fake_record_llm_metrics(*args: Any, **kwargs: Any) -> None:
        recorded_calls.append((args, kwargs))

    with patch(
        "parlant.adapters.nlp.cerebras_service.record_llm_metrics",
        side_effect=fake_record_llm_metrics,
    ):
        asyncio.run(gen._do_generate("test prompt"))

    assert recorded_calls, "record_llm_metrics must be called"
    _, kwargs = recorded_calls[0]
    assert "schema_name" in kwargs, "schema_name must be passed as kwarg to record_llm_metrics"
    assert kwargs["schema_name"] == "SampleSchema"
