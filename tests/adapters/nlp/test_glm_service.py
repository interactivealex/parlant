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

from parlant.adapters.nlp.glm_service import GLM_4_5, GLMEstimatingTokenizer
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import StdoutLogger
from parlant.core.meter import LocalMeter
from parlant.core.tracer import LocalTracer


class SampleSchema(DefaultBaseModel):
    value: str = "ok"


def test_that_glm_cached_tokens_are_read_from_usage_sub_attribute() -> None:
    """getattr(response, 'usage.prompt_cache_hit_tokens', 0) always returns 0.
    The correct form reads response.usage.prompt_tokens_details.cached_tokens."""
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    prompt_tokens_details = MagicMock()
    prompt_tokens_details.cached_tokens = 13

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

    with patch.dict(os.environ, {"GLM_API_KEY": "fake-key"}):
        gen: Any = GLM_4_5.__new__(GLM_4_5)
        gen.logger = logger
        gen.tracer = tracer
        gen.meter = meter
        gen.model_name = "glm-4.5"
        gen._tokenizer = GLMEstimatingTokenizer("glm-4.5")
        gen.schema = SampleSchema

    async_create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = async_create
    gen._client = mock_client

    recorded_calls: list[tuple[Any, ...]] = []

    async def fake_record_llm_metrics(*args: Any, **kwargs: Any) -> None:
        recorded_calls.append((args, kwargs))

    with patch(
        "parlant.adapters.nlp.glm_service.record_llm_metrics",
        side_effect=fake_record_llm_metrics,
    ):
        result = asyncio.run(gen._do_generate("test prompt"))

    assert recorded_calls, "record_llm_metrics must be called"
    _, kwargs = recorded_calls[0]
    assert kwargs.get("cached_input_tokens") == 13, (
        f"cached_input_tokens should be 13, got {kwargs.get('cached_input_tokens')!r}"
    )
    assert result.info.usage.extra.get("cached_input_tokens") == 13
