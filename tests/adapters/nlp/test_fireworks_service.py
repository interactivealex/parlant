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
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import StdoutLogger
from parlant.core.meter import LocalMeter
from parlant.core.tracer import LocalTracer


class SampleSchema(DefaultBaseModel):
    value: str = "ok"


@pytest.fixture()
def fake_fireworks_module() -> types.ModuleType:
    """Inject a fake fireworks.client module so the adapter can be imported."""
    fake_client_mod = types.ModuleType("fireworks.client")
    fake_error_mod = types.ModuleType("fireworks.client.error")

    class FakeRateLimitError(Exception):
        pass

    fake_client_mod.AsyncFireworks = MagicMock  # type: ignore[attr-defined]  # placeholder; overridden per-test
    fake_error_mod.RateLimitError = FakeRateLimitError  # type: ignore[attr-defined]

    # Insert into sys.modules BEFORE importing the adapter
    sys.modules.setdefault("fireworks", types.ModuleType("fireworks"))
    sys.modules["fireworks.client"] = fake_client_mod
    sys.modules["fireworks.client.error"] = fake_error_mod
    return fake_client_mod


def _make_fake_response(content: str) -> MagicMock:
    """Build a minimal fake fireworks chat response."""
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


def test_that_fireworks_do_generate_awaits_create(
    fake_fireworks_module: types.ModuleType,
) -> None:
    """create() on AsyncFireworks must be awaited; without await the coroutine is
    never executed and response is a coroutine object, not a response object.
    This test verifies the call is properly awaited by checking that the returned
    content matches the mock's return value (which would fail if response were
    still a coroutine)."""
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    fake_response = _make_fake_response('{"value": "hello"}')

    # Use a fresh import so the patched sys.modules are in effect
    from parlant.adapters.nlp import fireworks_service  # noqa: PLC0415

    # Use a concrete subclass; patch __init__ to avoid real API key
    gen: Any = fireworks_service.FireworksLlama3_1_8B.__new__(
        fireworks_service.FireworksLlama3_1_8B
    )
    # Manually set up the attributes __init__ would set
    gen.logger = logger
    gen.tracer = tracer
    gen.meter = meter
    gen.model_name = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    gen._tokenizer = fireworks_service.FireworksEstimatingTokenizer(gen.model_name)
    gen.schema = SampleSchema

    async_create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = async_create
    gen._client = mock_client

    result = asyncio.run(gen._do_generate("test prompt"))

    # The create coroutine must have been awaited exactly once
    async_create.assert_awaited_once()
    assert result.content.value == "hello"


def test_that_fireworks_record_llm_metrics_passes_schema_name(
    fake_fireworks_module: types.ModuleType,
) -> None:
    """record_llm_metrics must receive schema_name; missing it causes TypeError."""
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)

    fake_response = _make_fake_response('{"value": "hello"}')

    from parlant.adapters.nlp import fireworks_service  # noqa: PLC0415

    gen: Any = fireworks_service.FireworksLlama3_1_8B.__new__(
        fireworks_service.FireworksLlama3_1_8B
    )
    gen.logger = logger
    gen.tracer = tracer
    gen.meter = meter
    gen.model_name = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    gen._tokenizer = fireworks_service.FireworksEstimatingTokenizer(gen.model_name)
    gen.schema = SampleSchema

    async_create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = async_create
    gen._client = mock_client

    recorded_calls: list[tuple[Any, ...]] = []

    async def fake_record_llm_metrics(*args: Any, **kwargs: Any) -> None:
        recorded_calls.append((args, kwargs))

    with patch(
        "parlant.adapters.nlp.fireworks_service.record_llm_metrics",
        side_effect=fake_record_llm_metrics,
    ):
        asyncio.run(gen._do_generate("test prompt"))

    assert recorded_calls, "record_llm_metrics must be called"
    _, kwargs = recorded_calls[0]
    assert "schema_name" in kwargs, "schema_name must be passed as kwarg to record_llm_metrics"
    assert kwargs["schema_name"] == "SampleSchema"
