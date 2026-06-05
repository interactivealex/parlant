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

from parlant.adapters.nlp.common import record_llm_metrics

from tests.test_utilities import RecordingMeter


async def test_that_record_llm_metrics_creates_counters_on_first_meter() -> None:
    meter = RecordingMeter()

    await record_llm_metrics(
        meter,
        model_name="test-model",
        schema_name="TestSchema",
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=2,
    )

    assert "input_tokens" in meter.counters
    assert "output_tokens" in meter.counters
    assert "cached_input_tokens" in meter.counters

    assert meter.counters["input_tokens"].calls == [
        (10, {"model_name": "test-model", "schema_name": "TestSchema"})
    ]
    assert meter.counters["output_tokens"].calls == [
        (5, {"model_name": "test-model", "schema_name": "TestSchema"})
    ]
    assert meter.counters["cached_input_tokens"].calls == [
        (2, {"model_name": "test-model", "schema_name": "TestSchema"})
    ]


async def test_that_record_llm_metrics_uses_separate_counters_per_meter() -> None:
    """Each Meter instance must get its own set of counters.

    Previously, a module-level _COUNTERS_INITIALIZED flag meant the first Meter
    to call record_llm_metrics would 'win' and all subsequent Meters would silently
    increment counters on the wrong Meter.
    """
    meter_a = RecordingMeter()
    meter_b = RecordingMeter()

    await record_llm_metrics(
        meter_a,
        model_name="model-a",
        schema_name="SchemaA",
        input_tokens=100,
        output_tokens=50,
        cached_input_tokens=10,
    )

    await record_llm_metrics(
        meter_b,
        model_name="model-b",
        schema_name="SchemaB",
        input_tokens=200,
        output_tokens=80,
        cached_input_tokens=0,
    )

    # meter_a counters must only contain its own calls
    assert meter_a.counters["input_tokens"].calls == [
        (100, {"model_name": "model-a", "schema_name": "SchemaA"})
    ]
    assert meter_a.counters["output_tokens"].calls == [
        (50, {"model_name": "model-a", "schema_name": "SchemaA"})
    ]
    assert meter_a.counters["cached_input_tokens"].calls == [
        (10, {"model_name": "model-a", "schema_name": "SchemaA"})
    ]

    # meter_b counters must only contain its own calls
    assert meter_b.counters["input_tokens"].calls == [
        (200, {"model_name": "model-b", "schema_name": "SchemaB"})
    ]
    assert meter_b.counters["output_tokens"].calls == [
        (80, {"model_name": "model-b", "schema_name": "SchemaB"})
    ]
    assert meter_b.counters["cached_input_tokens"].calls == [
        (0, {"model_name": "model-b", "schema_name": "SchemaB"})
    ]


async def test_that_record_llm_metrics_reuses_counters_for_same_meter() -> None:
    """Counters are created once per Meter and reused on subsequent calls."""
    meter = RecordingMeter()

    await record_llm_metrics(
        meter,
        model_name="model",
        schema_name="Schema",
        input_tokens=1,
        output_tokens=1,
    )

    first_input_counter = meter.counters["input_tokens"]

    await record_llm_metrics(
        meter,
        model_name="model",
        schema_name="Schema",
        input_tokens=2,
        output_tokens=2,
    )

    # Same counter object — create_counter was called only once
    assert meter.counters["input_tokens"] is first_input_counter
    assert len(meter.counters["input_tokens"].calls) == 2


async def test_that_record_llm_metrics_increments_openrouter_counters_with_schema_name() -> None:
    """Verifies the schema_name attribute is propagated correctly.

    This mirrors the check that should hold after the openrouter_service.py
    metrics call is added.
    """
    meter = RecordingMeter()

    await record_llm_metrics(
        meter,
        model_name="openai/gpt-4o",
        schema_name="TestSchema",
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=0,
    )

    attrs = meter.counters["input_tokens"].calls[0][1]
    assert attrs is not None
    assert attrs["schema_name"] == "TestSchema"
    assert attrs["model_name"] == "openai/gpt-4o"
