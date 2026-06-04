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


from typing import TypedDict
import weakref

from parlant.core.meter import Counter, Meter


def normalize_json_output(raw_output: str) -> str:
    json_start = raw_output.find("```json")

    if json_start != -1:
        json_start = json_start + 7
    else:
        json_start = 0

    json_end = raw_output[json_start:].rfind("```")

    if json_end == -1:
        json_end = len(raw_output[json_start:])

    return raw_output[json_start : json_start + json_end].strip()


class _MeterCounters(TypedDict):
    input_tokens: Counter
    output_tokens: Counter
    cached_input_tokens: Counter


# Per-meter counter cache. Weak keys let short-lived meters (e.g. in tests) be
# garbage-collected together with their counters.
_METER_COUNTERS: weakref.WeakKeyDictionary[Meter, _MeterCounters] = weakref.WeakKeyDictionary()


def _get_or_create_counters(meter: Meter) -> _MeterCounters:
    """Return the three LLM counters for *meter*, creating them on first call."""
    counters = _METER_COUNTERS.get(meter)
    if counters is None:
        counters = _MeterCounters(
            input_tokens=meter.create_counter(
                name="input_tokens",
                description="Number of input tokens sent to a LLM model",
            ),
            output_tokens=meter.create_counter(
                name="output_tokens",
                description="Number of output tokens received from a LLM model",
            ),
            cached_input_tokens=meter.create_counter(
                name="cached_input_tokens",
                description="Number of input tokens served from cache for a LLM model",
            ),
        )
        _METER_COUNTERS[meter] = counters
    return counters


async def record_llm_metrics(
    meter: Meter,
    model_name: str,
    schema_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> None:
    counters = _get_or_create_counters(meter)
    attributes = {"model_name": model_name, "schema_name": schema_name}

    await counters["input_tokens"].increment(input_tokens, attributes)
    await counters["output_tokens"].increment(output_tokens, attributes)
    await counters["cached_input_tokens"].increment(cached_input_tokens, attributes)
