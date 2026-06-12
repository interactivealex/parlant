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

"""Regression tests for the matcher batches' generation retry loops.

These four batches used to wrap their 3-attempt loop in a single ``try``
whose ``except`` sat AFTER the ``for`` — so the first generation exception
aborted the whole loop and the remaining attempts never ran. The tests pin
the repaired shape behaviorally: a transiently failing generator must not
fail the batch, and an always-failing one must consume all three attempts.

The batches are constructed via ``object.__new__`` with stubbed attributes
so the REAL ``process()`` runs without the full DI container or a live
generator.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from pytest import fixture, mark, raises

from parlant.core.engines.alpha.guideline_matching import common as gm_common
from parlant.core.engines.alpha.guideline_matching.generic.guideline_actionable_batch import (
    GenericActionableGuidelineMatchesSchema,
    GenericActionableGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_low_criticality_batch import (
    GenericLowCriticalityGuidelineMatchesSchema,
    GenericLowCriticalityGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_customer_dependent_batch import (
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatchingBatchError,
)

# (batch class, empty-result schema instance factory)
BATCH_CLASSES = [
    (
        GenericActionableGuidelineMatchingBatch,
        lambda: GenericActionableGuidelineMatchesSchema(checks=[]),
    ),
    (
        GenericPreviouslyAppliedActionableGuidelineMatchingBatch,
        lambda: GenericPreviouslyAppliedActionableGuidelineMatchesSchema(checks=[]),
    ),
    (
        GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingBatch,
        lambda: GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema(
            checks=[]
        ),
    ),
    (
        GenericLowCriticalityGuidelineMatchingBatch,
        lambda: GenericLowCriticalityGuidelineMatchesSchema(applies={}),
    ),
]


class _NullHistogram:
    def measure(self, attributes: dict[str, str]) -> Any:
        @asynccontextmanager
        async def cm() -> Any:
            yield

        return cm()


@fixture(autouse=True)
def _null_batch_histogram() -> Any:
    prior = gm_common._MATCHING_BATCH_DURATION_HISTOGRAM
    gm_common._MATCHING_BATCH_DURATION_HISTOGRAM = cast(Any, _NullHistogram())
    yield
    gm_common._MATCHING_BATCH_DURATION_HISTOGRAM = prior


class _FlakyGenerator:
    """Fails the first ``fail_n_times`` generate() calls, then succeeds
    with an empty result."""

    def __init__(self, empty_result_factory: Any, fail_n_times: int) -> None:
        self._empty_result_factory = empty_result_factory
        self._fail_n_times = fail_n_times
        self.calls = 0

    async def generate(self, prompt: Any, hints: dict[str, Any]) -> Any:
        self.calls += 1
        if self.calls <= self._fail_n_times:
            raise RuntimeError(f"synthetic transient failure #{self.calls}")
        return SimpleNamespace(
            content=self._empty_result_factory(),
            info=MagicMock(),
        )


def _make_batch(batch_class: type, generator: _FlakyGenerator) -> Any:
    batch: Any = object.__new__(batch_class)
    batch._logger = MagicMock()
    batch._meter = MagicMock()
    batch._optimization_policy = MagicMock(
        get_guideline_matching_batch_retry_temperatures=MagicMock(return_value=(0.3, 0.5, 0.9))
    )
    batch._schematic_generator = generator
    batch._guidelines = {}
    batch._journeys = []
    batch._context = MagicMock()
    batch._build_prompt = lambda shots: "stub prompt"

    async def _no_shots() -> list[Any]:
        return []

    batch.shots = _no_shots
    return batch


@mark.parametrize("batch_class,empty_result_factory", BATCH_CLASSES)
async def test_that_a_transient_generation_failure_is_retried(
    batch_class: type,
    empty_result_factory: Any,
) -> None:
    generator = _FlakyGenerator(empty_result_factory, fail_n_times=1)
    batch = _make_batch(batch_class, generator)

    result = await batch.process()

    assert result.matches == []
    assert generator.calls == 2


@mark.parametrize("batch_class,empty_result_factory", BATCH_CLASSES)
async def test_that_all_three_attempts_run_before_the_batch_fails(
    batch_class: type,
    empty_result_factory: Any,
) -> None:
    generator = _FlakyGenerator(empty_result_factory, fail_n_times=10)
    batch = _make_batch(batch_class, generator)

    with raises(GuidelineMatchingBatchError):
        await batch.process()

    assert generator.calls == 3
