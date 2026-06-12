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

from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.perceived_performance_policy import (
    AnnouncingPerceivedPerformancePolicy,
    BasicPerceivedPerformancePolicy,
    NullPerceivedPerformancePolicy,
)
from parlant.core.sessions import EventKind, EventSource


def _tool_event() -> EmittedEvent:
    return EmittedEvent(
        source=EventSource.AI_AGENT,
        kind=EventKind.TOOL,
        trace_id="test-trace",
        data={"tool_calls": []},
        metadata=None,
    )


async def test_that_basic_policy_does_not_require_tool_call_announcements() -> None:
    policy = BasicPerceivedPerformancePolicy()
    assert not await policy.is_tool_call_announcement_required(None, [_tool_event()])


async def test_that_null_policy_does_not_require_tool_call_announcements() -> None:
    policy = NullPerceivedPerformancePolicy()
    assert not await policy.is_tool_call_announcement_required(None, [_tool_event()])


async def test_that_announcing_policy_requires_announcements_when_tools_ran() -> None:
    policy = AnnouncingPerceivedPerformancePolicy()
    assert await policy.is_tool_call_announcement_required(None, [_tool_event()])


async def test_that_announcing_policy_skips_announcements_without_tool_events() -> None:
    policy = AnnouncingPerceivedPerformancePolicy()
    assert not await policy.is_tool_call_announcement_required(None, [])
    assert not await policy.is_tool_call_announcement_required(None, None)


async def test_that_announcing_policy_keeps_basic_preamble_behavior() -> None:
    policy = AnnouncingPerceivedPerformancePolicy()
    assert isinstance(policy, BasicPerceivedPerformancePolicy)
    assert not await policy.is_preamble_required(None)
