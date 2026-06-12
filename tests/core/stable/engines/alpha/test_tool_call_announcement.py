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
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from parlant.core.agents import CompositionMode
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.canned_response_generator import CannedResponseGenerator
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.engines.alpha.message_event_composer import MessageEventComposition
from parlant.core.engines.alpha.perceived_performance_policy import (
    AnnouncingPerceivedPerformancePolicy,
    BasicPerceivedPerformancePolicy,
    PerceivedPerformancePolicyProvider,
)
from parlant.core.sessions import EventKind, EventSource
from parlant.core.tags import Tag


def _tool_event() -> EmittedEvent:
    return EmittedEvent(
        source=EventSource.AI_AGENT,
        kind=EventKind.TOOL,
        trace_id="test-trace",
        data={
            "tool_calls": [
                {
                    "tool_id": "mcp:check_balance",
                    "arguments": {"account": "123"},
                    "result": {"data": {"balance": 42}, "metadata": {}, "control": {}},
                }
            ]
        },
        metadata=None,
    )


def _make_generator() -> Any:
    generator: Any = object.__new__(CannedResponseGenerator)
    generator._logger = Mock(trace=Mock())
    generator._tracer = Mock(trace_id="test-trace", add_event=Mock())
    generator._canrep_preamble_generator = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                content=SimpleNamespace(
                    preamble="I've checked your account",
                    model_dump_json=Mock(return_value="{}"),
                ),
                info={},
            )
        )
    )
    return generator


def _make_context(composition_mode: CompositionMode) -> Any:
    emit_result = SimpleNamespace(event="EMITTED_EVENT")
    return SimpleNamespace(
        agent=SimpleNamespace(
            id="agent-1",
            name="Test Agent",
            description="A helpful test agent",
            composition_mode=composition_mode,
        ),
        state=SimpleNamespace(
            ordinary_guideline_matches=[],
            tool_enabled_guideline_matches={},
            message_events=[],
            guidelines=[],
        ),
        interaction=SimpleNamespace(events=[]),
        session_event_emitter=SimpleNamespace(
            emit_message_event=AsyncMock(return_value=emit_result)
        ),
    )


async def test_that_an_announcement_is_generated_and_emitted_with_preamble_tag() -> None:
    generator = _make_generator()
    context = _make_context(CompositionMode.CANNED_FLUID)

    compositions = await CannedResponseGenerator._do_generate_tool_call_announcement(
        generator, context, [_tool_event()]
    )

    assert len(compositions) == 1
    assert list(compositions[0].events) == [cast(EmittedEvent, "EMITTED_EVENT")]

    emit_call = context.session_event_emitter.emit_message_event.await_args
    assert emit_call.kwargs["data"]["message"] == "I've checked your account"
    assert emit_call.kwargs["data"]["tags"] == [Tag.preamble().id]


async def test_that_strict_composition_mode_skips_announcements() -> None:
    generator = _make_generator()
    context = _make_context(CompositionMode.CANNED_STRICT)

    compositions = await CannedResponseGenerator._do_generate_tool_call_announcement(
        generator, context, [_tool_event()]
    )

    assert compositions == []
    generator._canrep_preamble_generator.generate.assert_not_awaited()


async def test_that_an_empty_completion_emits_nothing() -> None:
    generator = _make_generator()
    generator._canrep_preamble_generator.generate.return_value = SimpleNamespace(
        content=SimpleNamespace(preamble="", model_dump_json=Mock(return_value="{}")),
        info={},
    )
    context = _make_context(CompositionMode.CANNED_FLUID)

    compositions = await CannedResponseGenerator._do_generate_tool_call_announcement(
        generator, context, [_tool_event()]
    )

    assert compositions == []
    context.session_event_emitter.emit_message_event.assert_not_awaited()


def _make_engine(composer: Any, *, announcing: bool = True) -> Any:
    engine: Any = object.__new__(AlphaEngine)
    default_policy = (
        AnnouncingPerceivedPerformancePolicy() if announcing else BasicPerceivedPerformancePolicy()
    )
    engine._perceived_performance_policy_provider = PerceivedPerformancePolicyProvider(
        default_policy=default_policy
    )
    engine._fluid_message_generator = composer
    engine._canned_response_generator = composer
    engine._logger = Mock(warning=Mock())
    return engine


def _make_engine_context() -> Any:
    return SimpleNamespace(
        agent=SimpleNamespace(id="agent-1", composition_mode=CompositionMode.CANNED_FLUID),
        state=SimpleNamespace(tool_announcement_tasks=[], message_events=[]),
    )


async def test_that_announcement_events_are_collected_into_message_events() -> None:
    composer = SimpleNamespace(
        generate_tool_call_announcement=AsyncMock(
            return_value=[
                MessageEventComposition(
                    generation_info={}, events=[cast(EmittedEvent, "ANNOUNCEMENT_EVENT")]
                )
            ]
        )
    )
    engine = _make_engine(composer)
    context = _make_engine_context()

    await AlphaEngine._start_tool_announcement_task(engine, context, [_tool_event()])
    assert len(context.state.tool_announcement_tasks) == 1

    await AlphaEngine._collect_tool_announcements(engine, context)

    assert context.state.message_events == ["ANNOUNCEMENT_EVENT"]
    assert context.state.tool_announcement_tasks == []


async def test_that_basic_policy_spawns_no_announcement_task() -> None:
    composer = SimpleNamespace(generate_tool_call_announcement=AsyncMock())
    engine = _make_engine(composer, announcing=False)
    context = _make_engine_context()

    await AlphaEngine._start_tool_announcement_task(engine, context, [_tool_event()])

    assert context.state.tool_announcement_tasks == []
    composer.generate_tool_call_announcement.assert_not_awaited()


async def test_that_a_failing_announcement_does_not_raise_out_of_collection() -> None:
    composer = SimpleNamespace(
        generate_tool_call_announcement=AsyncMock(side_effect=RuntimeError("boom"))
    )
    engine = _make_engine(composer)
    context = _make_engine_context()

    await AlphaEngine._start_tool_announcement_task(engine, context, [_tool_event()])
    await AlphaEngine._collect_tool_announcements(engine, context)

    assert context.state.message_events == []
    assert context.state.tool_announcement_tasks == []


async def test_that_discard_cancels_pending_announcement_tasks() -> None:
    generation_started = asyncio.Event()

    async def slow_generation(**kwargs: Any) -> Any:
        generation_started.set()
        await asyncio.sleep(60)

    composer = SimpleNamespace(generate_tool_call_announcement=slow_generation)
    engine = _make_engine(composer)
    context = _make_engine_context()

    await AlphaEngine._start_tool_announcement_task(engine, context, [_tool_event()])
    await generation_started.wait()

    await AlphaEngine._discard_tool_announcements(engine, context)

    assert context.state.tool_announcement_tasks == []


async def test_that_multiple_announcement_tasks_are_all_collected() -> None:
    composer = SimpleNamespace(
        generate_tool_call_announcement=AsyncMock(
            side_effect=[
                [MessageEventComposition(generation_info={}, events=[cast(EmittedEvent, "A1")])],
                [MessageEventComposition(generation_info={}, events=[cast(EmittedEvent, "A2")])],
            ]
        )
    )
    engine = _make_engine(composer)
    context = _make_engine_context()

    await AlphaEngine._start_tool_announcement_task(engine, context, [_tool_event()])
    await AlphaEngine._start_tool_announcement_task(engine, context, [_tool_event()])
    assert len(context.state.tool_announcement_tasks) == 2

    await AlphaEngine._collect_tool_announcements(engine, context)

    assert set(context.state.message_events) == {"A1", "A2"}
    assert context.state.tool_announcement_tasks == []


async def test_that_cancellation_during_collection_finalizes_tasks() -> None:
    generation_started = asyncio.Event()

    async def slow_generation(**kwargs: Any) -> Any:
        generation_started.set()
        await asyncio.sleep(60)

    composer = SimpleNamespace(generate_tool_call_announcement=slow_generation)
    engine = _make_engine(composer)
    context = _make_engine_context()

    await AlphaEngine._start_tool_announcement_task(engine, context, [_tool_event()])
    announcement_task = context.state.tool_announcement_tasks[0]

    collection = asyncio.ensure_future(AlphaEngine._collect_tool_announcements(engine, context))
    await generation_started.wait()
    collection.cancel()

    with pytest.raises(asyncio.CancelledError):
        await collection

    assert announcement_task.cancelled()
    assert context.state.tool_announcement_tasks == []
