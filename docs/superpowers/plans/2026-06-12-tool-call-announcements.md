# Tool-Call Announcement Perceived-Performance Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third perceived-performance mode that, after each batch of tool calls completes, emits a short customer-visible status message ("I've checked your account") generated from the conversation history, active guidelines, and the executed tool calls + results — so perceived latency drops while the full response is still being composed.

**Architecture:** A new `AnnouncingPerceivedPerformancePolicy` (extends `Basic`) opts into announcements via a new policy method. The engine, right after `execute_tool_calls` returns in each preparation iteration, spawns a concurrent announcement-generation task (post-execution = result-aware, no false announcements). Pending tasks are gathered immediately before response generation, so announcements always precede the final message and land in `state.message_events` (the response prompt sees them — no repetition). Generation reuses the preamble machinery in `CannedResponseGenerator` (same `CannedResponsePreambleSchema` + `_canrep_preamble_generator` — zero DI churn) and emits the message tagged `Tag.preamble()` so downstream wire contracts are unchanged.

**Tech Stack:** Python 3.12, asyncio, parlant alpha engine, pytest (`asyncio_mode = auto`).

**Scope:** Parlant fork only (this worktree). The agent-server side (manifest knob, applicator policy selection, generation-budget patch, pin bump) is a separate follow-up plan in the interactive-agent repo.

**Worktree:** `/Users/ibanez/projects/parlant/parlant/.worktrees/feat-tool-call-announcements`, branch `feat/tool-call-announcements`.

---

## File Structure

| File | Change |
|---|---|
| `src/parlant/core/engines/alpha/perceived_performance_policy.py` | New abstract method + impls; new `AnnouncingPerceivedPerformancePolicy` |
| `src/parlant/core/engines/alpha/message_event_composer.py` | New abstract method `generate_tool_call_announcement` |
| `src/parlant/core/engines/alpha/message_generator.py` | Stub impl returning `[]` |
| `src/parlant/core/engines/alpha/engine_context.py` | `ResponseState.tool_announcement_tasks` field |
| `src/parlant/core/engines/alpha/canned_response_generator.py` | Real impl + default examples + histogram |
| `src/parlant/core/engines/alpha/engine.py` | Spawn task after tool execution (both iteration paths); collect before message generation; discard on unwind |
| `src/parlant/sdk.py` | Export `AnnouncingPerceivedPerformancePolicy` |
| `tests/core/stable/engines/alpha/test_perceived_performance_policy.py` | New — policy unit tests |
| `tests/core/stable/engines/alpha/test_tool_call_announcement.py` | New — engine helper + composer unit tests |

---

### Task 1: Policy method + AnnouncingPerceivedPerformancePolicy

**Files:**
- Modify: `src/parlant/core/engines/alpha/perceived_performance_policy.py`
- Test: `tests/core/stable/engines/alpha/test_perceived_performance_policy.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/stable/engines/alpha/test_perceived_performance_policy.py
# (Apache header — copy from a neighboring test file)
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
    # Basic returns False for a missing context
    assert not await policy.is_preamble_required(None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/core/stable/engines/alpha/test_perceived_performance_policy.py`
Expected: FAIL — `ImportError: cannot import name 'AnnouncingPerceivedPerformancePolicy'`

- [ ] **Step 3: Implement**

In `perceived_performance_policy.py`:

Imports — extend the existing block:
```python
from typing import Sequence, cast
from parlant.core.emissions import EmittedEvent
```

Add to the `PerceivedPerformancePolicy` ABC (after `is_preamble_required`):
```python
    @abstractmethod
    async def is_tool_call_announcement_required(
        self,
        context: EngineContext | None = None,
        tool_events: Sequence[EmittedEvent] | None = None,
    ) -> bool:
        """
        Determines if a status message should be generated after tool calls complete.

        :param context: The loaded context containing session and interaction details.
        :param tool_events: The tool events produced by the just-executed tool calls.
        :return: True if an announcement should be generated, False otherwise.
        """
        ...
```

Add to `BasicPerceivedPerformancePolicy` (after `is_preamble_required`) and the identical override to `NullPerceivedPerformancePolicy`:
```python
    @override
    async def is_tool_call_announcement_required(
        self,
        context: EngineContext | None = None,
        tool_events: Sequence[EmittedEvent] | None = None,
    ) -> bool:
        return False
```

Add after `VoiceOptimizedPerceivedPerformancePolicy`:
```python
class AnnouncingPerceivedPerformancePolicy(BasicPerceivedPerformancePolicy):
    """Extends the basic policy with post-tool-call status announcements.

    On top of the preamble behavior, after each batch of tool calls completes,
    the engine emits a short customer-visible message describing what was just
    done, so the customer perceives progress while the full response is still
    being composed.
    """

    @override
    async def is_tool_call_announcement_required(
        self,
        context: EngineContext | None = None,
        tool_events: Sequence[EmittedEvent] | None = None,
    ) -> bool:
        return bool(tool_events)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q tests/core/stable/engines/alpha/test_perceived_performance_policy.py`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/parlant/core/engines/alpha/perceived_performance_policy.py tests/core/stable/engines/alpha/test_perceived_performance_policy.py
git commit -m "feat(engine): announcing perceived-performance policy with tool-call announcement gate"
```

---

### Task 2: Composer interface + MessageGenerator stub + ResponseState field

**Files:**
- Modify: `src/parlant/core/engines/alpha/message_event_composer.py`
- Modify: `src/parlant/core/engines/alpha/message_generator.py:159-164`
- Modify: `src/parlant/core/engines/alpha/engine_context.py:145-161`

No standalone test (interface + trivial stub + dataclass field); covered by Task 3/4 tests.

- [ ] **Step 1: Add abstract method to `MessageEventComposer`** (after `generate_preamble`; `Sequence` and `EmittedEvent` are already imported):

```python
    @abstractmethod
    async def generate_tool_call_announcement(
        self,
        context: EngineContext,
        tool_events: Sequence[EmittedEvent],
    ) -> Sequence[MessageEventComposition]: ...
```

- [ ] **Step 2: Add stub to `MessageGenerator`** (after its `generate_preamble`):

```python
    @override
    async def generate_tool_call_announcement(
        self,
        context: EngineContext,
        tool_events: Sequence[EmittedEvent],
    ) -> Sequence[MessageEventComposition]:
        return []
```

- [ ] **Step 3: Add field to `ResponseState`** (in `engine_context.py`, after `additional_canned_response_fields`; add `import asyncio` to imports — `Any` and `field` are already imported). Loose `Any` typing avoids a circular import with `message_event_composer`:

```python
    tool_announcement_tasks: list[asyncio.Task[Any]] = field(default_factory=list)
```

- [ ] **Step 4: Sanity-run an existing test module to catch import breakage**

Run: `uv run pytest -q tests/core/stable/engines/alpha/test_render_section.py`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add src/parlant/core/engines/alpha/message_event_composer.py src/parlant/core/engines/alpha/message_generator.py src/parlant/core/engines/alpha/engine_context.py
git commit -m "feat(engine): tool-call announcement composer interface and response-state slot"
```

---

### Task 3: CannedResponseGenerator implementation

**Files:**
- Modify: `src/parlant/core/engines/alpha/canned_response_generator.py`
- Test: `tests/core/stable/engines/alpha/test_tool_call_announcement.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/stable/engines/alpha/test_tool_call_announcement.py
# (Apache header — copy from a neighboring test file)
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

from parlant.core.agents import CompositionMode
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.canned_response_generator import CannedResponseGenerator
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


def _make_generator() -> CannedResponseGenerator:
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


def _make_context(composition_mode: CompositionMode) -> SimpleNamespace:
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
    assert compositions[0].events == ["EMITTED_EVENT"]

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/core/stable/engines/alpha/test_tool_call_announcement.py`
Expected: FAIL — `AttributeError: ... has no attribute '_do_generate_tool_call_announcement'`

- [ ] **Step 3: Implement**

In `canned_response_generator.py`:

3a. Module-level defaults, next to `default_fluid_preamble_examples` (~line 2980):
```python
default_tool_call_announcement_examples: list[str] = [
    "I've pulled up your account details",
    "Just checked that in our system",
    "I've looked into that for you",
    "Got the latest information on that",
    "Alright, I've run a quick check on that",
]
```

3b. Histogram in `_define_histograms` (after `_hist_preamble_render_duration`):
```python
        self._hist_tool_announcement_duration = _create_histogram(
            name="tool_announcement",
            description="Duration of tool call announcement generation in milliseconds",
        )
```

3c. Methods after `_do_generate_preamble` (~line 923):
```python
    @override
    async def generate_tool_call_announcement(
        self,
        context: EngineContext,
        tool_events: Sequence[EmittedEvent],
    ) -> Sequence[MessageEventComposition]:
        with self._logger.scope("MessageEventComposer"):
            with self._logger.scope("CannedResponseGenerator"):
                async with self._hist_tool_announcement_duration.measure():
                    return await self._do_generate_tool_call_announcement(
                        context, tool_events
                    )

    async def _do_generate_tool_call_announcement(
        self,
        context: EngineContext,
        tool_events: Sequence[EmittedEvent],
    ) -> Sequence[MessageEventComposition]:
        agent = context.agent

        composition_mode = await self._resolve_composition_mode(context)

        if composition_mode == CompositionMode.CANNED_STRICT:
            # Strict agents may only speak in pre-approved canned responses,
            # which the announcement generator does not select from.
            return []

        prompt_builder = PromptBuilder(
            on_build=lambda prompt: self._logger.trace(
                f"Tool call announcement Prompt:\n{prompt}"
            )
        )

        prompt_builder.add_agent_identity(agent)

        guidelines_text = "\n".join(
            f"- When {m.guideline.content.condition}, then: {m.guideline.content.action}"
            for m in (
                *context.state.ordinary_guideline_matches,
                *context.state.tool_enabled_guideline_matches.keys(),
            )
            if m.guideline.content.action
        )

        announcement_choices_text = "".join(
            f"\n- {choice}" for choice in default_tool_call_announcement_examples
        )

        prompt_builder.add_section(
            name="tool-call-announcement-instructions",
            template="""\
You are an AI agent that just finished running one or more background operations
(tools) while preparing a full response for the customer. The full response will
be sent shortly by a smarter agent. Your only job is to generate a very short,
natural status message telling the customer what you just did, so they know
things are progressing.

The operations that just ran, including their results, are listed below under
STAGED EVENTS.

Behavioral guidelines currently active for this conversation (for tone and
context only — do not act on them): ###
{guidelines_text}
###

The announcement must:
- Be a single, short sentence (aim for under 12 words)
- Describe what was just done in plain, customer-friendly terms (e.g. "I've checked your account")
- NEVER mention internal tool names, ids, systems, or technical details
- NOT state the actual answer, results, figures, or conclusions — the full response comes next
- NOT ask questions, make commitments, or indicate next steps
- NOT repeat or paraphrase previous messages, preambles, or announcements — it must add something new

Here are some GOOD EXAMPLES of announcement messages: ###
{announcement_choices_text}
###

You must produce a JSON object with a single key, "preamble", holding the announcement message as a string.
""",
            props={
                "guidelines_text": guidelines_text or "- (none)",
                "announcement_choices_text": announcement_choices_text,
            },
        )

        prompt_builder.add_interaction_history_for_message_generation(
            context.interaction.events,
            context.state.message_events,
        )

        prompt_builder.add_staged_tool_events(tool_events)

        generation = await self._canrep_preamble_generator.generate(
            prompt=prompt_builder, hints={"temperature": 0.1}
        )

        self._logger.trace(
            f"Tool call announcement Completion:\n{generation.content.model_dump_json(indent=2)}"
        )

        if not generation.content.preamble:
            return []

        handle = await context.session_event_emitter.emit_message_event(
            trace_id=self._tracer.trace_id,
            data=MessageEventData(
                message=generation.content.preamble,
                participant=Participant(id=agent.id, display_name=agent.name),
                tags=[Tag.preamble().id],
            ),
        )

        self._tracer.add_event("canrep.tool_call_announcement_generated")

        return [
            MessageEventComposition(
                generation_info={"tool_call_announcement": generation.info},
                events=[handle.event],
            )
        ]
```

All names (`PromptBuilder`, `CompositionMode`, `MessageEventData`, `Participant`, `Tag`, `EmittedEvent`, `Sequence`) are already imported in this module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q tests/core/stable/engines/alpha/test_tool_call_announcement.py`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/parlant/core/engines/alpha/canned_response_generator.py tests/core/stable/engines/alpha/test_tool_call_announcement.py
git commit -m "feat(engine): result-aware tool-call announcement generation in canned-response composer"
```

---

### Task 4: Engine wiring

**Files:**
- Modify: `src/parlant/core/engines/alpha/engine.py` (both iteration paths, `uncancellable_section`, unwind cleanup, three new methods)
- Test: `tests/core/stable/engines/alpha/test_tool_call_announcement.py` (extend)

- [ ] **Step 1: Write the failing tests** (append to the Task 3 test file)

```python
import asyncio

from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.engines.alpha.message_event_composer import MessageEventComposition
from parlant.core.engines.alpha.perceived_performance_policy import (
    AnnouncingPerceivedPerformancePolicy,
    BasicPerceivedPerformancePolicy,
    PerceivedPerformancePolicyProvider,
)


def _make_engine(composer: Any, *, announcing: bool = True) -> AlphaEngine:
    engine: Any = object.__new__(AlphaEngine)
    default_policy = (
        AnnouncingPerceivedPerformancePolicy()
        if announcing
        else BasicPerceivedPerformancePolicy()
    )
    engine._perceived_performance_policy_provider = PerceivedPerformancePolicyProvider(
        default_policy=default_policy
    )
    engine._fluid_message_generator = composer
    engine._canned_response_generator = composer
    engine._logger = Mock(warning=Mock())
    return engine


def _make_engine_context() -> SimpleNamespace:
    return SimpleNamespace(
        agent=SimpleNamespace(id="agent-1", composition_mode=CompositionMode.CANNED_FLUID),
        state=SimpleNamespace(tool_announcement_tasks=[], message_events=[]),
    )


async def test_that_announcement_events_are_collected_into_message_events() -> None:
    composer = SimpleNamespace(
        generate_tool_call_announcement=AsyncMock(
            return_value=[
                MessageEventComposition(generation_info={}, events=["ANNOUNCEMENT_EVENT"])
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/core/stable/engines/alpha/test_tool_call_announcement.py`
Expected: 3 passed (Task 3), 4 FAIL — `AttributeError: ... '_start_tool_announcement_task'`

- [ ] **Step 3: Implement engine methods** (after `_generate_preamble`, ~engine.py:874)

```python
    async def _start_tool_announcement_task(
        self,
        context: EngineContext,
        tool_events: Sequence[EmittedEvent],
    ) -> None:
        policy = self._perceived_performance_policy_provider.get_policy(context.agent.id)

        if not await policy.is_tool_call_announcement_required(context, tool_events):
            return

        # Announce only after execution so we never tell the customer about
        # an operation that ends up not running. The generation runs
        # concurrently with the rest of the turn and is collected in
        # _collect_tool_announcements right before the response is composed.
        context.state.tool_announcement_tasks.append(
            asyncio.create_task(
                self._get_message_composer(context.agent).generate_tool_call_announcement(
                    context=context,
                    tool_events=tool_events,
                )
            )
        )

    async def _collect_tool_announcements(self, context: EngineContext) -> None:
        tasks = list(context.state.tool_announcement_tasks)
        context.state.tool_announcement_tasks.clear()

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                # The announcement is a nicety; never let it fail the turn.
                self._logger.warning(f"Tool call announcement failed: {result}")
                continue

            for composition in result:
                context.state.message_events += [e for e in composition.events if e]

    async def _discard_tool_announcements(self, context: EngineContext) -> None:
        tasks = list(getattr(context, "state", None) and context.state.tool_announcement_tasks or [])

        if not tasks:
            return

        context.state.tool_announcement_tasks.clear()

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
```

- [ ] **Step 4: Wire the call sites**

4a. In `_run_initial_preparation_iteration` (~line 660) and `_run_additional_preparation_iteration` (~line 770), extend the existing block:
```python
        if new_tool_events:
            context.state.tool_events += new_tool_events
            self._add_tool_events_to_tracer(new_tool_events)
            await self._start_tool_announcement_task(context, new_tool_events)
```

4b. In `_do_process`'s `uncancellable_section` (~line 340), immediately before the message-generation span:
```python
                # Make sure every pending tool-call announcement has been
                # emitted (and is visible to the response prompt as a staged
                # message) before the final response is composed.
                await self._collect_tool_announcements(context)

                # Money time: communicate with the customer given
                # all of the information we have prepared.
                with self._tracer.span(_MESSAGE_GENERATION_SPAN_NAME):
```

4c. Add a `finally` to `_do_process`'s main `try` (after the existing `except Exception` block, ~line 383):
```python
        finally:
            # No-op on the happy path (collected before message generation);
            # cancels orphans when the turn bails or unwinds early.
            await self._discard_tool_announcements(context)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q tests/core/stable/engines/alpha/test_tool_call_announcement.py tests/core/stable/engines/alpha/test_perceived_performance_policy.py`
Expected: 12 passed

- [ ] **Step 6: Commit**

```bash
git add src/parlant/core/engines/alpha/engine.py tests/core/stable/engines/alpha/test_tool_call_announcement.py
git commit -m "feat(engine): emit tool-call announcements concurrently after tool execution"
```

---

### Task 5: SDK export + final verification

**Files:**
- Modify: `src/parlant/sdk.py:256` (import block) and `__all__`

- [ ] **Step 1: Extend the existing import** (sdk.py ~line 256):

```python
from parlant.core.engines.alpha.perceived_performance_policy import (
    AnnouncingPerceivedPerformancePolicy,
    PerceivedPerformancePolicyProvider,
    ...  # keep existing names
)
```
Check what the block currently imports (`BasicPerceivedPerformancePolicy`, `NullPerceivedPerformancePolicy`, `VoiceOptimizedPerceivedPerformancePolicy`, `PerceivedPerformancePolicy` appear in `__all__`) and add the new name alphabetically.

- [ ] **Step 2: Add to `__all__`** — insert `"AnnouncingPerceivedPerformancePolicy",` between `"AnyOf"` and `"AuthorizationException"`.

- [ ] **Step 3: Verify import and run the scoped suite**

Run: `uv run python -c "from parlant.sdk import AnnouncingPerceivedPerformancePolicy; print('ok')"`
Expected: `ok`

Run: `uv run pytest -q tests/core/stable/engines/alpha/test_perceived_performance_policy.py tests/core/stable/engines/alpha/test_tool_call_announcement.py tests/core/stable/engines/alpha/test_render_section.py`
Expected: 33 passed

- [ ] **Step 4: Commit**

```bash
git add src/parlant/sdk.py
git commit -m "feat(sdk): export AnnouncingPerceivedPerformancePolicy"
```

---

## Follow-up (separate plan, interactive-agent repo)

1. Schemas: optional `tool_announcements:` block under `preamble:` (minor bump).
2. Applicator: block present → `perceived_performance_policy=p.AnnouncingPerceivedPerformancePolicy()`.
3. Patches: add `generate_tool_call_announcement` to `_CHAT_BUDGET_TARGETS` as skippable with a tight budget.
4. Pin bump: `server/pyproject.toml` rev + re-lock `server/uv.lock`.
5. E2E: chat smoke with a tool-bearing routine; verify announcement arrives as `Preamble` wire event.
