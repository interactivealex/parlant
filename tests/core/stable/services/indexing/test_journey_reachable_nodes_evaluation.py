from typing import Any, Mapping, Sequence, cast

from pytest import raises
from typing_extensions import override

from parlant.core.engines.alpha.optimization_policy import BasicOptimizationPolicy
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import (
    FallbackSchematicGenerator,
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.services.indexing.common import EvaluationError
from parlant.core.services.indexing.journey_reachable_nodes_evaluation import (
    ChildEvaluation,
    JourneyNodeKind,
    JourneyReachableNodesEvaluator,
    PathCondition,
    ReachableNodesEvaluationSchema,
    _ChildInfo,
    _JourneyNode,
    _ReachableFollowUps,
)
from parlant.core.services.tools.service_registry import ServiceRegistry


class _ScriptedSchematicGenerator(SchematicGenerator[ReachableNodesEvaluationSchema]):
    """A deterministic generator that returns a scripted sequence of responses.

    Records every prompt (built to a string) and hints it receives. Each call
    consumes the next scripted entry; once exhausted, the last entry repeats. An
    entry that is an Exception is raised instead of returned.
    """

    def __init__(
        self,
        responses: Sequence[ReachableNodesEvaluationSchema | Exception],
    ) -> None:
        self._responses = list(responses)
        self.received_prompts: list[str] = []
        self.received_hints: list[Mapping[str, Any]] = []

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[ReachableNodesEvaluationSchema]:
        self.received_prompts.append(
            prompt.build() if isinstance(prompt, PromptBuilder) else prompt
        )
        self.received_hints.append(dict(hints))

        call_index = len(self.received_prompts) - 1
        response = self._responses[min(call_index, len(self._responses) - 1)]

        if isinstance(response, Exception):
            raise response

        return SchematicGenerationResult(
            content=response,
            info=GenerationInfo(
                schema_name="ReachableNodesEvaluationSchema",
                model="scripted",
                duration=0.0,
                usage=UsageInfo(input_tokens=0, output_tokens=0),
            ),
        )

    @property
    @override
    def id(self) -> str:
        return "scripted"

    @property
    @override
    def max_tokens(self) -> int:
        return 4096

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()


def _make_evaluator(
    logger: Logger,
    schematic_generator: SchematicGenerator[ReachableNodesEvaluationSchema],
) -> JourneyReachableNodesEvaluator:
    return JourneyReachableNodesEvaluator(
        logger=logger,
        optimization_policy=BasicOptimizationPolicy(),
        schematic_generator=schematic_generator,
        service_registry=cast(ServiceRegistry, None),
    )


def _make_graph_with_tool_child() -> tuple[dict[str, _JourneyNode], dict[str, _ChildInfo]]:
    # Parent "1" (a chat step) has a single TOOL child "2". A TOOL child has an
    # empty id_to_reachable_follow_ups, so the prompt offers it NO forward-path
    # ids — yet the model may still hallucinate one.
    parent = _JourneyNode(
        id="1",
        action="ask the customer for their name",
        incoming_edges=[],
        outgoing_edges=[],
        kind=JourneyNodeKind.CHAT,
        customer_dependent_action=True,
    )
    child = _JourneyNode(
        id="2",
        action="run the lookup tool",
        incoming_edges=[],
        outgoing_edges=[],
        kind=JourneyNodeKind.TOOL,
        customer_dependent_action=False,
    )
    new_graph = {"1": parent, "2": child}
    children_info = {
        "2": _ChildInfo(
            action="run the lookup tool",
            edge_condition=None,
            id_to_reachable_follow_ups={},
        )
    }
    return new_graph, children_info


def _response_for_tool_child(forward_id: str | None = None) -> ReachableNodesEvaluationSchema:
    # A response for the parent of the TOOL child "2". When forward_id is given,
    # the model invents a forward path for that child (which has none); otherwise
    # it returns only the valid "reach child and stop" transition.
    return ReachableNodesEvaluationSchema(
        step_action="ask the customer for their name",
        step_action_completed="the customer provided their name",
        children_conditions=[
            ChildEvaluation(
                child_id="2",
                child_action="run the lookup tool",
                condition_to_child="the customer provided their name",
                condition_to_child_and_stop="the customer provided their name but the tool hasn't run",
                conditions_to_child_and_forward=(
                    [
                        PathCondition(
                            id=forward_id,
                            path_condition="some invented downstream condition",
                            condition_to_child_then_to_path="invented combined condition",
                        )
                    ]
                    if forward_id is not None
                    else None
                ),
            )
        ],
    )


_EXPECTED_STOP_TRANSITION = _ReachableFollowUps(
    condition="the customer provided their name but the tool hasn't run",
    path=["2"],
)


def _make_graph_with_forwardable_child() -> tuple[dict[str, _JourneyNode], dict[str, _ChildInfo]]:
    # Parent "1" has a CHAT child "2" that DOES have a forward path (offered id
    # "1"). When the model references an id other than "1", that's a recoverable
    # mistake — the model could pick the offered id on a reask.
    parent = _JourneyNode(
        id="1",
        action="ask the customer for their name",
        incoming_edges=[],
        outgoing_edges=[],
        kind=JourneyNodeKind.CHAT,
        customer_dependent_action=True,
    )
    child = _JourneyNode(
        id="2",
        action="confirm the customer's details",
        incoming_edges=[],
        outgoing_edges=[],
        kind=JourneyNodeKind.CHAT,
        customer_dependent_action=True,
    )
    new_graph = {"1": parent, "2": child}
    children_info = {
        "2": _ChildInfo(
            action="confirm the customer's details",
            edge_condition=None,
            id_to_reachable_follow_ups={
                "1": _ReachableFollowUps(condition="the customer confirmed", path=["3"])
            },
        )
    }
    return new_graph, children_info


def _response_for_forwardable_child(forward_id: str) -> ReachableNodesEvaluationSchema:
    return ReachableNodesEvaluationSchema(
        step_action="ask the customer for their name",
        step_action_completed="the customer provided their name",
        children_conditions=[
            ChildEvaluation(
                child_id="2",
                child_action="confirm the customer's details",
                condition_to_child="the customer provided their name",
                condition_to_child_and_stop="reached the child but details aren't confirmed",
                conditions_to_child_and_forward=[
                    PathCondition(
                        id=forward_id,
                        path_condition="the customer confirmed",
                        condition_to_child_then_to_path="details confirmed and the customer confirmed",
                    )
                ],
            )
        ],
    )


_EXPECTED_FORWARDABLE_STOP = _ReachableFollowUps(
    condition="reached the child but details aren't confirmed",
    path=["2"],
)
_EXPECTED_FORWARDABLE_FORWARD = _ReachableFollowUps(
    condition="details confirmed and the customer confirmed",
    path=["2", "3"],
)


async def test_that_a_wrong_offered_forward_path_id_triggers_a_reask_carrying_the_error(
    logger: Logger,
) -> None:
    # The child has an offered forward id ("1"); the model first picks "9" (wrong
    # but recoverable), then corrects to "1" on the reask.
    generator = _ScriptedSchematicGenerator(
        [_response_for_forwardable_child("9"), _response_for_forwardable_child("1")]
    )
    evaluator = _make_evaluator(logger, generator)
    new_graph, children_info = _make_graph_with_forwardable_child()

    result = await evaluator.do_node_evaluation(new_graph, "1", children_info)

    assert list(result) == [_EXPECTED_FORWARDABLE_STOP, _EXPECTED_FORWARDABLE_FORWARD]

    assert len(generator.received_prompts) == 2
    assert "CORRECTION REQUIRED" not in generator.received_prompts[0]
    assert "CORRECTION REQUIRED" in generator.received_prompts[1]
    assert "9" in generator.received_prompts[1]

    expected_first_temp = BasicOptimizationPolicy().get_guideline_matching_batch_retry_temperatures(
        hints={"type": "JourneyReachableNodesEvaluator"}
    )[0]
    assert generator.received_hints[0]["temperature"] == expected_first_temp
    assert generator.received_hints[1]["temperature"] == 0.0


async def test_that_persistently_wrong_forward_path_ids_are_dropped_after_retries_instead_of_crashing(
    logger: Logger,
) -> None:
    # The model keeps picking an offered-but-wrong id ("9"); after exhausting base
    # and fallback the bogus forward path is dropped and the stop transition kept.
    base_generator = _ScriptedSchematicGenerator([_response_for_forwardable_child("9")])
    fallback_generator = _ScriptedSchematicGenerator([_response_for_forwardable_child("9")])
    fallback = FallbackSchematicGenerator(base_generator, fallback_generator, logger=logger)

    evaluator = _make_evaluator(logger, fallback)
    new_graph, children_info = _make_graph_with_forwardable_child()

    result = await evaluator.do_node_evaluation(new_graph, "1", children_info)

    assert list(result) == [_EXPECTED_FORWARDABLE_STOP]

    # Three reask attempts on the base model, then three on the fallback model.
    assert len(base_generator.received_prompts) == 3
    assert len(fallback_generator.received_prompts) == 3

    expected_first_temp = BasicOptimizationPolicy().get_guideline_matching_batch_retry_temperatures(
        hints={"type": "JourneyReachableNodesEvaluator"}
    )[0]
    assert base_generator.received_hints[0]["temperature"] == expected_first_temp
    reask_temps = [h["temperature"] for h in base_generator.received_hints[1:]] + [
        h["temperature"] for h in fallback_generator.received_hints
    ]
    assert reask_temps == [0.0, 0.0, 0.0, 0.0, 0.0]


async def test_that_an_invented_forward_path_id_for_a_child_without_forward_paths_is_dropped_without_reasking(
    logger: Logger,
) -> None:
    # The TOOL child "2" has NO forward paths, so an invented forward id can never
    # be recovered by reasking. It must be dropped on the first attempt with no
    # reask — otherwise every such node costs base+fallback generations for nothing.
    generator = _ScriptedSchematicGenerator([_response_for_tool_child("1")])
    evaluator = _make_evaluator(logger, generator)
    new_graph, children_info = _make_graph_with_tool_child()

    result = await evaluator.do_node_evaluation(new_graph, "1", children_info)

    assert list(result) == [_EXPECTED_STOP_TRANSITION]
    assert len(generator.received_prompts) == 1
    assert "CORRECTION REQUIRED" not in generator.received_prompts[0]


async def test_that_a_hallucinated_child_id_is_reported_and_dropped(
    logger: Logger,
) -> None:
    invented_child_response = ReachableNodesEvaluationSchema(
        step_action="ask the customer for their name",
        step_action_completed="the customer provided their name",
        children_conditions=[
            ChildEvaluation(
                child_id="999",
                child_action="a step that does not exist",
                condition_to_child="some condition",
                condition_to_child_and_stop="some condition",
                conditions_to_child_and_forward=None,
            )
        ],
    )
    generator = _ScriptedSchematicGenerator([invented_child_response])
    evaluator = _make_evaluator(logger, generator)
    new_graph, children_info = _make_graph_with_tool_child()

    result = await evaluator.do_node_evaluation(new_graph, "1", children_info)

    assert list(result) == []
    assert len(generator.received_prompts) == 3
    assert "CORRECTION REQUIRED" not in generator.received_prompts[0]
    assert "CORRECTION REQUIRED" in generator.received_prompts[1]
    assert "999" in generator.received_prompts[-1]


async def test_that_evaluation_error_is_raised_when_every_generation_attempt_throws(
    logger: Logger,
) -> None:
    underlying = RuntimeError("model unavailable")
    generator = _ScriptedSchematicGenerator([underlying])
    evaluator = _make_evaluator(logger, generator)
    new_graph, children_info = _make_graph_with_tool_child()

    with raises(EvaluationError) as exc_info:
        await evaluator.do_node_evaluation(new_graph, "1", children_info)

    # The original generation failure is preserved as the chained cause.
    assert exc_info.value.__cause__ is underlying
    assert len(generator.received_prompts) == 3


def test_that_a_plain_generator_reports_itself_as_its_only_generator(
    logger: Logger,
) -> None:
    generator = _ScriptedSchematicGenerator([_response_for_tool_child()])

    assert list(generator.generators) == [generator]


def test_that_nested_fallback_generators_flatten_in_order(
    logger: Logger,
) -> None:
    base = _ScriptedSchematicGenerator([_response_for_tool_child()])
    inner_a = _ScriptedSchematicGenerator([_response_for_tool_child()])
    inner_b = _ScriptedSchematicGenerator([_response_for_tool_child()])
    inner_fallback = FallbackSchematicGenerator(inner_a, inner_b, logger=logger)
    outer_fallback = FallbackSchematicGenerator(base, inner_fallback, logger=logger)

    # The evaluator drives `.generators` directly, so flattening a composite into
    # its concrete delegates (base first) is what gives it base->fallback order
    # without knowing the concrete generator type.
    assert list(outer_fallback.generators) == [base, inner_a, inner_b]
