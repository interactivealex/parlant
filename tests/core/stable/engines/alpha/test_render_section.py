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

from typing import Any

import pytest

from parlant.core.engines.alpha.prompt_builder import PromptBuilder, render_section


# --- str.format parity (the fast path must be byte-identical) ---------------


@pytest.mark.parametrize(
    ("template", "props"),
    [
        ("{name}", {"name": "Alice"}),
        ("Hello {a} and {b}!", {"a": "x", "b": "y"}),
        ('{{ "key": "{val}" }}', {"val": "x"}),
        ("{{escaped}}", {}),
        ("plain text, no placeholders", {}),
        ("", {}),
        ("{x:.2f}", {"x": 3.14159}),
        ("{v!r}", {"v": "hello"}),
        ("{{{{generative.name}}}}", {}),
        ('JSON example:\n{{\n    "field": "{val}"\n}}', {"val": "ok"}),
    ],
)
def test_render_section_matches_str_format_for_format_safe_templates(
    template: str, props: dict[str, Any]
) -> None:
    assert render_section(template, props) == template.format(**props)


# --- Leniency where str.format raises ----------------------------------------


def test_unknown_placeholder_is_left_verbatim() -> None:
    assert render_section("{unknown}", {}) == "{unknown}"


def test_unknown_placeholder_with_spec_is_left_verbatim() -> None:
    assert render_section("{foo:>10}", {}) == "{foo:>10}"


def test_unknown_placeholder_does_not_block_known_substitution() -> None:
    assert (
        render_section("when {cond}, then 'has_account': {'value': false}", {"cond": "asked"})
        == "when asked, then 'has_account': {'value': false}"
    )


def test_stray_open_brace_is_left_verbatim() -> None:
    assert render_section("{ stray brace", {}) == "{ stray brace"


def test_stray_close_brace_is_left_verbatim() -> None:
    assert render_section("stray }", {}) == "stray }"


def test_json_in_template_with_no_props_is_left_verbatim() -> None:
    template = '{"answer": {"value": true}, "list": [1, 2]}'
    assert render_section(template, {}) == template


def test_double_brace_collapses_even_on_lenient_path() -> None:
    # The stray "{ x" forces the lenient path; the "{{" must still collapse.
    assert render_section("{{literal}} and { x", {}) == "{literal} and { x"


def test_brace_containing_prop_value_is_inserted_verbatim() -> None:
    assert (
        render_section("{agent_description}", {"agent_description": 'persona {"json": true}'})
        == 'persona {"json": true}'
    )


def test_render_section_never_raises_on_adversarial_content() -> None:
    for template in ("{", "}", "{}", "{!r}", "{:>}", "{a{b}", "}{", "{{", "}}"):
        render_section(template, {})  # must not raise
        render_section(template, {"a": 1})  # must not raise


# --- build() integration ------------------------------------------------------


def test_build_renders_brace_containing_section_without_crashing() -> None:
    builder = PromptBuilder()
    builder.add_section(
        name="guidelines-with-braces",
        template="when asked, then reply 'has_account': {'value': false}",
        props={},
    )
    assert "'has_account': {'value': false}" in builder.build()


def test_build_collapses_double_brace_escapes_like_format() -> None:
    builder = PromptBuilder()
    builder.add_section(
        name="json-example",
        template='Output the following:\n{{\n    "produced_reply": false\n}}',
        props={},
    )
    built = builder.build()
    assert '{\n    "produced_reply": false\n}' in built
    assert "{{" not in built
