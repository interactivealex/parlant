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

import pytest
from lagom import Container
from openai import BadRequestError

from parlant.adapters.nlp.common import GenerationRefusedError
from parlant.adapters.nlp.openai_service import GPT_4o
from parlant.core.common import DefaultBaseModel
from parlant.core.loggers import Logger, StdoutLogger
from parlant.core.meter import LocalMeter, Meter
from parlant.core.tracer import LocalTracer, Tracer


class SimpleSchema(DefaultBaseModel):
    value: str


@pytest.fixture
def container() -> Container:
    c = Container()
    tracer = LocalTracer()
    logger = StdoutLogger(tracer)
    meter = LocalMeter(logger)
    c[Logger] = logger
    c[Tracer] = tracer
    c[Meter] = meter
    return c


def _make_usage(
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cached_tokens: int | None = 0,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    if cached_tokens is not None:
        details = MagicMock()
        details.cached_tokens = cached_tokens
        usage.prompt_tokens_details = details
    else:
        usage.prompt_tokens_details = None
    return usage


def _make_parse_response(
    parsed: Any,
    refusal: str | None = None,
    usage: Any = None,
) -> MagicMock:
    msg = MagicMock()
    msg.parsed = parsed
    msg.refusal = refusal
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage if usage is not None else _make_usage()
    return resp


def _make_create_response(
    content: str,
    usage: Any = None,
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage if usage is not None else _make_usage()
    return resp


def _make_generator(container: Container) -> GPT_4o[SimpleSchema]:  # type: ignore[type-arg]
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
        gen: GPT_4o[SimpleSchema] = GPT_4o[SimpleSchema](  # type: ignore[assignment]
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
        )
    return gen


# ---------------------------------------------------------------------------
# (a) Default path uses chat.completions.parse with response_format=schema class
# ---------------------------------------------------------------------------


def test_that_default_path_calls_parse_with_schema_class(container: Container) -> None:
    gen = _make_generator(container)
    parsed = SimpleSchema(value="hello")
    resp = _make_parse_response(parsed=parsed)

    with patch.object(
        gen._client.chat.completions,
        "parse",
        new_callable=AsyncMock,
        return_value=resp,
    ) as mock_parse:
        result = asyncio.run(gen.do_generate("test prompt"))

    mock_parse.assert_called_once()
    call_kwargs = mock_parse.call_args.kwargs
    assert call_kwargs["response_format"] is SimpleSchema
    assert result.content == parsed


# ---------------------------------------------------------------------------
# (b) hints strict=False forces the legacy json_object path (create)
# ---------------------------------------------------------------------------


def test_that_strict_false_hint_uses_json_object_create(container: Container) -> None:
    gen = _make_generator(container)
    resp = _make_create_response(content='{"value": "world"}')

    with (
        patch.object(
            gen._client.chat.completions,
            "parse",
            new_callable=AsyncMock,
        ) as mock_parse,
        patch.object(
            gen._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ) as mock_create,
    ):
        result = asyncio.run(gen.do_generate("test prompt", hints={"strict": False}))

    mock_parse.assert_not_called()
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert result.content.value == "world"


# ---------------------------------------------------------------------------
# (c) BadRequestError on parse falls back to json_object and succeeds;
#     second call skips parse entirely
# ---------------------------------------------------------------------------


def test_that_bad_request_error_on_parse_falls_back_to_json_object(container: Container) -> None:
    gen = _make_generator(container)
    resp = _make_create_response(content='{"value": "fallback"}')

    bad_request = BadRequestError(
        message="Invalid schema for response_format",
        response=MagicMock(status_code=400, headers={}),
        body={"error": {"message": "Invalid schema for response_format"}},
    )

    with (
        patch.object(
            gen._client.chat.completions,
            "parse",
            new_callable=AsyncMock,
            side_effect=bad_request,
        ) as mock_parse,
        patch.object(
            gen._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ) as mock_create,
    ):
        result = asyncio.run(gen.do_generate("test prompt"))

    mock_parse.assert_called_once()
    mock_create.assert_called_once()
    assert result.content.value == "fallback"
    assert gen._strict_incompatible is True


def test_that_second_call_after_bad_request_skips_parse(container: Container) -> None:
    gen = _make_generator(container)
    resp = _make_create_response(content='{"value": "cached-fallback"}')

    bad_request = BadRequestError(
        message="Invalid schema for response_format",
        response=MagicMock(status_code=400, headers={}),
        body={"error": {"message": "Invalid schema for response_format"}},
    )

    with (
        patch.object(
            gen._client.chat.completions,
            "parse",
            new_callable=AsyncMock,
            side_effect=bad_request,
        ) as mock_parse,
        patch.object(
            gen._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ) as mock_create,
    ):
        asyncio.run(gen.do_generate("first"))
        asyncio.run(gen.do_generate("second"))

    # parse called once (first call failed), not called on second call
    assert mock_parse.call_count == 1
    assert mock_create.call_count == 2


# ---------------------------------------------------------------------------
# (d) Refusal raises GenerationRefusedError
# ---------------------------------------------------------------------------


def test_that_refusal_raises_generation_refused_error(container: Container) -> None:
    gen = _make_generator(container)
    resp = _make_parse_response(parsed=None, refusal="I cannot help with that.")

    with patch.object(
        gen._client.chat.completions,
        "parse",
        new_callable=AsyncMock,
        return_value=resp,
    ):
        with pytest.raises(GenerationRefusedError) as exc_info:
            asyncio.run(gen.do_generate("test prompt"))

    assert "I cannot help with that." in str(exc_info.value)


# ---------------------------------------------------------------------------
# (e) Missing usage / prompt_tokens_details does not crash; records 0s
# ---------------------------------------------------------------------------


def test_that_missing_usage_on_parse_path_does_not_crash(container: Container) -> None:
    gen = _make_generator(container)
    parsed = SimpleSchema(value="ok")
    resp = _make_parse_response(parsed=parsed, usage=None)
    resp.usage = None  # no usage at all

    with patch.object(
        gen._client.chat.completions,
        "parse",
        new_callable=AsyncMock,
        return_value=resp,
    ):
        result = asyncio.run(gen.do_generate("test prompt"))

    assert result.content == parsed
    assert result.info.usage.input_tokens == 0
    assert result.info.usage.output_tokens == 0


def test_that_missing_prompt_tokens_details_on_parse_path_does_not_crash(
    container: Container,
) -> None:
    gen = _make_generator(container)
    parsed = SimpleSchema(value="ok")
    resp = _make_parse_response(parsed=parsed, usage=_make_usage(cached_tokens=None))

    with patch.object(
        gen._client.chat.completions,
        "parse",
        new_callable=AsyncMock,
        return_value=resp,
    ):
        result = asyncio.run(gen.do_generate("test prompt"))

    assert result.content == parsed
    assert result.info.usage.extra is not None
    assert result.info.usage.extra.get("cached_input_tokens", -1) == 0


def test_that_missing_usage_on_json_object_path_does_not_crash(container: Container) -> None:
    gen = _make_generator(container)
    resp = _make_create_response(content='{"value": "no-usage"}')
    resp.usage = None

    with patch.object(
        gen._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=resp,
    ):
        result = asyncio.run(gen.do_generate("test prompt", hints={"strict": False}))

    assert result.content.value == "no-usage"
    assert result.info.usage.input_tokens == 0
    assert result.info.usage.output_tokens == 0
