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
import time

import pytest
from lagom import Container

from parlant.core.agents import Agent
from parlant.core.emissions import EventEmitterFactory
from parlant.core.loggers import StdoutLogger
from parlant.core.services.tools.mcp_service import MCPToolClient, MCPToolServer
from parlant.core.tools import ToolError
from parlant.core.tracer import LocalTracer
from parlant.sdk import ToolContext
from tests.core.stable.engines.alpha.test_mcp import create_client, greet_me_like_pirate
from tests.test_utilities import SERVER_BASE_URL, get_random_port


async def slow_tool(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return f"slept {seconds}"


# --- Cancellation guard (the trigger fix) ----------------------------------
#
# A turn barge-in cancels in-flight preparation-phase tool calls. If the MCP
# client abruptly abandons the in-flight HTTP request mid-stream, it can leave a
# stateless MCP server's response handler writing to a half-open connection,
# which escapes into the shared session-manager task group and kills it
# permanently (fastmcp#823, modelcontextprotocol/python-sdk#1104, both open).
# The client must therefore *drain* an in-flight request (let it finish) within
# a bounded grace window before propagating the cancellation.


async def test_that_a_cancelled_mcp_call_drains_the_in_flight_request_before_propagating(
    container: Container,
    agent: Agent,
) -> None:
    async with MCPToolServer([slow_tool], port=get_random_port()) as server:
        client = create_client(server, container)

        async with client:
            task = asyncio.create_task(
                client.call_tool("slow_tool", ToolContext("", "", ""), {"seconds": 0.5})
            )
            await asyncio.sleep(0.1)  # let the request reach the server

            start = time.monotonic()
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            elapsed = time.monotonic() - start

            # Without the guard the cancellation propagates instantly (~0s),
            # abandoning the in-flight request. With the guard the call waits
            # for the request to drain (~the remaining ~0.4s) before re-raising.
            assert elapsed >= 0.2


async def test_that_a_cancelled_mcp_call_aborts_after_the_grace_window_for_a_slow_request(
    container: Container,
    agent: Agent,
) -> None:
    async with MCPToolServer([slow_tool], port=get_random_port()) as server:
        client = create_client(server, container)
        client._cancellation_grace_seconds = 0.3  # type: ignore[attr-defined]

        async with client:
            task = asyncio.create_task(
                client.call_tool("slow_tool", ToolContext("", "", ""), {"seconds": 3.0})
            )
            await asyncio.sleep(0.1)  # let the request reach the server

            start = time.monotonic()
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            elapsed = time.monotonic() - start

            # The guard waits up to the grace window (~0.3s) for the request to
            # drain, then gives up rather than blocking for the full 3s request.
            assert elapsed >= 0.2
            assert elapsed < 1.5

            # After aborting an undrained request the connection is poisoned, so
            # the next call transparently reconnects and still works.
            result = await client.call_tool("slow_tool", ToolContext("", "", ""), {"seconds": 0.0})
            assert "slept 0.0" in result.data


# --- Reconnect / retry (ported from upstream eb71f4ee5) --------------------


async def test_that_mcp_client_reconnects_after_its_session_is_closed(
    container: Container,
    agent: Agent,
) -> None:
    async with MCPToolServer([greet_me_like_pirate], port=get_random_port()) as server:
        client = create_client(server, container)

        async with client:
            result = await client.call_tool(
                "greet_me_like_pirate",
                ToolContext("", "", ""),
                {"name": "Short Jon Nickel", "lucky_number": 7},
            )
            assert "Ahoy Short Jon Nickel! I doubled your lucky number to 14 !" in result.data

            assert client._client is not None
            await client._client.close()  # type: ignore[no-untyped-call]

            reconnected_result = await client.call_tool(
                "greet_me_like_pirate",
                ToolContext("", "", ""),
                {"name": "Another Pirate", "lucky_number": 9},
            )

            assert (
                "Ahoy Another Pirate! I doubled your lucky number to 18 !"
                in reconnected_result.data
            )


async def test_that_mcp_client_retries_initial_connection(
    container: Container,
) -> None:
    client = MCPToolClient(
        url=SERVER_BASE_URL,
        event_emitter_factory=container[EventEmitterFactory],
        logger=StdoutLogger(LocalTracer()),
        tracer=LocalTracer(),
        port=get_random_port(),
    )

    class FakeClient:
        def __init__(self, should_fail: bool) -> None:
            self.should_fail = should_fail
            self.connected = False

        async def __aenter__(self) -> "FakeClient":
            if self.should_fail:
                raise asyncio.TimeoutError()
            self.connected = True
            return self

        async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
            self.connected = False
            return False

        def is_connected(self) -> bool:
            return self.connected

    attempted_clients: list[FakeClient] = []

    def fake_create_client() -> FakeClient:
        fake_client = FakeClient(should_fail=not attempted_clients)
        attempted_clients.append(fake_client)
        return fake_client

    client._create_client = fake_create_client  # type: ignore[method-assign, assignment]

    async with client:
        assert len(attempted_clients) == 2
        assert attempted_clients[-1].is_connected()


# --- Error-message hygiene -------------------------------------------------


async def test_that_read_tool_for_unknown_tool_names_the_tool_in_the_error(
    container: Container,
    agent: Agent,
) -> None:
    async with MCPToolServer([greet_me_like_pirate], port=get_random_port()) as server:
        client = create_client(server, container)

        async with client:
            with pytest.raises(ToolError) as exc_info:
                await client.read_tool("does_not_exist")

            assert "does_not_exist" in str(exc_info.value)
