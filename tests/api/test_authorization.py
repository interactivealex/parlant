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

import pytest
from fastapi import Request
from limits import RateLimitItemPerMinute

from parlant.api.authorization import (
    AuthorizationException,
    Operation,
    BasicRateLimiter,
    ProductionAuthorizationPolicy,
)


def make_request(
    *,
    path: str = "/",
    x_forwarded_for: str | None = "203.0.113.10",
    client_host: str | None = "127.0.0.1",
) -> Request:
    headers = []

    if x_forwarded_for is not None:
        headers.append((b"x-forwarded-for", x_forwarded_for.encode("latin-1")))

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers,
        "client": (client_host, 12345) if client_host is not None else None,
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "http",
        "server": ("testserver", 80),
    }

    return Request(scope)


async def test_that_a_configured_operation_is_limited_per_minute() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(2),
        }
    )

    request = make_request()

    assert await limiter.check(request, Operation.LIST_EVENTS) is True
    assert await limiter.check(request, Operation.LIST_EVENTS) is True
    assert await limiter.check(request, Operation.LIST_EVENTS) is False


async def test_that_limits_are_isolated_per_operation_bucket() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )

    request = make_request()

    assert await limiter.check(request, Operation.LIST_EVENTS) is True
    assert await limiter.check(request, Operation.LIST_EVENTS) is False


async def test_that_limits_are_isolated_per_client_ip() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )

    req_ip1 = make_request(x_forwarded_for="198.51.100.7")
    req_ip2 = make_request(x_forwarded_for="198.51.100.8")

    assert await limiter.check(req_ip1, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_ip2, Operation.LIST_EVENTS) is True

    assert await limiter.check(req_ip1, Operation.LIST_EVENTS) is False


async def test_that_x_forwarded_for_overrides_request_client_host_for_ip_selection() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )

    req_a = make_request(x_forwarded_for="1.1.1.1", client_host="10.0.0.5")
    req_b = make_request(x_forwarded_for="1.1.1.2", client_host="10.0.0.5")

    assert await limiter.check(req_a, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_b, Operation.LIST_EVENTS) is True
    assert await limiter.check(req_a, Operation.LIST_EVENTS) is False


async def test_that_missing_client_ip_raises_authorization_exception() -> None:
    limiter = BasicRateLimiter(
        rate_limit_item_per_operation={
            Operation.LIST_EVENTS: RateLimitItemPerMinute(1),
        }
    )
    request = make_request(x_forwarded_for=None, client_host=None)

    with pytest.raises(AuthorizationException):
        await limiter.check(request, Operation.LIST_EVENTS)


def _default_limiter(policy: ProductionAuthorizationPolicy) -> BasicRateLimiter:
    assert isinstance(policy.default_limiter, BasicRateLimiter)
    return policy.default_limiter


async def test_that_an_env_var_overrides_the_default_rate_limit_for_an_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_RATELIMIT_READ_AGENT", "50")

    policy = ProductionAuthorizationPolicy()
    limits = _default_limiter(policy).rate_limit_item_per_operation

    assert limits[Operation.READ_AGENT] == RateLimitItemPerMinute(50)
    # Operations without an override keep their built-in default.
    assert limits[Operation.LIST_EVENTS] == RateLimitItemPerMinute(240)


async def test_that_rate_limits_use_built_in_defaults_when_no_env_overrides_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for operation in (
        Operation.READ_AGENT,
        Operation.CREATE_GUEST_SESSION,
        Operation.READ_SESSION,
        Operation.LIST_EVENTS,
        Operation.CREATE_CUSTOMER_EVENT,
        Operation.CREATE_STATUS_EVENT,
    ):
        monkeypatch.delenv(f"PARLANT_RATELIMIT_{operation.name}", raising=False)

    policy = ProductionAuthorizationPolicy()
    limits = _default_limiter(policy).rate_limit_item_per_operation

    assert limits[Operation.READ_AGENT] == RateLimitItemPerMinute(30)
    assert limits[Operation.CREATE_GUEST_SESSION] == RateLimitItemPerMinute(10)
    assert limits[Operation.READ_SESSION] == RateLimitItemPerMinute(30)
    assert limits[Operation.LIST_EVENTS] == RateLimitItemPerMinute(240)
    assert limits[Operation.CREATE_CUSTOMER_EVENT] == RateLimitItemPerMinute(30)
    assert limits[Operation.CREATE_STATUS_EVENT] == RateLimitItemPerMinute(60)


async def test_that_an_invalid_rate_limit_env_var_raises_a_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_RATELIMIT_READ_AGENT", "not-a-number")

    with pytest.raises(ValueError):
        ProductionAuthorizationPolicy()


async def test_that_a_non_positive_rate_limit_env_var_raises_a_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PARLANT_RATELIMIT_READ_AGENT", "0")

    with pytest.raises(ValueError):
        ProductionAuthorizationPolicy()
