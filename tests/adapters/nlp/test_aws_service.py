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

import os
from unittest.mock import patch

from parlant.adapters.nlp.aws_service import BedrockService


def test_that_aws_verify_environment_checks_aws_credentials_not_anthropic_key() -> None:
    """verify_environment must check AWS credentials, not ANTHROPIC_API_KEY.
    The adapter uses AsyncAnthropicBedrock which authenticates via AWS credentials."""
    # With no env vars, should return an error about AWS credentials
    with patch.dict(os.environ, {}, clear=True):
        error = BedrockService.verify_environment()
    assert error is not None, "verify_environment must return an error when no AWS creds are set"
    assert "AWS_ACCESS_KEY_ID" in error or "AWS_PROFILE" in error or "AWS_BEARER_TOKEN" in error, (
        f"Error must mention an AWS credential, got: {error!r}"
    )
    assert "ANTHROPIC_API_KEY" not in error, (
        "Error must not mention ANTHROPIC_API_KEY — this adapter uses AWS auth, not Anthropic auth"
    )


def test_that_aws_verify_environment_returns_error_when_only_anthropic_key_is_set() -> None:
    """An Anthropic API key is NOT a valid Bedrock credential — verify_environment
    must still report missing AWS credentials (this was the original bug)."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        error = BedrockService.verify_environment()
    assert error is not None, (
        "verify_environment must return an error when only ANTHROPIC_API_KEY is set"
    )


def test_that_aws_verify_environment_returns_none_when_aws_key_id_and_secret_are_set() -> None:
    """verify_environment must pass when AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY are set."""
    with patch.dict(
        os.environ,
        {"AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE", "AWS_SECRET_ACCESS_KEY": "secret"},
        clear=True,
    ):
        error = BedrockService.verify_environment()
    assert error is None, (
        f"verify_environment should return None when AWS creds are present, got: {error!r}"
    )


def test_that_aws_verify_environment_returns_none_when_aws_profile_is_set() -> None:
    """verify_environment must also accept AWS_PROFILE as a valid credential source."""
    with patch.dict(os.environ, {"AWS_PROFILE": "my-profile"}, clear=True):
        error = BedrockService.verify_environment()
    assert error is None, (
        f"verify_environment should return None when AWS_PROFILE is present, got: {error!r}"
    )
