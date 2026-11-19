# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_qwen3_5_cp_q_push",
    os.path.join(os.path.dirname(__file__), "test_qwen3_5_cp_q_push.py"),
)
cp_q_push = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp_q_push)

pytestmark = cp_q_push.pytestmark
# the multi-request draft steps' widths
REQUESTS = (2, 4, 8, 12, 16)


@pytest.fixture(autouse=True)
def draft_step_layout(monkeypatch):
    # one q row a request
    monkeypatch.setattr(cp_q_push, "QL", 1)


@pytest.mark.parametrize("requests", REQUESTS)
def test_the_pushed_draft_rows_and_the_wait_half_give_the_full_gathers_bytes(requests):
    cp_q_push.test_the_pushed_words_and_the_wait_half_give_the_full_gathers_bytes(
        requests
    )


@pytest.mark.parametrize("requests", REQUESTS)
def test_lamport_draft_rows_give_the_full_gathers_bytes_and_rearm(requests):
    cp_q_push.test_lamport_words_give_the_full_gathers_bytes_and_rearm(requests)


@pytest.mark.parametrize("requests", REQUESTS)
def test_the_gathering_draft_front_lands_the_full_gathers_bytes(requests):
    cp_q_push.test_the_gathering_q_writer_lands_the_full_gathers_bytes(requests)
