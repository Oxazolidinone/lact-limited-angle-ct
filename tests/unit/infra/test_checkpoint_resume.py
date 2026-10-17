"""Checkpoints must restore the GPU RNG and never silently restart from step 0."""
import random

import numpy as np
import pytest
import torch

from lact.infra.stores.run_store import RunStore
from lact.infra.system.runtime import rng_state, set_rng_state


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rng_round_trip_with_checkpoint_map_location(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is needed to check that checkpoints move to the GPU")
    original = rng_state()

    def draw():
        values = [random.random(), np.random.random(), torch.rand(4)]
        if device == "cuda":
            values.append(torch.rand(4, device="cuda").cpu())
        return values

    try:
        run = RunStore(tmp_path)
        run.save_checkpoint("last", torch.nn.Linear(2, 2), step=7)
        expected = draw()
        draw()
        state = run.load_latest("last", map_location=device, restore_rng=True)
        assert state["step"] == 7
        actual = draw()
        assert actual[:2] == expected[:2]
        for a, b in zip(actual[2:], expected[2:]):
            assert torch.equal(a, b)
    finally:
        set_rng_state(original)


def test_missing_checkpoints_allow_new_run(tmp_path):
    assert RunStore(tmp_path).load_latest("last") is None


@pytest.mark.parametrize("has_backup", [False, True])
def test_unreadable_checkpoints_stop_resume(tmp_path, has_backup):
    run = RunStore(tmp_path)
    run.checkpoint("last").write_bytes(b"truncated checkpoint")
    if has_backup:
        run.checkpoint("last_prev").write_bytes(b"truncated backup")
    with pytest.raises(RuntimeError, match="Cannot resume from a checkpoint"):
        run.load_latest("last", restore_rng=True)
    assert run.checkpoint("last").read_bytes() == b"truncated checkpoint"


def test_unreadable_last_uses_valid_backup(tmp_path):
    run = RunStore(tmp_path)
    run.save_checkpoint("last_prev", torch.nn.Linear(2, 2), step=12)
    run.checkpoint("last").write_bytes(b"truncated checkpoint")
    assert run.load_latest("last")["step"] == 12
    assert any(event["event"] == "checkpoint_fallback" for event in run.events())
