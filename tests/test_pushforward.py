"""Regression tests for the PushforwardTrainer warmup path.

The base Trainer rejects multi-step targets in train mode (training.py:498-499,
"Multiple step prediction in train mode not yet supported"). Pushforward needs
`n_steps_output >= pushforward_steps` to seed the advance, so during the warmup
ramp -- when the effective K collapses to 1 and no advance runs -- the batch must
still be reduced to a single future frame before it reaches the base train path.
"""

import pytest
import torch

from walrus.trainer import PushforwardTrainer
from walrus.trainer.training import Trainer

T_IN = 3
N = 8
C = 1


class DummyModel:
    causal_in_time = False


def make_batch(n_steps_output):
    return {
        "input_fields": torch.zeros(2, T_IN, N, C),
        "output_fields": torch.arange(
            2 * n_steps_output * N * C, dtype=torch.float32
        ).reshape(2, n_steps_output, N, C),
        "metadata": object(),  # identity-checked below: must be passed through
    }


def make_trainer(steps, warmup, epoch):
    """A PushforwardTrainer with only the attributes rollout_model touches --
    the real __init__ needs a full training stack (data, optimizer, logger)."""
    trainer = PushforwardTrainer.__new__(PushforwardTrainer)
    trainer.pushforward_steps = steps
    trainer.pushforward_warmup_epochs = warmup
    trainer._current_epoch = epoch
    return trainer


@pytest.fixture
def captured_base(monkeypatch):
    """Replace the base rollout with a capture so the test targets the batch
    preprocessing in PushforwardTrainer, not the full training stack."""
    seen = {}

    def fake_rollout(self, model, batch, formatter, train=True, fake_pass=False):
        seen["batch"] = batch
        seen["train"] = train
        return torch.zeros(1), {}

    monkeypatch.setattr(Trainer, "rollout_model", fake_rollout)
    return seen


def test_warmup_ramp_holds_k_at_one():
    """K=2, W=5 -> epoch 1 still teacher-forces. This is the state that crashed."""
    assert make_trainer(2, 5, 1)._effective_pushforward_steps() == 1
    assert make_trainer(2, 5, 0)._effective_pushforward_steps() == 1
    assert make_trainer(2, 5, 5)._effective_pushforward_steps() == 2


def test_warmup_slices_target_to_single_frame(captured_base):
    """k == 1: no advance, but the 2-frame target must be cut to 1 frame."""
    trainer = make_trainer(steps=2, warmup=5, epoch=1)
    batch = make_batch(n_steps_output=2)

    trainer.rollout_model(DummyModel(), batch, formatter=None, train=True)

    forwarded = captured_base["batch"]
    assert forwarded["output_fields"].shape[1] == 1
    # Must be the FIRST future frame, not an arbitrary slice.
    assert torch.equal(forwarded["output_fields"], batch["output_fields"][:, :1])
    # Input window is untouched during teacher-forcing.
    assert torch.equal(forwarded["input_fields"], batch["input_fields"])


def test_warmup_does_not_mutate_caller_batch(captured_base):
    """The slice is a shallow copy: the caller's batch keeps both frames, and
    unrelated entries are passed through by reference."""
    trainer = make_trainer(steps=2, warmup=5, epoch=1)
    batch = make_batch(n_steps_output=2)
    metadata = batch["metadata"]

    trainer.rollout_model(DummyModel(), batch, formatter=None, train=True)

    assert batch["output_fields"].shape[1] == 2
    assert captured_base["batch"]["metadata"] is metadata


def test_eval_path_keeps_full_target(captured_base):
    """Eval rolls out multiple steps -- slicing there would break it."""
    trainer = make_trainer(steps=2, warmup=5, epoch=1)
    batch = make_batch(n_steps_output=2)

    trainer.rollout_model(DummyModel(), batch, formatter=None, train=False)

    assert captured_base["batch"]["output_fields"].shape[1] == 2


def test_fake_pass_keeps_full_target(captured_base):
    trainer = make_trainer(steps=2, warmup=5, epoch=1)
    batch = make_batch(n_steps_output=2)

    trainer.rollout_model(
        DummyModel(), batch, formatter=None, train=True, fake_pass=True
    )

    assert captured_base["batch"]["output_fields"].shape[1] == 2


def test_disabled_pushforward_is_untouched(captured_base):
    """pushforward_steps == 1 must be byte-identical to the base Trainer."""
    trainer = make_trainer(steps=1, warmup=0, epoch=0)
    batch = make_batch(n_steps_output=1)

    trainer.rollout_model(DummyModel(), batch, formatter=None, train=True)

    assert captured_base["batch"] is batch


def test_insufficient_n_steps_output_fails_fast(captured_base):
    """Guard fires against the TARGET K, at epoch 0, before the silent no-op."""
    trainer = make_trainer(steps=2, warmup=5, epoch=0)
    batch = make_batch(n_steps_output=1)

    with pytest.raises(ValueError, match="n_steps_output"):
        trainer.rollout_model(DummyModel(), batch, formatter=None, train=True)


def test_causal_model_rejected(captured_base):
    trainer = make_trainer(steps=2, warmup=5, epoch=1)
    batch = make_batch(n_steps_output=2)

    class CausalModel:
        causal_in_time = True

    with pytest.raises(NotImplementedError, match="non-causal"):
        trainer.rollout_model(CausalModel(), batch, formatter=None, train=True)


def test_advance_runs_once_ramp_completes(monkeypatch, captured_base):
    """k > 1 takes the advance path instead of the warmup slice."""
    trainer = make_trainer(steps=2, warmup=5, epoch=5)
    batch = make_batch(n_steps_output=2)
    advanced = make_batch(n_steps_output=1)
    calls = []

    def fake_advance(self, model, b, formatter, n_advance):
        calls.append(n_advance)
        return advanced

    monkeypatch.setattr(PushforwardTrainer, "_advance_window", fake_advance)
    trainer.rollout_model(DummyModel(), batch, formatter=None, train=True)

    assert calls == [1]
    assert captured_base["batch"] is advanced
