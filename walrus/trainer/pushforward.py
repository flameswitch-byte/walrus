"""Pushforward training as a Trainer FORK.

Rationale (knowledge_base/walrus_general_addons_pushforward_registers.md): pushforward is
train-rollout control flow that interleaves with the documented-fragile eval path (the
L573-577 normalized-loss vs denorm conflict). So it is gated as a *fork* (a Trainer
subclass selected by config `_target_`), NOT a flag woven into the shared `rollout_model`.

The base `Trainer` and its eval path stay byte-identical. `PushforwardTrainer` only
*preprocesses the batch* in the train path: it advances the input window `K-1` steps using
the model's OWN predictions (via the base eval rollout, under no_grad), shifts the target to
the true frame `K-1` ahead, then calls the unchanged `super().rollout_model` for the single
graded step (grad-on-last pushforward, Brandstetter 2022). This sidesteps the L573-577
blocker structurally: the advance is outside the graded loop and needs no gradient.

Gate: `pushforward_steps` (1 = base behavior). Requires `data.n_steps_output >= pushforward_steps`.
`pushforward_warmup_epochs` linearly ramps the effective K from 1 -> pushforward_steps over the
first N epochs (teacher-force while the model's own predictions are still garbage), 0 = no ramp.
Non-causal models only for now (causal whole-window delta target needs separate handling).
"""

from __future__ import annotations

import torch

from .training import Trainer


class PushforwardTrainer(Trainer):
    def __init__(
        self,
        *args,
        pushforward_steps: int = 1,
        pushforward_warmup_epochs: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Target rollout steps supervised per batch. 1 == teacher-forced one-step
        # (identical to base Trainer). K > 1 == advance K-1 steps on the model's own
        # predictions, then grad on the final (K-th) step.
        self.pushforward_steps = pushforward_steps
        # Linearly ramp the effective K from 1 -> pushforward_steps over the first
        # `pushforward_warmup_epochs` epochs (0 = no ramp, use K immediately). Early in
        # training the model's own predictions are garbage; feeding them back at full K
        # injects far-off-manifold noise, so warm up with teacher-forcing first.
        self.pushforward_warmup_epochs = pushforward_warmup_epochs
        self._current_epoch = 0  # updated by train_one_epoch; read by the ramp

    def train_one_epoch(self, epoch, dataloader):
        # Capture the epoch so rollout_model can compute the ramped K. Base train loop
        # otherwise unchanged.
        self._current_epoch = epoch
        return super().train_one_epoch(epoch, dataloader)

    def _effective_pushforward_steps(self) -> int:
        """Ramped K for the current epoch: 1 at epoch 0, rising linearly to
        `pushforward_steps` by `pushforward_warmup_epochs` (then held)."""
        K = self.pushforward_steps
        W = self.pushforward_warmup_epochs
        if K <= 1 or W <= 0:
            return K
        frac = min(1.0, self._current_epoch / W)
        return 1 + round((K - 1) * frac)

    def rollout_model(self, model, batch, formatter, train=True, fake_pass=False):
        if train and (not fake_pass) and self.pushforward_steps > 1:
            if getattr(model, "causal_in_time", False):
                raise NotImplementedError(
                    "PushforwardTrainer supports non-causal models only; causal "
                    "whole-window delta supervision needs separate handling."
                )
            # Fail fast on misconfiguration: without enough future frames the advance
            # would silently clamp to 0 steps and behave like the baseline (a silent
            # no-op). Check against the TARGET K so it fails at epoch 0, not mid-ramp.
            n_out = batch["output_fields"].shape[1]
            if n_out < self.pushforward_steps:
                raise ValueError(
                    f"pushforward_steps={self.pushforward_steps} requires "
                    f"data.module_parameters.n_steps_output >= {self.pushforward_steps}, "
                    f"but the batch has n_steps_output={n_out}. Set it, or pushforward "
                    f"would silently no-op."
                )
            k = self._effective_pushforward_steps()  # ramped K for this epoch
            if k > 1:
                batch = self._advance_window(model, batch, formatter, k - 1)
            else:
                # Warmup teacher-force (k == 1): no advance, but the batch still
                # carries the multi-frame target (n_steps_output >= pushforward_steps,
                # needed to seed the advance at higher k). Reduce it to the first
                # future frame so the base single-step train path applies -- else the
                # base rollout rejects the >1-step target ("Multiple step prediction
                # in train mode not yet supported").
                batch = dict(batch)  # shallow: keep metadata/bcs/field_indices refs
                batch["output_fields"] = batch["output_fields"][:, :1]
        # Base graded step (train) / eval — completely unchanged.
        return super().rollout_model(
            model, batch, formatter, train=train, fake_pass=fake_pass
        )

    def _advance_window(self, model, batch, formatter, n_advance):
        """Return a batch whose input window is advanced ``n_advance`` steps using the
        model's own (no_grad, denormalized) predictions, with the target set to the true
        frame ``n_advance`` ahead. Reuses the base eval rollout so no per-step denorm/
        channel logic is re-implemented here."""
        T_in = batch["input_fields"].shape[1]
        # Run the advance the way test-time rollout actually runs: model in EVAL mode
        # (dropout / drop_path OFF -> the drift matches inference and is deterministic),
        # and a single forward per step (not the eval ensemble -> avoid ensemble x K cost).
        was_training = model.training
        saved_ensemble = self.validation_one_step_ensemble_size
        model.eval()
        self.validation_one_step_ensemble_size = 1
        try:
            with torch.no_grad():
                # Base eval rollout feeds predictions back -> exactly the advance we want.
                y_pred_traj, _ = super().rollout_model(
                    model, batch, formatter, train=False, fake_pass=False
                )
        finally:
            model.train(was_training)
            self.validation_one_step_ensemble_size = saved_ensemble
        dev = y_pred_traj.device
        inp = batch["input_fields"].to(dev)  # (b, T_in, ..., c_field)
        out = batch["output_fields"].to(dev)  # (b, n_steps_output, ..., c_field)
        # Clamp to what's actually available (short trajectories degrade gracefully).
        n_avail = max(0, min(n_advance, y_pred_traj.shape[1] - 1, out.shape[1] - 1))
        full_seq = torch.cat([inp, y_pred_traj[:, :n_avail]], dim=1)
        new_batch = dict(batch)  # shallow: keep metadata/bcs/field_indices refs
        new_batch["input_fields"] = full_seq[:, -T_in:]  # last T_in frames (advanced)
        new_batch["output_fields"] = out[:, n_avail : n_avail + 1]  # true frame n_avail ahead
        return new_batch
