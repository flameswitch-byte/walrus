"""End-to-end smoke test: medium_walrus_fmm through the full IsotropicModel + Trainer.

Reuses the trainer-test harness (TestDataModule + train) with a SHRUNK medium_walrus_fmm
config, to confirm the FMM space_mixing wires correctly into the encoder/processor/decoder
and trains on a tiny constant-dynamics batch (should drive loss toward ~0, like the existing
isotropic-model trainer test). Not a quality check — just "does the full model run + learn".

    python scratch_fmm_walrus_smoke.py
"""

import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from walrus.train import CONFIG_DIR
from tests.test_trainer_linear_model import B, T, N, C, TestDataModule, train

# The shared test harness predates a trainer API change (train_dataloader now takes a
# sampling_rank arg). Tolerate the extra positional arg so the loop runs (harness-only fix).
for _name in ("train_dataloader", "val_dataloader", "test_dataloader"):
    _orig = getattr(TestDataModule, _name, None)
    if _orig is not None:
        setattr(TestDataModule, _name,
                (lambda f: (lambda self, *a, **k: f(self)))(_orig))

torch.set_num_threads(8)
torch.manual_seed(0)


def main():
    cfg_dir = os.path.abspath(CONFIG_DIR)
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        trainer_cfg = compose(config_name="trainer/debug")
        optimizer_cfg = compose(config_name="optimizer/adam")
        model_cfg = compose(config_name="model/medium_walrus_fmm")

    # shrink to a tiny model; keep num_heads dividing hidden_dim (32/4) and the FMM knobs
    m = model_cfg.model
    with open_dict(m):
        m.processor.space_mixing.num_heads = 4
        m.processor.time_mixing.num_heads = 4
    print("space_mixing target:", m.processor.space_mixing._target_)

    model = instantiate(
        m, processor_blocks=2, hidden_dim=64, n_states=1, groups=2, jitter_patches=False
    )
    n_param = sum(p.numel() for p in model.parameters())
    print(f"instantiated OK — {n_param/1e3:.0f}K params")

    nb_epochs = 8
    optimizer = instantiate(optimizer_cfg.optimizer, params=model.parameters(), lr=1e-2)
    datamodule = TestDataModule(torch.ones(B, T, N, N, C))  # constant dynamics
    loss = train(
        trainer_cfg=trainer_cfg, model=model, optimizer=optimizer,
        lr_scheduler=None, datamodule=datamodule, max_epoch=nb_epochs,
        prediction_type="full", enable_amp=False,
    )
    print(f"\nfinal test loss = {loss:.3e}")
    assert torch.isfinite(torch.tensor(loss)), "non-finite loss"
    print("[OK] FMM runs end-to-end through IsotropicModel + Trainer"
          + (" and learns (~0)" if loss < 1e-3 else " (loss not ~0, inspect)"))


if __name__ == "__main__":
    main()
