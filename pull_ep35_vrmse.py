#!/usr/bin/env python
"""Per-dataset full_VRMSE (mean+median) at a target epoch, pulled from wandb cloud.
Run: with-proxy python pull_ep35_vrmse.py   (env: WB_EPOCH default 35)
Prints a dataset x system table. Use to fill runs that have no local per-dataset pkls
(e.g. medium-flat, twogrid) and to cross-check the pkl-derived numbers.
"""
import os
import wandb

ENTITY = os.environ.get("WB_ENTITY", "amesduong313-self")
PROJECT = os.environ.get("WB_PROJECT", "walrus")
EPOCH = int(os.environ.get("WB_EPOCH", "35"))

# run-name substring -> column label (edit/extend as needed)
WANT = {
    "medium_walrus_devserver_amp_gpu_oom": "medium(flat)",
    "medium_walrus_devserver_amp_gpu_twogrid_oom": "twogrid",
    "large_walrus_devserver_amp_gpu_oom-": "large(flat)",
    "twogrid_highres_coarse_global": "coarse_global",
    "medium_walrus_scaled_devserver_amp_gpu_spectral": "scaled_spec",
    "medium_walrus_devserver_amp_gpu_spectral_002_floor": "spectral",
    "twogrid_gpu_oom_highres": "highres(med)",
}
DSETS = [
    "acoustic_scattering_discontinuous", "acoustic_scattering_inclusions", "active_matter",
    "gray_scott_reaction_diffusion", "helmholtz_staircase", "planetswe", "rayleigh_benard",
    "shear_flow", "turbulent_radiative_layer_2D", "viscoelastic_instability",
]

api = wandb.Api(timeout=120)
runs = list(api.runs(f"{ENTITY}/{PROJECT}"))


def label_for(name):
    for sub, lab in WANT.items():
        if sub in name:
            return lab
    return None


# label -> {dataset -> (mean, median)}
data = {}
for run in runs:
    lab = label_for(run.name)
    if not lab or lab in data:  # take first match per label
        continue
    keys = [f"valid_{ds}/full_VRMSE_T=all_{s}" for ds in DSETS for s in ("mean", "median")]
    row_at_epoch = None
    for row in run.scan_history(keys=keys + ["epoch"]):
        if row.get("epoch") == EPOCH and any(k in row for k in keys):
            row_at_epoch = row
            break
    if row_at_epoch is None:
        print(f"  [warn] {lab}: no epoch={EPOCH} per-dataset row found")
        continue
    data[lab] = {
        ds: (
            row_at_epoch.get(f"valid_{ds}/full_VRMSE_T=all_mean"),
            row_at_epoch.get(f"valid_{ds}/full_VRMSE_T=all_median"),
        )
        for ds in DSETS
    }
    print(f"  [ok] {lab}  ({run.name[:55]})")

cols = [c for c in WANT.values() if c in data]
for j, stat in enumerate(("mean", "median")):
    print(f"\n===== full_VRMSE_T=all_{stat} @ ep{EPOCH} =====")
    hdr = "dataset".ljust(34) + "".join(c.rjust(14) for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for ds in DSETS:
        line = ds[:33].ljust(34)
        for c in cols:
            v = data[c][ds][j]
            line += (f"{v:.4f}".rjust(14) if isinstance(v, (int, float)) else "—".rjust(14))
        print(line)
