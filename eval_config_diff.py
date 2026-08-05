#!/usr/bin/env python
"""Diff the *eval-run* configs behind the rollout table, to test whether the ep100
eval passes were actually run the same way.

The models' training configs live in checkpoints/<run>/0/extended_config.yaml, but the
rollout numbers come from separate standalone *eval* runs (the `id:` pins in
eval_rollout_table.py). Those runs' configs only exist on wandb. This pulls them and
prints the fields that would break cross-run comparability.

Agent identity is blocked from api.wandb.ai, so RUN THIS YOURSELF:
    with-proxy python eval_config_diff.py
"""

import os
from collections import defaultdict

import wandb

PROJECT = os.environ.get("WANDB_PROJECT", "walrus")

# label -> eval run id (mirrors MODELS in eval_rollout_table.py, ids only)
EVAL_IDS = {
    "medF@100": "vqthxpeo",
    "medTG@100": "hvmmz5j1",
    "lgF@100": "pvmbwsst",
    "lgHR@100": "kx1ejx35",
    "medHR@100": "m8ljuj01",
    "hetero@100": "d9m7h84b",
    "nostride@100": "5wlyvv8p",
    "specop@100": "as74nl73",
    "hetero_specop@100": "52picxhn",
    "sc21m@100": "97eiqxc0",
    "medF64@100": "6ozutb2p",
    "medFMM_fused@100": "byoaoj0b",
    "lgF_fp32@50": "c2iwjvox",
    "lgF_fp32@100": "0bwam20m",
    "medF_noise05@100": "zcns3t63",
    "medF_reg8@100": "n976ara8",
}

# dotted config paths that change what the eval measures
PROBE = [
    "trainer/max_rollout_steps",
    "trainer/num_time_intervals",
    "trainer/max_val_batches",
    "trainer/validation_one_step_ensemble_size",
    "trainer/validation_full_trajectory_ensemble_size",
    "trainer/enable_amp",
    "trainer/amp_type",
    "trainer/input_noise_std",
    "trainer/rollout_test",
    "data/well_base_path",
    "data/module_parameters/max_samples",
    "data/module_parameters/batch_size",
    "data/module_parameters/n_steps_input",
    "data/module_parameters/n_steps_output",
    "validation_mode",
]

# the four datasets §5.6 claims were source-switched, plus the HF one
DS_PROBE = [
    "active_matter",
    "viscoelastic_instability",
    "turbulent_radiative_layer_2D",
    "helmholtz_staircase",
    "shear_flow",
]


def dig(cfg, path):
    cur = cfg
    for part in path.split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return "<missing>"
    return cur


def main():
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity

    cfgs, meta = {}, {}
    for lbl, rid in EVAL_IDS.items():
        try:
            r = api.run(f"{entity}/{PROJECT}/{rid}")
        except Exception as e:
            print(f"# {lbl:20s} FETCH FAILED ({str(e)[:60]})")
            continue
        cfgs[lbl] = r.config
        meta[lbl] = {
            "created": str(getattr(r, "created_at", "?")),
            "commit": (getattr(r, "commit", None) or "?")[:12],
            "host": (r.metadata or {}).get("host", "?") if hasattr(r, "metadata") else "?",
            "args": " ".join((r.metadata or {}).get("args", []))[:200]
            if hasattr(r, "metadata")
            else "?",
        }

    print("\n" + "=" * 100)
    print("EVAL RUN PROVENANCE (when / where / what code)")
    print("=" * 100)
    for lbl in cfgs:
        m = meta[lbl]
        print(f"{lbl:20s} created={m['created'][:19]}  commit={m['commit']}  host={m['host']}")
        print(f"{'':20s} args= {m['args']}")

    print("\n" + "=" * 100)
    print("EVAL SETTINGS THAT AFFECT COMPARABILITY (value -> which runs)")
    print("=" * 100)
    for p in PROBE:
        groups = defaultdict(list)
        for lbl, c in cfgs.items():
            groups[repr(dig(c, p))].append(lbl)
        flag = "  <<< DIFFERS" if len(groups) > 1 else ""
        print(f"\n{p}{flag}")
        for val, who in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"    {val:44s} <- {', '.join(who)}")

    print("\n" + "=" * 100)
    print("PER-DATASET RESOLVED PATH (tests the §5.6 'data source switch' claim)")
    print("=" * 100)
    for ds in DS_PROBE:
        groups = defaultdict(list)
        for lbl, c in cfgs.items():
            info = dig(c, "data/module_parameters/well_dataset_info")
            base = dig(c, "data/well_base_path")
            if not isinstance(info, dict) or ds not in info:
                p = "<absent>"
            else:
                p = (info[ds] or {}).get("path") or f"<base>{base}"
            groups[str(p)].append(lbl)
        flag = "  <<< DIFFERS" if len(groups) > 1 else ""
        print(f"\n{ds}{flag}")
        for val, who in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"    {val:60s} <- {', '.join(who)}")


if __name__ == "__main__":
    main()
