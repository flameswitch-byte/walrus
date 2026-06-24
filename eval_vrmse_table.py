#!/usr/bin/env python
"""Per-dataset one-step full_VRMSE_T=all_mean at a fixed epoch, across all comparison runs.

Pulls epoch-aligned values straight from the wandb API (the reliable source) and prints
a dataset x model table plus the equal-weight mean-across-datasets ranking row.

Usage (from an authenticated host, through the proxy):
    with-proxy python eval_vrmse_table.py --list     # list all run names (to fill MODELS)
    with-proxy python eval_vrmse_table.py            # epoch 70 (default)
    with-proxy python eval_vrmse_table.py 65         # any epoch
    with-proxy python eval_vrmse_table.py 35 --median
    with-proxy python eval_vrmse_table.py 100 --fields       # + per-field VRMSE (pressure etc.)
    with-proxy python eval_vrmse_table.py 100 --spectral     # + per-k-bin spectral_error_mse (toy_full)
    with-proxy python eval_vrmse_table.py 100 --fields --spectral --median
    WANDB_ENTITY=<you> with-proxy python eval_vrmse_table.py 70

Notes:
- A MODELS value may be an exact experiment NAME (auto-follows the furthest-along run) OR
  "id:<wandb_run_id>" to PIN one exact run (use when a name is reused across runs, or to be
  sure you're reading the right training run's one-step metrics). The TARGET-epoch selection
  still applies to a pinned run -- the id picks the run, its history is read at TARGET.
  (Same id: convention as eval_rollout_table.py.) Use `--list` to find run ids.
- If a run has no validation exactly at the target epoch, falls back to the nearest
  validation <= target and reports epoch_used per model -- check that before trusting a row.
- Ranking metric is MEAN(all ds) of full_VRMSE (equal weight per dataset), NOT the
  training `valid` loss scalar.
"""

import os
import re
import sys
from collections import defaultdict

import wandb

# args: optional epoch positional + optional --median (default mean, preserves old behavior)
STAT = "median" if "--median" in sys.argv else "mean"
WANT_FIELDS = (
    "--fields" in sys.argv
)  # per-field VRMSE breakdown (pressure/velocity/...)
WANT_SPECTRAL = (
    "--spectral" in sys.argv
)  # per-k-bin spectral_error_mse breakdown (toy_full)
_pos = [a for a in sys.argv[1:] if not a.startswith("-")]
TARGET = int(_pos[0]) if _pos else 70
PROJECT = os.environ.get("WANDB_PROJECT", "walrus")

# label -> exact experiment name (between "10_source_2d_" and "-all2d"), OR "id:<runid>" to
# pin one specific run (e.g. "twogrid": "id:1fdrvqpo"). Fill ids via `--list`.
MODELS = {
    "medF": "id:xpzxds7z",
    "spec": "id:ma4qxzj0",
    "medTG": "id:8d2i0qtm",
    "lgF": "id:5r71o1b2",
    "medHR": "id:t2wvzlrm",
    "scaled21m": "medium_walrus_devserver_amp_local_data_sclaed_21m",
    # lgHR (large two-grid highres coarse-global) — name auto-follows the furthest-along
    # run of this name (currently crashed dlpmne5v @ep63; switches to the resume once it passes 63)
    "lgHR": "id:dlpmne5v",
    "lgHR_resume": "id:uqsjmnie",  # resume of the above (currently at ep63)
    # --- new runs (this session) ---
    "fullF": "id:7hmy8rw1",  # fp32 full_walrus (running ~ep11) — auto-follows latest
    "lgF_fp32": "id:c2iwjvox",  # fp32 large_walrus (running ~ep 11) — auto-follows latest
    "med_hetero": "id:wfu0w90v",  # hetero-multipole (running ~ep30) — auto-follows latest
    "med_hetero_nostride": "id:kkz4t6my",  # hetero-nostride (running ~ep30) — auto-follows latest
    "med_spectral_op": "id:n16sljs7",  # spectral op (running ~ep30) — auto-follows latest
    "med_spectral_op_hetero_multipole": "id:snxzmfhg",  # spectral op
}
# one-step valid keys.  full table uses field=="full"; --fields/--spectral use the rest.
VRMSE_RE = re.compile(rf"^valid_(?P<ds>.+?)/(?P<field>[A-Za-z_]+?)_VRMSE_T=all_{STAT}$")
SPEC_RE = re.compile(
    rf"^valid_(?P<ds>.+?)/(?P<field>[A-Za-z_]+?)_spectral_error_mse_per_bin_(?P<bin>\d+)_T=all_{STAT}$"
)


def exp_name(run):
    return run.name.replace("10_source_2d_", "").split("-all2d")[0]


def main():
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    runs = list(api.runs(f"{entity}/{PROJECT}"))

    if "--list" in sys.argv:
        # Discover exact experiment names (the string MODELS must match). Use this to
        # fill/fix the new entries above.
        for r in sorted(runs, key=lambda r: str(r.created_at)):
            print(
                f"{exp_name(r):60s} id={r.id} ep={r.summary.get('epoch')} state={r.state}"
            )
        return

    def pick(spec):
        # spec may be:
        #   "id:<runid>"  -> fetch THAT exact run directly via the API (robust pin; works
        #                    even if the run isn't in the paginated listing, and survives
        #                    name reuse). Same convention as eval_rollout_table.py.
        #   "<runid>"     -> raw id present in the fetched listing (back-compat).
        #   "<exp_name>"  -> exact experiment name (auto-follows the furthest-along run).
        # NOTE: the TARGET-epoch selection below still applies to a pinned run -- the id
        # picks the RUN, then its history is read at TARGET (one-step is epoch-indexed,
        # unlike the rollout table's single epoch-101 snapshot).
        if spec.startswith("id:"):
            rid = spec[3:]
            try:
                return api.run(f"{entity}/{PROJECT}/{rid}")
            except Exception as e:
                print(f"# (id:{rid} fetch failed: {str(e)[:60]})")
                return None
        by_id = [r for r in runs if r.id == spec]
        if by_id:
            return by_id[0]
        cand = [r for r in runs if exp_name(r) == spec]
        cand.sort(key=lambda r: (r.summary.get("epoch") or -1))
        return cand[-1] if cand else None

    result, used = {}, {}
    for lbl, nsub in MODELS.items():
        r = pick(nsub)
        if not r:
            result[lbl], used[lbl] = {}, None
            print(f"# {lbl:9s} NO RUN FOUND for '{nsub}'")
            continue
        # Collect only what's needed: full_VRMSE always; all VRMSE fields if --fields;
        # spectral bins if --spectral (keeps the default scan cheap).
        keys = [
            k
            for k in r.summary.keys()
            if (
                VRMSE_RE.match(k)
                and (WANT_FIELDS or VRMSE_RE.match(k)["field"] == "full")
            )
            or (WANT_SPECTRAL and SPEC_RE.match(k))
        ]
        byep = defaultdict(dict)
        for h in r.scan_history(keys=["epoch"] + keys):
            e = h.get("epoch")
            if e is None:
                continue
            for k in keys:
                if h.get(k) is not None:
                    byep[int(e)][k] = h[k]
        if TARGET in byep and byep[TARGET]:
            e = TARGET
        else:
            below = [x for x in byep if x <= TARGET and byep[x]]
            e = max(below) if below else None
        used[lbl] = e
        # store the raw {key: value} snapshot at the chosen epoch (parsed by the printers)
        result[lbl] = dict(byep.get(e, {})) if e is not None else {}
        print(f"# {lbl:9s} run={exp_name(r):46s} id={r.id} epoch_used={e}")

    labels = list(MODELS)

    def vr(lbl, ds, field):
        return result[lbl].get(f"valid_{ds}/{field}_VRMSE_T=all_{STAT}")

    def sp(lbl, ds, b):
        return result[lbl].get(
            f"valid_{ds}/full_spectral_error_mse_per_bin_{b}_T=all_{STAT}"
        )

    # datasets present, from the full_VRMSE keys across all models
    dsets = sorted(
        {
            VRMSE_RE.match(k)["ds"]
            for snap in result.values()
            for k in snap
            if VRMSE_RE.match(k) and VRMSE_RE.match(k)["field"] == "full"
        }
    )

    # ---- default: full_VRMSE_T=all + MEAN(all ds) row ----
    print(f"\nfull_VRMSE_T=all_{STAT} @ epoch {TARGET} (lower is better)\n")
    print("dataset".ljust(34) + "".join(l.rjust(11) for l in labels))
    for d in dsets:
        print(
            d.ljust(34)
            + "".join(
                (
                    f"{vr(l, d, 'full'):11.4f}"
                    if vr(l, d, "full") is not None
                    else " " * 11
                )
                for l in labels
            )
        )

    def mean_full(lbl):
        vals = [vr(lbl, d, "full") for d in dsets]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    print(
        "MEAN(all ds)".ljust(34)
        + "".join(
            (f"{mean_full(l):11.4f}" if mean_full(l) is not None else " " * 11)
            for l in labels
        )
    )

    # ---- --fields: per-field VRMSE_T=all per dataset ----
    if WANT_FIELDS:
        print(f"\n## per-field VRMSE_T=all_{STAT} @ epoch {TARGET}\n")
        for d in dsets:
            fields = sorted(
                {
                    VRMSE_RE.match(k)["field"]
                    for snap in result.values()
                    for k in snap
                    if VRMSE_RE.match(k) and VRMSE_RE.match(k)["ds"] == d
                }
            )
            if not fields:
                continue
            print(f"### {d}")
            print("field".ljust(16) + "".join(l.rjust(11) for l in labels))
            for f in fields:
                print(
                    f.ljust(16)
                    + "".join(
                        (
                            f"{vr(l, d, f):11.4f}"
                            if vr(l, d, f) is not None
                            else " " * 11
                        )
                        for l in labels
                    )
                )
            print()

    # ---- --spectral: per-k-bin full spectral_error_mse_T=all per dataset (toy_full only) ----
    if WANT_SPECTRAL:
        print(
            f"\n## per-bin full_spectral_error_mse_T=all_{STAT} @ epoch {TARGET} (low->high k)\n"
        )
        for d in dsets:
            bins = sorted(
                {
                    int(SPEC_RE.match(k)["bin"])
                    for snap in result.values()
                    for k in snap
                    if SPEC_RE.match(k)
                    and SPEC_RE.match(k)["ds"] == d
                    and SPEC_RE.match(k)["field"] == "full"
                }
            )
            if not bins:
                continue
            print(f"### {d}")
            print("k-bin".ljust(10) + "".join(l.rjust(13) for l in labels))
            for b in bins:
                print(
                    f"bin_{b}".ljust(10)
                    + "".join(
                        (
                            f"{sp(l, d, b):13.6f}"
                            if sp(l, d, b) is not None
                            else " " * 13
                        )
                        for l in labels
                    )
                )
            print()


if __name__ == "__main__":
    main()
