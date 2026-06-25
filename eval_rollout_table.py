#!/usr/bin/env python
"""Per-dataset ROLLOUT metrics from wandb, across the two-grid-vs-flat comparison runs.

Companion to eval_vrmse_table.py (one-step `valid` means). Pulls the *rollout_test* metrics
behind §5.2 / §2.2 of knowledge_base/walrus_twogrid_vs_flat_results.md.

The agent identity is blocked from api.wandb.ai by the fwdproxy filter, so RUN THIS YOURSELF:
    with-proxy python eval_rollout_table.py                 # latest rollout per run, T=all + growth
    with-proxy python eval_rollout_table.py 100             # rollout at/<=epoch 100
    with-proxy python eval_rollout_table.py --fields        # + per-field T=all (pressure etc.)
    with-proxy python eval_rollout_table.py --buckets       # + per-bucket full_VRMSE trend (rollout curve)
    with-proxy python eval_rollout_table.py --spectral      # + per-bin spectral_error_mse (toy_full only)
    with-proxy python eval_rollout_table.py --buckets --spectral --fields   # everything

PINNING A SPECIFIC RUN BY ID (the fix for the epoch-101 collision):
  A standalone eval logs at epoch = max_epoch+1 (101), regardless of which checkpoint it used,
  so an ep40 eval and an ep100 eval of the SAME name both land at epoch 101 and the name-match
  picks one arbitrarily. To pin the exact run, put `id:<runid>` in MODELS (see below) -- it
  fetches that run directly and ignores the epoch TARGET. Example already wired:
  `medF@40` -> id:79105vr9, `hetero@40` -> id:kmqdzw23.

Notes:
- Name entries match the EXACT experiment name (between "10_source_2d_" and "-all2d") and take
  the LATEST epoch (or nearest <= TARGET). `id:` entries pin one run, epoch-agnostic.
- MEDIAN is the metric of record (mean is wrecked by diverged systems). Only T=0:1 and T=all are
  guaranteed cross-comparable across runs; other bucket edges depend on num_time_intervals, so
  --buckets is a within-eval trend, compare buckets across runs only when their edges match.
- Blank cell = that run/field has no rollout on wandb.
"""

import os
import re
import sys
from collections import defaultdict

import wandb

PROJECT = os.environ.get("WANDB_PROJECT", "walrus")

# --- args -----------------------------------------------------------------
WANT_FIELDS = "--fields" in sys.argv
WANT_BUCKETS = "--buckets" in sys.argv
WANT_SPECTRAL = "--spectral" in sys.argv
pos = [a for a in sys.argv[1:] if not a.startswith("-")]
TARGET = int(pos[0]) if pos else None  # None => latest rollout epoch per run

# label -> EITHER an exact experiment name (name-match, latest/<=TARGET epoch)
#          OR "id:<wandb_run_id>" to pin one specific run (epoch-agnostic; the epoch-101 fix).
MODELS = {
    # "medF@100": "medium_walrus_devserver_amp_gpu_oom",
    # "medTG": "medium_walrus_devserver_amp_gpu_twogrid_oom",
    # "medHR@100": "twogrid_gpu_oom_highres",
    # "sc21mSp@100": "medium_walrus_scaled_devserver_amp_gpu_spectral_002_floor",
    # "spec@100": "medium_walrus_devserver_amp_gpu_spectral_002_floor",
    # "hetero@100": "medium_walrus_devserver_hetero",
    # "nostride@100": "medium_walrus_devserver_hetero_nostride",
    # --- pinned-by-id custom pulls (epoch-101 collision fix); comment out if not wanted ---
    # "medF@40": "id:79105vr9",  # medF rollout from the step_40 eval
    # "hetero@40": "id:kmqdzw23",  # hetero rollout from the step_40 eval
    "medF@100": "id:vqthxpeo",  # medF rollout from the step_100 eval
    "medTG@100": "id:hvmmz5j1",  # medTG rollout from the step_100 eval
    "lgF@100": "id:pvmbwsst",
    "lgHR@100": "id:kx1ejx35",  # lgHR rollout from the step_100 eval
    "medHR@100": "id:m8ljuj01",  # medHR rollout from the step_100 eval
    "hetero@100": "id:d9m7h84b",  # hetero rollout from the step_100 eval
    "nostride@100": "id:5wlyvv8p",  # nostride rollout from the step
    "specop@100": "id:as74nl73",  # specop rollout from the step_100 eval
    "hetero_specop_multipole@100": "id:52picxhn",  # hetero_specop_multipole rollout from the step
    "sc21m@100": "id:97eiqxc0",  # sc21m rollout from the step_100 eval
    "medF64@100": "id:6ozutb2p",  # flat FullAttention @ 64 tok/axis (resolution-on-flat cut, §14.8)
    "medFMM_fused@100": "id:byoaoj0b",
}

# canonical dataset order (short label -> wandb dataset name embedded in the metric key)
DATASETS = {
    "helmholtz": "helmholtz_staircase",
    "planetswe": "planetswe",
    "acoustic_disc": "acoustic_scattering_discontinuous",
    "acoustic_incl": "acoustic_scattering_inclusions",
    "rayleigh": "rayleigh_benard",
    "shear": "shear_flow",
    "viscoelastic": "viscoelastic_instability",
    "turbulent": "turbulent_radiative_layer_2D",
    "active_matter": "active_matter",
    "gray_scott": "gray_scott_reaction_diffusion",
}

# rollout_test_<ds>/<field>_VRMSE_T=<bucket>_median   (field == "full" for the headline number)
VRMSE_RE = re.compile(
    r"^rollout_test_(?P<ds>.+?)/(?P<field>[A-Za-z_]+?)_VRMSE_T=(?P<bucket>[^_]+)_median$"
)
# rollout_test_<ds>/<field>_spectral_error_mse_per_bin_<bin>_T=<bucket>_median
SPEC_RE = re.compile(
    r"^rollout_test_(?P<ds>.+?)/(?P<field>[A-Za-z_]+?)_spectral_error_mse_per_bin_(?P<bin>\d+)_T=(?P<bucket>[^_]+)_median$"
)


def exp_name(run):
    return run.name.replace("10_source_2d_", "").split("-all2d")[0]


def bucket_start(b):
    """Sort key for a bucket string like '0:1', '21:59', 'all'. 'all' sorts last."""
    if b == "all":
        return (1, 0)
    try:
        return (0, int(b.split(":")[0]))
    except Exception:
        return (0, 9999)


def collect(run):
    """Return (byep, summary, sum_ep) of ALL rollout_test *_median keys (vrmse + spectral)."""
    keys = [
        k
        for k in run.summary.keys()
        if k.startswith("rollout_test_") and k.endswith("_median")
    ]
    summary = {}
    for k in keys:
        v = run.summary.get(k)
        if v is not None:
            try:
                summary[k] = float(v)
            except (TypeError, ValueError):
                pass
    sum_ep = run.summary.get("epoch")
    sum_ep = int(sum_ep) if sum_ep is not None else None
    byep = defaultdict(dict)
    if keys:
        for h in run.scan_history(keys=["epoch"] + keys):
            e = h.get("epoch")
            if e is None:
                continue
            for k in keys:
                if h.get(k) is not None:
                    byep[int(e)][k] = h[k]
    return byep, summary, sum_ep


def resolve(byep, summary, sum_ep, pinned):
    """(snapshot {rawkey:val}, epoch_used). Pinned-by-id => always the run's summary."""
    if pinned or TARGET is None:
        if summary:
            return summary, sum_ep
        if byep:
            e = max(byep)
            return byep[e], e
        return {}, None
    if TARGET in byep:
        return byep[TARGET], TARGET
    below = [e for e in byep if e <= TARGET]
    if below:
        e = max(below)
        return byep[e], e
    return (summary, sum_ep) if summary else ({}, None)


def vrmse(snap, ds, field, bucket):
    return snap.get(f"rollout_test_{ds}/{field}_VRMSE_T={bucket}_median")


def main():
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    runs = list(api.runs(f"{entity}/{PROJECT}"))
    by_name = defaultdict(list)
    for r in runs:
        by_name[exp_name(r)].append(r)

    def fetch(spec):
        if spec.startswith("id:"):
            rid = spec[3:]
            try:
                return api.run(f"{entity}/{PROJECT}/{rid}"), True
            except Exception as e:
                print(f"# (id:{rid} fetch failed: {str(e)[:60]})")
                return None, True
        cand = sorted(
            by_name.get(spec, []), key=lambda r: (r.summary.get("epoch") or -1)
        )
        return (cand[-1] if cand else None), False

    def full_buckets(snap):
        """The T= bucket edges present for the `full` field (the rollout-time dimension)."""
        return tuple(
            sorted(
                {
                    VRMSE_RE.match(k)["bucket"]
                    for k in snap
                    if VRMSE_RE.match(k) and VRMSE_RE.match(k)["field"] == "full"
                },
                key=bucket_start,
            )
        )

    data, used, bkts = {}, {}, {}
    for lbl, spec in MODELS.items():
        r, pinned = fetch(spec)
        if not r:
            data[lbl], used[lbl], bkts[lbl] = {}, None, ()
            print(f"# {lbl:9s} NO RUN FOUND for '{spec}'")
            continue
        byep, summary, sum_ep = collect(r)
        snap, e = resolve(byep, summary, sum_ep, pinned)
        data[lbl], used[lbl] = snap, e
        bkts[lbl] = full_buckets(snap)
        tag = "" if snap else "  (no rollout on wandb)"
        pin = " [pinned id]" if pinned else ""
        bstr = "[" + ",".join(bkts[lbl]) + "]" if bkts[lbl] else "[-]"
        print(
            f"# {lbl:9s} run={exp_name(r):46s} id={r.id} epoch_used={e}{pin}  "
            f"buckets={bstr}{tag}"
        )

    labels = list(MODELS)

    # ---- comparability check on the bucket (rollout-time) dimension ----
    distinct = {b for b in bkts.values() if b}
    print()
    if len(distinct) <= 1:
        only = next(iter(distinct), ())
        print(
            f"# buckets: ALL runs share edges {list(only)} -> every bucket is cross-comparable."
        )
    else:
        print(
            "# ⚠ buckets DIFFER across runs (different num_time_intervals/max_rollout_steps)."
        )
        for b in sorted(distinct, key=lambda t: (len(t), t)):
            who = [l for l in labels if bkts[l] == b]
            print(f"#    {list(b)}  <- {', '.join(who)}")
        common = set.intersection(*[set(b) for b in distinct]) if distinct else set()
        print(
            f"#    cross-comparable across ALL: {sorted(common, key=bucket_start)} "
            f"(only these buckets are safe to compare run-to-run)."
        )

    # ---- headline: full_VRMSE T=all median / growth (T=all / T=0:1) ----
    print(
        f"\n## rollout full_VRMSE_T=all median  /  growth(=T=all/T=0:1)   "
        f"[epoch: {'latest' if TARGET is None else TARGET}, id-pins epoch-agnostic]\n"
    )
    print("dataset".ljust(15) + "".join(l.rjust(16) for l in labels))
    for dl, dn in DATASETS.items():
        row = dl.ljust(15)
        for l in labels:
            allv = vrmse(data[l], dn, "full", "all")
            onev = vrmse(data[l], dn, "full", "0:1")
            if allv is None:
                row += "".rjust(16)
            elif onev:
                row += f"{allv:.3f}/{allv / onev:.0f}x".rjust(16)
            else:
                row += f"{allv:.3f}/?".rjust(16)
        print(row)

    # ---- --buckets: per-bucket full_VRMSE trend (the rollout curve) ----
    if WANT_BUCKETS:
        print(
            "\n## per-bucket full_VRMSE_T=<bucket> median (rollout trend; "
            "compare buckets ACROSS runs only when edges match)\n"
        )
        for dl, dn in DATASETS.items():
            buckets = sorted(
                {
                    VRMSE_RE.match(k)["bucket"]
                    for l in labels
                    for k in data[l]
                    if VRMSE_RE.match(k)
                    and VRMSE_RE.match(k)["ds"] == dn
                    and VRMSE_RE.match(k)["field"] == "full"
                },
                key=bucket_start,
            )
            if not buckets:
                continue
            print(f"### {dl} ({dn})")
            print("T=bucket".ljust(12) + "".join(l.rjust(13) for l in labels))
            for b in buckets:
                row = f"{b}".ljust(12)
                for l in labels:
                    v = vrmse(data[l], dn, "full", b)
                    row += f"{v:13.4f}" if v is not None else "".rjust(13)
                print(row)
            print()

    # ---- --fields: per-field full_VRMSE T=all median ----
    if WANT_FIELDS:
        print("\n## per-field rollout VRMSE_T=all median (for §2.2 deltas)\n")
        for dl, dn in DATASETS.items():
            fields = sorted(
                {
                    VRMSE_RE.match(k)["field"]
                    for l in labels
                    for k in data[l]
                    if VRMSE_RE.match(k)
                    and VRMSE_RE.match(k)["ds"] == dn
                    and VRMSE_RE.match(k)["bucket"] == "all"
                }
            )
            if not fields:
                continue
            print(f"### {dl} ({dn})")
            print("field".ljust(14) + "".join(l.rjust(12) for l in labels))
            for f in fields:
                row = f.ljust(14)
                for l in labels:
                    v = vrmse(data[l], dn, f, "all")
                    row += f"{v:12.4f}" if v is not None else "".rjust(12)
                print(row)
            print()

    # ---- --spectral: per-bin full spectral_error_mse T=all median (toy_full) ----
    if WANT_SPECTRAL:
        print(
            "\n## per-bin full_spectral_error_mse_T=all median (low->high k; toy_full only)\n"
        )
        for dl, dn in DATASETS.items():
            bins = sorted(
                {
                    int(SPEC_RE.match(k)["bin"])
                    for l in labels
                    for k in data[l]
                    if SPEC_RE.match(k)
                    and SPEC_RE.match(k)["ds"] == dn
                    and SPEC_RE.match(k)["field"] == "full"
                    and SPEC_RE.match(k)["bucket"] == "all"
                }
            )
            if not bins:
                continue
            print(f"### {dl} ({dn})")
            print("k-bin".ljust(10) + "".join(l.rjust(14) for l in labels))
            for b in bins:
                row = f"bin_{b}".ljust(10)
                for l in labels:
                    v = data[l].get(
                        f"rollout_test_{dn}/full_spectral_error_mse_per_bin_{b}_T=all_median"
                    )
                    row += f"{float(v):14.6f}" if v is not None else "".rjust(14)
                print(row)
            print()


if __name__ == "__main__":
    main()
