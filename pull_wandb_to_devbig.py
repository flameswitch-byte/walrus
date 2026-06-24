#!/usr/bin/env python
"""Pull all wandb runs (history + summary + config + files) from the cloud to local disk,
for backup to devbig. Run from an authed terminal:  with-proxy python pull_wandb_to_devbig.py
Then rsync the OUT dir to devbig (see the printed hint at the end).

Env overrides: WB_ENTITY, WB_PROJECT, WB_OUT, WB_REPULL_RUNNING(=1 to re-pull running runs).
Resumable: a run with a complete marker is skipped unless it's still running.
"""
import json
import os
import sys

import wandb

ENTITY = os.environ.get("WB_ENTITY", "amesduong313-self")
PROJECT = os.environ.get("WB_PROJECT", "walrus")
OUT = os.environ.get("WB_OUT", "/data/repos/neural_pde/walrus/wandb_cloud_backup")
REPULL_RUNNING = os.environ.get("WB_REPULL_RUNNING", "1") == "1"

os.makedirs(OUT, exist_ok=True)
api = wandb.Api(timeout=120)
runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
print(f"Found {len(runs)} runs in {ENTITY}/{PROJECT}  ->  {OUT}")


def safe(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))[:140]


done = skipped = errored = 0
for i, run in enumerate(runs, 1):
    d = os.path.join(OUT, f"{run.id}__{safe(run.name)}")
    marker = os.path.join(d, ".complete")
    # skip already-pulled finished runs (re-pull only if still running)
    if os.path.exists(marker) and not (REPULL_RUNNING and run.state == "running"):
        skipped += 1
        continue
    try:
        os.makedirs(os.path.join(d, "files"), exist_ok=True)
        # 1) metadata (id/name/state/config/summary)
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(
                {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "url": run.url,
                    "created_at": str(run.created_at),
                    "config": dict(run.config),
                    "summary": {k: v for k, v in dict(run.summary).items()},
                },
                fh,
                default=str,
                indent=2,
            )
        # 2) full metric history -> JSONL (one logged step per line, all keys)
        n = 0
        with open(os.path.join(d, "history.jsonl"), "w") as fh:
            for row in run.scan_history():
                fh.write(json.dumps(row, default=str) + "\n")
                n += 1
        # 3) logged files (config.yaml, output.log, wandb-metadata.json, media, ...)
        for f in run.files():
            try:
                f.download(root=os.path.join(d, "files"), replace=True)
            except Exception as e:  # noqa: BLE001
                print(f"    [file] {f.name}: {e}")
        open(marker, "w").close()
        done += 1
        print(f"[{i}/{len(runs)}] {run.id} {run.name[:60]}  rows={n} state={run.state}")
    except Exception as e:  # noqa: BLE001
        errored += 1
        print(f"[{i}/{len(runs)}] ERROR {run.id} {run.name[:60]}: {e}")

print(f"\nDone. pulled={done} skipped={skipped} errored={errored}  in {OUT}")
print("Now back it up to devbig (from this authed terminal):")
print(
    f"  with-proxy rsync -ah --info=progress2 {OUT}/ "
    f"devvm20873.ldc0.facebook.com:/data/users/jarvislam1999/neural_pde/wandb_cloud/"
)
