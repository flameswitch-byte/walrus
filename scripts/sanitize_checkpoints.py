#!/usr/bin/env python
"""Sanitize a checkpoint tree before/after migrating to a new machine.

Why this exists
---------------
Older runs (and bulk rsync/cp syncs) leave ``last`` / ``best`` as *symlinks*
(e.g. ``last -> step_100``). When the tree is then accessed through an sshfs
mount with ``follow_symlinks``, the client cannot tell the symlink from a real
directory (``os.path.islink`` returns False). The training saver deletes
``last`` every epoch with ``shutil.rmtree`` -- which then follows the hidden
symlink and DELETES the ``step_*`` checkpoint it points at. That is the
corruption this script removes ahead of time.

RUN THIS ON THE HOST THAT OWNS THE REAL FILESYSTEM (e.g. the devbig host), NOT
through an sshfs mount. On the owning host symlinks are visible, so the script
can convert each ``last``/``best`` symlink into a real copy of its target
(default) or just drop the link (``--delete``).

Safety properties (by construction):
  * It only ever acts on entries for which ``Path.is_symlink()`` is True, and
    only on the names ``last``/``best``.
  * On a symlink it calls ``unlink()`` (removes the LINK, never the target) and
    then copies the target's contents into a fresh real directory.
  * It NEVER calls ``shutil.rmtree`` on a directory, and it NEVER creates a
    symlink -- so it cannot delete a checkpoint folder or (re)introduce a
    ``last`` symlink.
  * On a symlink-following mount, ``is_symlink()`` is False for the hidden
    links, so the script simply finds nothing and is a harmless no-op (it warns
    you to run it on the owning host instead).
  * It does not descend into ``step_*``/``last``/``best`` directories, so it is
    fast even over a network filesystem.

Usage
-----
    # on the host that owns the files (NOT via sshfs):
    python sanitize_checkpoints.py /data/users/<you>/neural_pde/checkpoints
    python sanitize_checkpoints.py <root> --delete     # drop links instead of copying
    python sanitize_checkpoints.py <root> --dry-run    # report only
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

LINK_NAMES = ("last", "best")


def iter_checkpoint_dirs(root: pathlib.Path):
    """Yield dirs that contain last/best/step_* without descending into them."""
    for dirpath, dirnames, _ in os.walk(root):
        here = pathlib.Path(dirpath)
        names = set(dirnames)
        if any(n in names for n in LINK_NAMES) or any(
            n.startswith("step_") for n in names
        ):
            yield here
            # Do not recurse into checkpoint payload dirs (huge over sshfs).
            dirnames[:] = [
                d
                for d in dirnames
                if d not in LINK_NAMES and not d.startswith("step_")
            ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=pathlib.Path, help="checkpoints root to scan")
    ap.add_argument(
        "--delete",
        action="store_true",
        help="remove last/best symlinks instead of converting them to real copies",
    )
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    converted = deleted = broken = found = 0
    for ckpt_dir in iter_checkpoint_dirs(root):
        for name in LINK_NAMES:
            p = ckpt_dir / name
            # SAFETY: act only on real, visible symlinks. A real directory (or a
            # symlink hidden by a follow_symlinks mount) is left untouched.
            if not p.is_symlink():
                continue
            found += 1
            link_target = os.readlink(p)
            resolved = (ckpt_dir / link_target).resolve()
            if not resolved.exists():
                print(f"[broken]  {p} -> {link_target} (target missing; removing link)")
                broken += 1
                if not args.dry_run:
                    p.unlink()  # removes the dangling LINK only
                continue
            if args.delete:
                print(f"[delete]  {p} -> {link_target}")
                deleted += 1
                if not args.dry_run:
                    p.unlink()  # removes the LINK only, never the target
            else:
                print(f"[convert] {p} -> real copy of {resolved.name}")
                converted += 1
                if not args.dry_run:
                    p.unlink()  # remove LINK first so copytree writes a fresh dir
                    shutil.copytree(resolved, p)

    if found == 0:
        print(
            "No last/best symlinks found. If you expected some, you are probably "
            "running through a symlink-following mount that hides them -- run this "
            "on the host that owns the filesystem instead."
        )
    print(
        f"\nDone{' (dry-run)' if args.dry_run else ''}: "
        f"converted={converted} deleted={deleted} broken_removed={broken}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
