"""Directly inspired from https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import warnings
from concurrent.futures import Future
from typing import Callable, Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)

CHECKPOINT_METADATA_FILENAME = "metadata.pt"


def delete_folder_or_symlink(folder: pathlib.Path):
    """Delete a folder, or unlink a symlink to a folder, without following links.

    ``lexists`` catches broken/dangling symlinks; ``ignore_errors`` on the
    recursive delete tolerates half-written/cache-inconsistent dirs on network
    filesystems.

    WARNING (sshfs + ``follow_symlinks``): if the mount resolves symlinks
    server-side, ``os.path.islink`` returns False for a real symlink here, so a
    stale ``last``/``best`` symlink is indistinguishable from a directory and the
    ``rmtree`` branch will follow it and delete its *target* checkpoint. There is
    no reliable client-side guard for that case (``realpath``/``rename`` are
    equally fooled -- verified empirically). Safeguard operationally instead:
    keep the checkpoint tree free of symlinks (the save path below only ever
    writes real copies, never symlinks) and do NOT mount with ``follow_symlinks``.
    """
    if os.path.islink(folder):
        folder.unlink()
    elif os.path.lexists(folder):
        shutil.rmtree(folder, ignore_errors=True)


def checkpoint_already_exists(checkpoint_dirname: pathlib.Path) -> bool:
    """Check if a given checkpoint already exists."""
    return checkpoint_dirname.exists()


def link_checkpoint(src_checkpoint: pathlib.Path, target_checkpoint: pathlib.Path):
    """Materialize `target_checkpoint` as a full copy of `src_checkpoint`.

    Despite the name, this copies rather than symlinks. Symlinks are unreliable on
    some network filesystems (e.g. sshfs), where ``symlink_to`` can silently leave a
    broken empty directory behind and crash the next save. Copying is filesystem
    agnostic at the cost of extra space for the `last`/`best` checkpoints.
    To be used typically for saving last checkpoint that refers to an already existing one.
    """
    logger.info(f"Copy checkpoint {target_checkpoint} from {src_checkpoint}")
    # Robustly clear any prior target. delete_folder_or_symlink never recurses
    # through a symlink (even one sshfs hides from islink()), so it cannot delete
    # the source checkpoint a stale `last`/`best` link points at.
    delete_folder_or_symlink(target_checkpoint)
    # dirs_exist_ok=True so a leftover dir (e.g. one rmtree could not remove on a
    # network FS) does not crash the copy with FileExistsError.
    shutil.copytree(src_checkpoint, target_checkpoint, dirs_exist_ok=True)


def save_metadata(
    checkpoint_dir: pathlib.Path,
    epoch: Optional[int] = None,
    val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
):
    """Checkpoint information that do not require synchronization across GPUs,
    or which are already synchronized.
    To be used in combination of FSDP checkpointing strategy.

    """
    state_dict = {"epoch": epoch, "val_loss": val_loss, "best_val_loss": best_val_loss}
    torch.save(state_dict, checkpoint_dir / CHECKPOINT_METADATA_FILENAME)


def on_future(fn: Callable, *args, **kwargs) -> Callable:
    """Wrap any callable to be run as future callback."""

    def future_wrapper(future: Future):
        return fn(*args, **kwargs)

    return future_wrapper


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(
        self,
        model,
        optimizer=None,
    ):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        if self.optimizer is None:
            model_state_dict, _ = get_state_dict(self.model, [])
            return {
                "model": model_state_dict,
            }
        else:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        if self.optimizer is None:
            set_state_dict(
                model=self.model,
                optimizers=[],
                model_state_dict=state_dict["model"],
                optim_state_dict=None,
                options=StateDictOptions(strict=False),
            )
        else:
            set_state_dict(
                model=self.model,
                optimizers=self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
                options=StateDictOptions(strict=False),
            )


class CheckPointLoader:
    """Base class for checkpointing.
    It only load checkpoints.

    """

    def __init__(
        self,
        save_dir: pathlib.Path | str,
        rank=0,
        load_checkpoint_path: Optional[pathlib.Path | str] = None,
        coalesced_checkpoint_path: Optional[pathlib.Path | str] = None,
        prioritize_resume: bool = True,
    ) -> None:
        if load_checkpoint_path is not None and coalesced_checkpoint_path is not None:
            raise ValueError(
                "Cannot specify both load_checkpoint_path and coalesced_checkpoint_path"
            )
        self.save_dir = pathlib.Path(save_dir).resolve()
        if prioritize_resume and self.last_checkpoint is not None:
            self.load_checkpoint_path = self.last_checkpoint
            if load_checkpoint_path is not None:
                logger.info(
                    "Finetuning path provided, but resume enabled. Default behavior"
                    "is to resume from last checkpoint. To restart from finetuning path, set"
                    " auto_resume=False or config.prioritize_resume=False"
                )
        else:
            self.load_checkpoint_path = (
                pathlib.Path(load_checkpoint_path).resolve()
                if load_checkpoint_path is not None
                else self.last_checkpoint
            )
        logger.info(f"Source checkpoint path set to: {self.load_checkpoint_path}")
        self.rank = rank
        self._best_metrics: Optional[float] = None

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict=False,  # Unused parameters can get annoying when true on
        local=False,
    ) -> Tuple[int | None, float | None]:
        """Load the model and optimizer state from a checkpoint.

        Args:
            model (torch.nn.Module): The model to load the state into.
            optimizer (torch.optim.Optimizer): The optimizer to load the state into.

        Returns:
            Tuple[int | None, float | None]: A tuple containing the epoch number and
                                              validation loss from the checkpoint.
        """
        state_dict = {"app": AppState(model, optimizer)}
        # Default values if no checkpoint is loaded
        epoch = None
        val_loss = None
        if self.load_checkpoint_path is not None and self.load_checkpoint_path.exists():
            metadata_file = self.load_checkpoint_path / CHECKPOINT_METADATA_FILENAME
            assert metadata_file.exists(), (
                f"Metadata file {metadata_file} does not exist"
            )
            try:
                if local:
                    try:
                        state_dict = torch.load(
                            os.path.join(
                                self.load_checkpoint_path, "full_checkpoint.pt"
                            ),
                            map_location=(
                                "cuda" if torch.cuda.is_available() else "cpu"
                            ),
                            weights_only=False,
                        )
                        model.load_state_dict(state_dict["app"]["model"])
                        if optimizer is not None:
                            optimizer.load_state_dict(state_dict["app"]["optimizer"])
                    except Exception as e:
                        logger.warning(
                            f"Failed to load local checkpoint from {self.load_checkpoint_path}: {e}"
                            ". Trying DCP loading as fallback instead."
                        )
                        if strict:
                            dcp.load(
                                state_dict=state_dict,
                                checkpoint_id=self.load_checkpoint_path,
                            )
                        else:
                            dcp.load(
                                state_dict=state_dict,
                                checkpoint_id=self.load_checkpoint_path,
                                planner=dcp.default_planner.DefaultLoadPlanner(
                                    allow_partial_load=True
                                ),
                            )
                        logger.info(
                            f"Successfully loaded checkpoint {self.load_checkpoint_path} using DCP fallback."
                        )
                else:
                    # Load the state dictionary from the checkpoint
                    if strict:
                        dcp.load(
                            state_dict=state_dict,
                            checkpoint_id=self.load_checkpoint_path,
                        )
                    else:
                        dcp.load(
                            state_dict=state_dict,
                            checkpoint_id=self.load_checkpoint_path,
                            planner=dcp.default_planner.DefaultLoadPlanner(
                                allow_partial_load=True
                            ),
                        )
                    # Load the metadata (epoch and validation loss)
                checkpoint_metadata = torch.load(metadata_file, weights_only=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint from {self.load_checkpoint_path}: {e}"
                )
            else:
                epoch = checkpoint_metadata.get("epoch", None)
                val_loss = checkpoint_metadata.get("val_loss", None)
                self._best_metrics = checkpoint_metadata.get("best_val_loss", None)
        return epoch, val_loss

    def save_if_necessary(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        val_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        force: bool = False,
        local: bool = False,
    ):
        pass

    @staticmethod
    def _is_usable_checkpoint(path: pathlib.Path) -> bool:
        """A checkpoint is usable only if it has its metadata sidecar."""
        return path.exists() and (path / CHECKPOINT_METADATA_FILENAME).exists()

    @property
    def last_checkpoint(self) -> pathlib.Path | None:
        """Return the checkpoint to resume from, resiliently.

        Prefers ``last``, but only when it is actually usable (has metadata.pt).
        Otherwise falls back to the highest-numbered ``step_N`` that has its
        metadata. ``step_*`` directories are always real directories written
        directly by the saver, so -- unlike ``last``/``best`` -- they are immune
        to the stale-symlink corruption that an sshfs ``follow_symlinks`` mount
        can introduce. This makes resume self-healing across machine migrations:
        a broken/empty/symlinked ``last`` no longer aborts the run.
        """
        save_dir = pathlib.Path(self.save_dir)
        last_checkpoint_dir = save_dir / "last"
        if self._is_usable_checkpoint(last_checkpoint_dir):
            return last_checkpoint_dir.resolve()

        steps = []
        if save_dir.exists():
            for p in save_dir.iterdir():
                suffix = p.name[len("step_") :]
                if (
                    p.name.startswith("step_")
                    and suffix.isdigit()
                    and self._is_usable_checkpoint(p)
                ):
                    steps.append((int(suffix), p))
        if steps:
            best = max(steps, key=lambda t: t[0])[1]
            warnings.warn(
                f"`last` checkpoint missing/unusable; resuming from highest valid "
                f"step checkpoint instead: {best}"
            )
            return best.resolve()

        warnings.warn("No usable checkpoint found")
        return None


class CheckPointer(CheckPointLoader):
    """Class to checkpoint training state_dict under FSDP strategy."""

    def __init__(
        self,
        save_dir: pathlib.Path,
        save_best: bool = True,
        checkpoint_frequency: int = 0,
        rank: int = 0,
        load_checkpoint_path: Optional[pathlib.Path] = None,
        coalesced_checkpoint_path: Optional[pathlib.Path] = None,
        prioritize_resume: bool = True,
        **kwargs,
    ):
        super().__init__(
            save_dir,
            rank,
            load_checkpoint_path,
            coalesced_checkpoint_path,
            prioritize_resume,
        )
        logger.info(f"Checkpointing to {save_dir}")
        if rank == 0:
            self.save_dir.mkdir(exist_ok=True)
        self.save_best = save_best
        self.checkpoint_frequency = checkpoint_frequency

    def save_if_necessary(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        val_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        force: bool = False,
        local: bool = False,
    ) -> Optional[Future]:
        """Check if checkpoints must be saved.
        If so triggers the asynchronous writing of the file.
        Upon writing link additional checkpoints to the written file.
        Returns an optional future.

        Args:
            model: Model whose state is to save
        optimizer: Optimizer state to save
        val_loss: Loss saved as checkpoint metadata and used to save best model
        epoch: Epoch saved as checkpoint metadata and used to save the model every n epochs
        force: Boolean flag to force saving the model, preempts saving every n epoch behavior
        """
        state_dict = {"app": AppState(model, optimizer)}
        checkpoint_future = None
        checkpoint_dirnames = []
        # Save checkpoint based on epoch
        # Those checkpoints as they are absolute (comparatively to best) should be the ones saved
        # Other checkpoints can refer to these ones and thus be linked to avoid hard copies
        save_this_epoch = False
        epoch_checkpoint_dirname = self.save_dir / f"step_{epoch}"
        # Force saving checkpoint which does not already exists
        # Typically occurs when saving checkpoint at the end of training
        if checkpoint_already_exists(epoch_checkpoint_dirname):
            save_this_epoch = False
        elif force:
            save_this_epoch = True
        # Epoch number triggers checkpointing
        elif (
            epoch is not None
            and self.checkpoint_frequency
            and (epoch % self.checkpoint_frequency) == 0
        ):
            save_this_epoch = True
        if save_this_epoch:
            checkpoint_dirnames.append(epoch_checkpoint_dirname)

        # Best metrics triggers checkpointing
        if self.save_best:
            assert val_loss is not None, (
                "Expect to save best metrics but no metrics provided."
            )
            if self._best_metrics is None or val_loss < self._best_metrics:
                self._best_metrics = val_loss
                checkpoint_dirnames.append(self.save_dir / "best")

        # Several files should be saved for the same checkpoints
        checkpoint_dirnames.append(
            self.save_dir / "last"
        )  # This is supposed to write every "epoch"
        if checkpoint_dirnames:
            actual_checkpoint_dirname = checkpoint_dirnames[0]
            # Something weird going on with "last" checkpoint in torch 2.5
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                if actual_checkpoint_dirname.exists() and self.rank == 0:
                    delete_folder_or_symlink(actual_checkpoint_dirname)
                torch.distributed.barrier()
            else:
                if actual_checkpoint_dirname.exists():
                    delete_folder_or_symlink(actual_checkpoint_dirname)

            # Only save the first checkpoints
            logger.info(f"Save checkpoint {actual_checkpoint_dirname}")
            # If local or ddp, just save a checkpoint normally
            if local:
                if self.rank == 0:
                    os.makedirs(actual_checkpoint_dirname, exist_ok=True)
                    torch.save(
                        {
                            "app": {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            }
                        },
                        os.path.join(actual_checkpoint_dirname, "full_checkpoint.pt"),
                    )
            else:
                checkpoint_future = dcp.async_save(
                    state_dict, checkpoint_id=actual_checkpoint_dirname
                )
            # Save already synchronized data on rank 0 only
            # To be used typically for saving epoch and loss
            if self.rank == 0:
                # Only async if this is local
                if not local:
                    checkpoint_future.add_done_callback(
                        on_future(
                            save_metadata,
                            checkpoint_dir=actual_checkpoint_dirname,
                            epoch=epoch,
                            val_loss=val_loss,
                            best_val_loss=self._best_metrics,
                        )
                    )
                else:
                    save_metadata(
                        checkpoint_dir=actual_checkpoint_dirname,
                        epoch=epoch,
                        val_loss=val_loss,
                        best_val_loss=self._best_metrics,
                    )
            # Link the other checkpoints to the one that has been saved
            # Only to be performed once, hence on rank 0
            if self.rank == 0:
                for linked_checkpoint_filename in checkpoint_dirnames[1:]:
                    if not local:
                        checkpoint_future.add_done_callback(
                            on_future(
                                link_checkpoint,
                                actual_checkpoint_dirname,
                                linked_checkpoint_filename,
                            )
                        )
                    else:
                        link_checkpoint(
                            actual_checkpoint_dirname,
                            linked_checkpoint_filename,
                        )

        if force and checkpoint_future is not None:
            # Make checkpoint saving synchronous
            checkpoint_future.result()
            checkpoint_future = None

        return checkpoint_future
