import dataclasses
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from the_well.data import WellDataset
from the_well.data.augmentation import Augmentation
from the_well.data.utils import WELL_DATASETS, flatten_field_names
from torch.utils.data import Dataset

from walrus.data.inflated_dataset import (
    BatchInflatedWellDataset,
)

from .utils import get_dict_depth

logger = logging.getLogger(__name__)

# Number of times to retry a single-sample read before giving up, and the base
# (exponential) backoff in seconds. Streamed datasets (e.g. shear_flow over
# hf://) can transiently drop a connection mid-chunk
# (httpx.RemoteProtocolError), which would otherwise kill the whole run.
_READ_MAX_RETRIES = int(os.environ.get("WALRUS_READ_MAX_RETRIES", "5"))
_READ_BACKOFF_BASE = float(os.environ.get("WALRUS_READ_BACKOFF_BASE", "1.0"))


def _read_with_retry(read_fn, *, describe):
    """Call read_fn(), retrying on transient errors with exponential backoff.

    Retries any exception (network reads surface as a variety of httpx/httpcore/
    OSError types), logging each attempt. Re-raises the last error if all
    attempts fail so the caller can handle it as before.
    """
    last_exc = None
    for attempt in range(_READ_MAX_RETRIES + 1):
        try:
            return read_fn()
        except Exception as e:  # noqa: BLE001 - intentionally broad; network reads vary
            last_exc = e
            if attempt < _READ_MAX_RETRIES:
                delay = _READ_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"Read failed for {describe} (attempt {attempt + 1}/"
                    f"{_READ_MAX_RETRIES + 1}): {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.1f}s."
                )
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def update_field_names(metadata, field_name_transforms):
    scalar_names = [
        field_name_transforms.get(name, name) for name in metadata.scalar_names
    ]
    field_names = {
        k: [
            f"{field_name_transforms[name]}_{name}"
            if name in field_name_transforms
            else name
            for name in field_names
        ]
        for k, field_names in metadata.field_names.items()
    }
    constant_field_names = {
        k: [
            f"{field_name_transforms[name]}_{name}"
            if name in field_name_transforms
            else name
            for name in field_names
        ]
        for k, field_names in metadata.constant_field_names.items()
    }
    return dataclasses.replace(
        metadata,
        scalar_names=scalar_names,
        field_names=field_names,
        constant_field_names=constant_field_names,
    )


class MixedWellDataset(Dataset):
    """
    Combination of multiple Well datasets. Returns data in B x T x H [x W [x D]] x C format.

    Train/Test/Valid is assumed to occur on a folder level and this is not performed in this
    object.

    Most parameters are passed to inner datasets.

    Parameters
    ----------
    paths :
        Path to directory of HDF5 files, one of path or well_base_path+well_dataset_name
          must be specified
    normalization_path:
        Path to normalization constants - assumed to be in same format as constructed data.
    well_base_path :
        Path to well dataset directory, only used with dataset_name
    well_dataset_info:
        Dictionary containing for each dataset:
        - include_filters: List of strings to filter files to include
        - exclude_filters: List of strings to filter files to exclude
        - path: Optional custom path for this specific dataset
        - field_transforms: Dict of transforms to apply to fields in this dataset
    well_split_name :
        Name of split to load - options are 'train', 'valid', 'test'
    include_filters :
        Only include files whose name contains at least one of these strings
    exclude_filters :
        Exclude any files whose name contains at least one of these strings
    use_normalization:
        Whether to normalize data in the dataset
    include_normalization_in_sample: bool, default=False
        Whether to include normalization constants in the sample
    n_steps_input :
        Number of steps to include in each sample
    n_steps_output :
        Number of steps to include in y
    dt_stride :
        Minimum stride between samples
    max_dt_stride :
        Maximum stride between samples
    restrict_num_trajectories :
        If set to a float in (0, 1), restrict the number of trajectories in the dataset to this fraction of the total.
        If set to an int > 1, restrict the number of trajectories in the dataset to this number.
        If None, use all trajectories.
    restrict_num_samples :
        If set to a float in (0, 1), restrict the number of samples in the dataset to this fraction of the total.
        If set to an int > 1, restrict the number of samples in the dataset to this number.
        If None, use all samples.
    restriction_seed :
        Seed used to generate restriction set. Necessary to ensure same set is sampled across runs.
    start_output_steps_at_t :
        During rollout validation, start outputting predictions at this time index.
        This is useful to compare models that require different initial context to make accurate predictions.
        Default is -1, which means to start at the beginning of the trajectory.
    flatten_tensors :
        Whether to flatten tensor valued field into channels
    cache_small :
        Whether to cache all values that do not vary in time or sample
          in memory for faster access
    max_cache_size :
        Maximum numel of constant tensor to cache
    return_grid :
        Whether to return grid coordinates
    normalize_time_grid :
        Whether to normalize the time grid so that it returns relative rather than absolute time.
        Default is True as absolute time generally leads to better scenario fitting in the well,
        but poor generalization.
    boundary_return_type : options=['padding', 'mask', 'exact']
        How to return boundary conditions. Currently only padding supported.
    full_trajectory_mode :
        Overrides to return full trajectory starting from t0 instead of samples
            for long run validation.
    name_override :
        Override name of dataset (used for more precise logging)
    transforms :
        Dict of transforms to apply to data. Each key should be a dataset name.
    global_field_transforms :
        Dict of pointwise transforms to apply to specific fields across all datasets. Each key should be a field name.
    min_std:
        Minimum standard deviation for field normalization. If a field standard
        deviation is lower than this value, it is replaced by this value.
    storage_options :
            Option for the ffspec storage.
    """

    def __init__(
        self,
        *,
        well_base_path: str,
        well_dataset_info: Dict[
            str,
            Dict[
                Literal[
                    "include_filters", "exclude_filters", "path", "field_transforms"
                ],
                List[str] | str | Dict[str, str],
            ],
        ],
        tie_fields: bool = True,
        use_effective_batch_size: bool = False,
        prefetch_field_names: bool = True,
        normalization_path: Optional[str] = None,  # "../stats/",
        well_split_name: str = "train",
        use_normalization: bool = False,
        max_rollout_steps=100,
        field_index_map_override: Dict[str, int] = {},
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        restrict_num_trajectories: Optional[float | int] = None,
        restrict_num_samples: Optional[float | int] = None,
        restriction_seed: int = 0,
        start_output_steps_at_t: int = -1,
        flatten_tensors: bool = True,
        cache_small: bool = True,
        max_cache_size: float = 1e9,
        return_grid: bool = True,
        normalize_time_grid: bool = True,
        boundary_return_type: str = "padding",
        full_trajectory_mode: bool = False,
        name_override: Optional[str] = None,
        transform: Optional[Union[Augmentation, Dict[str, Augmentation]]] = None,
        global_field_transforms: Optional[Dict[str, str]] = {},
        min_std: float = 1e-4,
        storage_options: Optional[Dict] = None,
        inner_dataset_type: WellDataset = BatchInflatedWellDataset,
        dataset_kws: Optional[Union[dict, Dict[str, dict]]] = None,
    ):
        super().__init__()
        # Global dicts used by Mixed DSET.
        self.well_base_path = well_base_path
        self.well_dataset_info = well_dataset_info
        self.prefetch_field_names = prefetch_field_names
        self.tie_fields = tie_fields
        self.well_split_name = well_split_name
        self.sub_dsets = []
        self.use_effective_batch_size = use_effective_batch_size
        self.restrict_num_trajectories = restrict_num_trajectories
        self.restrict_num_samples = restrict_num_samples
        self.restriction_seed = restriction_seed
        self.start_output_steps_at_t = start_output_steps_at_t
        self.effective_batch_sizes = []
        self.offsets = [0]
        self.dset_to_metadata: dict[str, Any] = {}
        self.field_name_transforms = {}
        self.inner_dataset_type = inner_dataset_type
        global_field_transforms = (
            OmegaConf.to_container(global_field_transforms)
            if isinstance(global_field_transforms, DictConfig)
            else global_field_transforms
        )
        global_field_name_transforms = {
            k: v.split(".", 1)[-1] for k, v in global_field_transforms.items()
        }

        if transform is not None:
            # If transform is a single Augmentation, apply it to all datasets
            if isinstance(transform, Augmentation):
                transform = {k: transform for k in well_dataset_info.keys()}

            # Check that dataset names in transform match those in well_dataset_info
            assert set(transform.keys()).issubset(set(well_dataset_info.keys())), (
                f"Expected transform keys {transform.keys()} to be a subset of well_dataset_info keys {well_dataset_info.keys()}."
            )

        if dataset_kws is not None:
            # If dataset_kws is a single dict of depth 1, apply it to all datasets
            if isinstance(dataset_kws, dict) and get_dict_depth(dataset_kws) == 1:
                dataset_kws = {k: dataset_kws for k in well_dataset_info.keys()}

            # Check that dataset names in dataset_kws match those in well_dataset_info
            assert set(dataset_kws.keys()).issubset(set(well_dataset_info.keys())), (
                f"Expected dataset_kws keys {dataset_kws.keys()} to be a subset of well_dataset_info keys {well_dataset_info.keys()}."
            )

        for dataset_name, info in well_dataset_info.items():
            include_filters = info.get("include_filters", [])
            exclude_filters = info.get("exclude_filters", [])
            step_downsample_factor = info.get("step_downsample_factor", 1)
            batch_downsample_factor = info.get("batch_downsample_factor", 1)
            dset_field_transforms = (
                OmegaConf.to_container(info["field_transforms"])
                if "field_transforms" in info
                else {}
            )
            local_name_transforms = {
                k: v.split(".", 1)[-1] for k, v in dset_field_transforms.items()
            }
            field_name_transforms = global_field_name_transforms | local_name_transforms
            dset_field_transforms = global_field_transforms | dset_field_transforms
            dataset_path = info.get("path", None)
            normalization_path = info.get("normalization_path", None)

            try:
                subdset = self.inner_dataset_type(
                    path=dataset_path,
                    normalization_path=normalization_path,
                    well_base_path=well_base_path,
                    well_dataset_name=dataset_name,
                    well_split_name=well_split_name,
                    include_filters=include_filters,
                    exclude_filters=exclude_filters,
                    use_normalization=use_normalization,
                    max_rollout_steps=max_rollout_steps,
                    n_steps_input=max(1, int(n_steps_input * step_downsample_factor)),
                    n_steps_output=n_steps_output,
                    min_dt_stride=min_dt_stride,
                    max_dt_stride=max_dt_stride,
                    restrict_num_trajectories=restrict_num_trajectories,
                    restrict_num_samples=restrict_num_samples,
                    restriction_seed=restriction_seed,
                    start_output_steps_at_t=start_output_steps_at_t,
                    flatten_tensors=flatten_tensors,
                    cache_small=cache_small,
                    max_cache_size=max_cache_size,
                    return_grid=return_grid,
                    normalize_time_grid=normalize_time_grid,
                    boundary_return_type=boundary_return_type,
                    full_trajectory_mode=full_trajectory_mode,
                    name_override=name_override,
                    transform=(
                        transform[dataset_name]
                        if transform is not None and dataset_name in transform
                        else None
                    ),
                    field_transforms=dset_field_transforms,
                    min_std=min_std,
                    storage_options=storage_options,
                    **(
                        dataset_kws[dataset_name]
                        if dataset_kws is not None and dataset_name in dataset_kws
                        else {}
                    ),
                )
            except OSError as e:
                data_dir = dataset_path or os.path.join(
                    well_base_path, dataset_name, "data", well_split_name
                )
                corrupt = []
                if os.path.isdir(data_dir):
                    import h5py
                    for fname in sorted(os.listdir(data_dir)):
                        if not fname.endswith((".hdf5", ".h5")):
                            continue
                        fpath = os.path.join(data_dir, fname)
                        try:
                            with h5py.File(fpath, "r"):
                                pass
                        except OSError:
                            corrupt.append(fpath)
                if corrupt:
                    raise OSError(
                        f"Corrupt file(s) in {data_dir}:\n"
                        + "\n".join(f"  {f}" for f in corrupt)
                    ) from e
                raise OSError(f"Failed to load dataset '{dataset_name}' from {data_dir}") from e
            self.field_name_transforms[subdset.metadata.dataset_name] = (
                field_name_transforms
            )
            try:
                offset = len(subdset)
                self.offsets.append(self.offsets[-1] + offset)
            except ValueError:
                raise ValueError(
                    f"Dataset '{dataset_name}' (path={dataset_path}) is empty. "
                    f"Check that n_steps ({max(1, int(n_steps_input * step_downsample_factor))} + {n_steps_output}) "
                    f"< trajectory_length in file, or that the dataset exists at "
                    f"{well_base_path}/{dataset_name}/data/{well_split_name}/"
                )
            self.sub_dsets.append(subdset)
            self.dset_to_metadata[dataset_name] = subdset.metadata
            self.effective_batch_sizes.append(batch_downsample_factor)
            self.offsets[0] = -1  # So 0 is in the first segment
        self.field_to_index_map = self._build_subset_dict(field_index_map_override)

    def _build_subset_dict(
        self, field_index_override: Dict[str, int]
    ) -> Dict[str, int]:
        # Maps fields to subsets of variables
        field_to_index = field_index_override
        max_index = (
            0
            if len(field_index_override) == 0
            else 1 + max(list(field_index_override.values()))
        )
        if self.prefetch_field_names:
            for dataset_name in WELL_DATASETS:
                try:
                    temp_dset = self.inner_dataset_type(
                        well_base_path=self.well_base_path,
                        well_dataset_name=dataset_name,
                        well_split_name=self.well_split_name,
                        use_normalization=False,  # Don't need normalization to get this data
                    )
                except Exception:
                    logger.warning(f"Failed to load {dataset_name} dataset")
                    continue
                metadata = temp_dset.metadata
                metadata = update_field_names(
                    metadata, self.field_name_transforms.get(metadata.dataset_name, {})
                )
                field_names = flatten_field_names(metadata)
                for field_name in field_names:
                    # If we're not tying field names, then add dataset name to field name for the key
                    if not self.tie_fields:
                        field_name = f"{dataset_name}_{field_name}"
                    if field_name not in field_to_index:
                        field_to_index[field_name] = max_index
                        max_index += 1
        # If we added any extras, make sure they're represented as well
        for dataset_name, info in self.well_dataset_info.items():
            if dataset_name in WELL_DATASETS and self.prefetch_field_names:
                continue  # Already processed
            dataset_path = info.get("path", None)
            if dataset_path is not None:
                temp_dset = self.inner_dataset_type(
                    path=dataset_path,
                    well_split_name=self.well_split_name,
                    use_normalization=False,
                )
            elif dataset_name in WELL_DATASETS:
                temp_dset = self.inner_dataset_type(
                    well_base_path=self.well_base_path,
                    well_dataset_name=dataset_name,
                    well_split_name=self.well_split_name,
                    use_normalization=False,  # Don't need normalization to get this data
                )
            else:
                raise ValueError(
                    f"Unknown dataset {dataset_name}. Please provide path."
                )
            metadata = temp_dset.metadata
            metadata = update_field_names(
                metadata, self.field_name_transforms.get(metadata.dataset_name, {})
            )
            field_names = flatten_field_names(metadata)
            for field_name in field_names:
                # If we're not tying field names, then add dataset name to field name for the key
                if not self.tie_fields:
                    field_name = f"{dataset_name}_{field_name}"
                if field_name not in field_to_index:
                    field_to_index[field_name] = max_index
                    max_index += 1
        return field_to_index

    def __getitem__(self, indices: int | List[int]):
        if isinstance(indices, int):
            indices = [indices]
        # This will likely fail if the list covers multiple datasets, so just assume its
        # all from one.
        file_idx = (
            np.searchsorted(self.offsets, indices[0], side="right") - 1
        )  # which dataset are we are on
        local_indexes = [index - max(self.offsets[file_idx], 0) for index in indices]
        try:
            data = _read_with_retry(
                lambda: self.sub_dsets[file_idx][local_indexes],
                describe=f"dataset={self.sub_dsets[file_idx].metadata.dataset_name} "
                f"file_idx={file_idx} local_indexes={local_indexes}",
            )
        except Exception:
            raise IndexError(
                "FAILED AT ",
                file_idx,
                local_indexes,
                indices,
                int(os.environ.get("RANK", 0)),
            )
        current_metadata = self.sub_dsets[file_idx].metadata
        current_metadata = update_field_names(
            current_metadata,
            self.field_name_transforms.get(current_metadata.dataset_name, {}),
        )
        field_names = flatten_field_names(current_metadata)
        if not self.tie_fields:
            field_names = [
                f"{current_metadata.dataset_name}_{field}" for field in field_names
            ]
        field_indices = [self.field_to_index_map[field] for field in field_names]
        data["field_indices"] = torch.tensor(field_indices)
        data["metadata"] = current_metadata
        return data

    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])


if __name__ == "__main__":
    well_base_path = "/mnt/home/polymathic/ceph/the_well/"
    data = MixedWellDataset(
        well_base_path=well_base_path,
        well_dataset_info={
            "active_matter": {"include_filters": [], "exclude_filters": []},
            "planetswe": {"include_filters": [], "exclude_filters": []},
        },
    )

    for i in range(len(data)):
        x = data[i]
        if i % 1000 == 0:
            print(x)

    # print(len(data))
    # print(data[0])
