import os, re

import numpy as np
import torch
import pandas as pd
from typing import Literal, Union

import h5py
import pyvista as pv
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.ticker as mticker
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

import seaborn as sns
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

from ..EarendilsLight import EarendilsLight


class SAM():
    """
    SAM – Simulations & Analytics Module
    ─────────────────────────────────────
    The steadfast companion to FRODO.

    SAM is a utility module designed to assist with machine-learning
    workflows, data transformation, and surrogate model preparation for
    CFD simulations and other engineering data pipelines.

    Sub-modules
    -----------
    Gardener   – joint tensor assembly and normalisation for ML training.
    HDF5reader – thin HDF5 file reader (numpy / torch output).
    Backpack   – file utilities, mesh helpers, CSV parsers.
    Weapons    – point-cloud sorting, surface derivatives, GMM clustering,
                 mesh-to-mesh interpolation.
    DictVisualizer – rich / networkx / pprint helpers for nested dicts.
    """

    light = EarendilsLight(__name__)

    @classmethod
    def some_light(cls, name=None):
        """Shortcut to Eärendil's Light help system."""
        return cls.light.help(name)

    # =========================================================================
    # GARDENER
    # =========================================================================
    class Gardener:

        @staticmethod
        def create_final_tensor(
            tensor_ptos: Union[torch.Tensor, np.ndarray],
            tensor_flcc: Union[torch.Tensor, np.ndarray],
            tensors_out: list,
            tensors_aux: list = None,
            sol: Union[list, tuple, int, str] = 'all',
            idx_flcc: Union[list, tuple, str] = 'all',
            ref: Union[dict, None] = None,
            verbose: bool = False,
        ) -> dict:
            """
            Assemble mesh coordinates, flight conditions, auxiliary arrays and
            output variables into a single flat ML-ready tensor with optional
            min-max normalisation.

            The tensor layout per row is::

                [x_coords | flight_conditions | aux_features | output_variables]

            Each row corresponds to one (point, case) pair. The total number
            of rows is ``n_points × n_cases``.

            Parameters
            ----------
            tensor_ptos : array-like, shape (n_points, n_coord)
                Mesh node or cell-centroid coordinates.
            tensor_flcc : array-like, shape (n_cases, n_dvars)
                Design / flight condition variables (AoA, Mach, …).
            tensors_out : list of array-like
                Each element must be 2-D ``(n_points, n_cases)`` or 3-D
                ``(n_points, n_cases, n_outputs)``.  1-D arrays raise
                ``ValueError``.
            tensors_aux : list of array-like or None
                Auxiliary features.  Accepted shapes:
                * 1-D ``(n_points,)``  – point-only feature repeated per case.
                * 2-D ``(n_points, n_cases)``
                * 3-D ``(n_points, n_cases, n_aux)``
            sol : 'all', int or list[int]
                Which output channel(s) to keep after stacking. 'all' keeps
                every channel; an int selects a single channel; a list selects
                multiple channels.
            idx_flcc : 'all' or list[int]
                Subset of case indices to include.  Applied before sol
                selection.
            ref : dict or None
                Reference normalisation dict with keys 'mins' and 'maxs'
                (torch.Tensor).  If None, min/max are computed from the data.
            verbose : bool
                Print intermediate tensor shapes.

            Returns
            -------
            dict with keys:
                * ``'tensor'``  – raw joint tensor, shape (n_pts×n_cases, n_cols)
                * ``'scaled'``  – min-max normalised tensor, same shape
                * ``'mins'``    – per-column minimum values (1-D tensor)
                * ``'maxs'``    – per-column maximum values (1-D tensor)
                * ``'info'``    – dict with ``'ninputs'`` and ``'noutputs'``

            Examples
            --------
            ::

                result = SAM.Gardener.create_final_tensor(
                    tensor_ptos = coords,          # (500, 2)
                    tensor_flcc = flight_conds,    # (100, 2)
                    tensors_out = [cp_array],      # (500, 100)
                    tensors_aux = [],
                    sol='all',
                )
                # result['tensor'].shape → (50000, 5)
            """
            if tensors_aux is None:
                tensors_aux = []

            def _to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.clone()
                elif isinstance(x, np.ndarray):
                    return torch.from_numpy(x).clone()
                raise TypeError(
                    "All inputs must be torch.Tensor or np.ndarray."
                )

            tensor_ptos = _to_tensor(tensor_ptos)
            tensor_flcc = _to_tensor(tensor_flcc)
            tensors_out = [_to_tensor(t) for t in tensors_out]
            tensors_aux = [_to_tensor(t) for t in tensors_aux]

            nptos  = tensor_ptos.shape[0]
            ncases = tensor_flcc.shape[0]

            if tensor_ptos.dim() == 1:
                tensor_ptos = tensor_ptos.unsqueeze(1)
            if tensor_flcc.dim() == 1:
                tensor_flcc = tensor_flcc.unsqueeze(1)

            # Normalise outputs to 3-D (n_points, n_cases, n_out)
            for i, out in enumerate(tensors_out):
                if out.dim() == 1:
                    raise ValueError(
                        f"tensors_out[{i}] must be at least 2-D (found 1-D)."
                    )
                if out.dim() == 2:
                    tensors_out[i] = out.unsqueeze(2)
                elif out.dim() > 3:
                    raise ValueError(
                        f"tensors_out[{i}] has {out.dim()} dims; expected 2 or 3."
                    )

            # ── Case selection ───────────────────────────────────────────────
            if idx_flcc == 'all':
                idx_selected = torch.arange(ncases)
            else:
                if not isinstance(idx_flcc, (list, tuple)):
                    raise TypeError("idx_flcc must be list, tuple or 'all'.")
                idx_selected = torch.tensor(idx_flcc, dtype=torch.long)
                if (idx_selected >= ncases).any() or (idx_selected < 0).any():
                    raise IndexError("idx_flcc contains out-of-range indices.")

            tensor_flcc = tensor_flcc[idx_selected, :]
            ncases = tensor_flcc.shape[0]
            tensors_out = [
                out[:, idx_selected, :] if out.dim() >= 2 else out
                for out in tensors_out
            ]
            new_aux = []
            for aux in tensors_aux:
                if aux.dim() == 1:
                    new_aux.append(aux)
                elif aux.dim() == 2:
                    new_aux.append(aux[:, idx_selected])
                elif aux.dim() == 3:
                    new_aux.append(aux[:, idx_selected, :])
            tensors_aux = new_aux

            # ── Channel selection ────────────────────────────────────────────
            if sol == 'all':
                pass
            elif isinstance(sol, int):
                tensors_out = [
                    out[:, :, sol:sol + 1] if out.shape[2] > 1 else out
                    for out in tensors_out
                ]
                tensors_aux = [
                    aux[:, :, sol:sol + 1]
                    if (aux.dim() == 3 and aux.shape[2] > 1) else aux
                    for aux in tensors_aux
                ]
            elif isinstance(sol, (list, tuple)):
                tensors_out = [
                    out[:, :, sol] if out.shape[2] > 1 else out
                    for out in tensors_out
                ]
                tensors_aux = [
                    aux[:, :, sol]
                    if (aux.dim() == 3 and aux.shape[2] > 1) else aux
                    for aux in tensors_aux
                ]
            else:
                raise TypeError("sol must be int, list of ints, or 'all'.")

            # ── Expand to row-wise (n_points × n_cases) ──────────────────────
            points_rep  = tensor_ptos.repeat(ncases, 1)
            flcc_exp    = tensor_flcc.repeat_interleave(nptos, dim=0)

            if verbose:
                print(f'points_repeated : {points_rep.shape}')
                print(f'flcc_expanded   : {flcc_exp.shape}')

            def _flatten_aux(aux):
                if aux.dim() == 1:
                    return aux.repeat(ncases).unsqueeze(1)
                elif aux.dim() == 2:
                    return aux.permute(1, 0).reshape(-1, 1)
                elif aux.dim() == 3:
                    return aux.permute(1, 0, 2).reshape(-1, aux.shape[2])
                raise RuntimeError(f"Unexpected aux dim {aux.dim()}.")

            def _flatten_out(out):
                return out.permute(1, 0, 2).reshape(-1, out.shape[2])

            aux_flat = []
            for aux in tensors_aux:
                flat = _flatten_aux(aux)
                if verbose:
                    print(f'aux dim {aux.dim()} → {flat.shape}')
                aux_flat.append(flat)

            out_flat = []
            for out in tensors_out:
                flat = _flatten_out(out)
                if verbose:
                    print(f'out {out.shape} → {flat.shape}')
                out_flat.append(flat)

            final_tensor = torch.cat(
                [points_rep, flcc_exp] + aux_flat + out_flat, dim=1
            )
            if verbose:
                print(f'Final tensor shape: {final_tensor.shape}')

            # ── Min-max normalisation ────────────────────────────────────────
            if ref is None:
                min_vals = final_tensor.min(dim=0, keepdim=True)[0]
                max_vals = final_tensor.max(dim=0, keepdim=True)[0]
            else:
                min_vals = ref['mins']
                max_vals = ref['maxs']

            denom = max_vals - min_vals
            denom[denom == 0] = 1e-8
            final_tensor_scaled = (final_tensor - min_vals) / denom

            aux_cols = sum(
                1 if a.dim() == 1 else 1 if a.dim() == 2 else a.shape[2]
                for a in tensors_aux
            )
            noutputs = sum(o.shape[2] for o in tensors_out)

            return {
                'tensor': final_tensor,
                'scaled': final_tensor_scaled,
                'mins':   min_vals.squeeze(),
                'maxs':   max_vals.squeeze(),
                'info': {
                    'ninputs':  tensor_ptos.shape[1] + tensor_flcc.shape[1] + aux_cols,
                    'noutputs': noutputs,
                },
            }

        @staticmethod
        def create_final_tensor_scored(
            tensor_ptos: torch.Tensor,
            tensor_flcc: torch.Tensor,
            tensor_out: torch.Tensor,
            score_law: str,
            sol='all',
            score_csv_path=None,
            n=None,
            ref=None,
            verbose: bool = False,
        ) -> dict:
            """
            Assemble a joint tensor and attach an importance score to each
            row based on the output value distribution.

            The score rewards under-represented value ranges so that training
            samplers can up-weight rare regions of the output space.

            Parameters
            ----------
            tensor_ptos : torch.Tensor, shape (n_points, n_coord)
            tensor_flcc : torch.Tensor, shape (n_cases, n_dvars)
            tensor_out  : torch.Tensor, shape (n_points, n_cases, n_outputs)
                Must be 3-D; 2-D arrays are automatically unsqueezed.
            score_law : str
                Formula used to derive the per-bin importance weight from
                frequency.  Options:

                * ``'log10'``    – ``1 / log10(freq)``
                * ``'inv'``      – ``1 / freq``
                * ``'sqrt_inv'`` – ``1 / sqrt(freq)``
                * ``'exp_inv'``  – ``exp(-freq / max_freq)``
                * ``'linear'``   – ``1 - freq / max_freq``

            sol : int or 'all'
                Output channel index to select.
            score_csv_path : str or None
                If provided, saves one CSV of scores per output variable to
                this directory.
            n : int or None
                Limit to the first *n* cases.
            ref : dict or None
                Reference normalisation dict (keys 'mins', 'maxs').
            verbose : bool
                Print progress information.

            Returns
            -------
            dict with keys: 'tensor', 'scaled', 'mins', 'maxs', 'score',
            'score_array', 'n_vec', 'bin_vec', 'score_per_bin', 'info'.

            Examples
            --------
            ::

                result = SAM.Gardener.create_final_tensor_scored(
                    tensor_ptos, tensor_flcc, tensor_out,
                    score_law='inv',
                )
                weights = result['score']   # shape (n_pts × n_cases, n_out)
            """
            if len(tensor_out.shape) == 2:
                print('WARNING: tensor_out is 2-D; adding a third dimension.')
                tensor_out = tensor_out.unsqueeze(2)
            elif len(tensor_out.shape) == 1:
                raise ValueError(
                    "tensor_out must be 3-D but received a 1-D tensor."
                )

            if n is not None:
                tensor_flcc = tensor_flcc[:n, :]
                tensor_out  = tensor_out[:, :n, :]

            if sol != 'all':
                tensor_out = tensor_out[:, :, sol]

            if verbose:
                print('BUILDING SCORED DATASET\n')
                print(f'  Ptos : {tensor_ptos.shape}')
                print(f'  FLCC : {tensor_flcc.shape}')
                print(f'  out  : {tensor_out.shape}')

            P, F, C = tensor_out.shape

            points_rep  = tensor_ptos.repeat(tensor_flcc.size(0), 1)
            flcc_exp    = tensor_flcc.repeat_interleave(tensor_ptos.size(0), dim=0)
            out_flat    = tensor_out.permute(1, 0, 2).reshape(-1, C)

            tensor_score   = torch.zeros_like(out_flat)
            n_vec, bin_vec, score_per_bin_vec = [], [], []

            for c in range(out_flat.shape[1]):
                freq, bins = np.histogram(out_flat[:, c].numpy(), bins=100)

                if score_law == 'log10':
                    inv_freq = np.where(freq > 0, 1 / np.log10(freq + 1), 0)
                elif score_law == 'inv':
                    inv_freq = np.where(freq > 1, 1 / freq, 1)
                elif score_law == 'sqrt_inv':
                    inv_freq = np.where(freq > 0, 1 / np.sqrt(freq), 0)
                elif score_law == 'exp_inv':
                    max_n = max(freq.max(), 1)
                    inv_freq = np.exp(-freq / max_n)
                elif score_law == 'linear':
                    max_n = max(freq.max(), 1)
                    inv_freq = 1 - freq / max_n
                else:
                    raise ValueError(
                        f"score_law '{score_law}' not recognised. "
                        "Options: 'log10', 'inv', 'sqrt_inv', 'exp_inv', 'linear'."
                    )

                spb = inv_freq * np.ones(len(inv_freq))
                spb = np.nan_to_num(spb, nan=0.0)
                rng = spb.max() - spb.min()
                if rng > 0:
                    spb = (spb - spb.min()) / rng

                bin_idx = np.clip(
                    np.digitize(out_flat[:, c].numpy(), bins, right=True) - 1,
                    0, len(freq) - 1,
                )
                n_vec.append(freq)
                bin_vec.append(bin_idx)
                score_per_bin_vec.append(spb)
                tensor_score[:, c] = torch.from_numpy(spb[bin_idx])

            final_tensor = torch.cat((points_rep, flcc_exp, out_flat), dim=1)

            if ref is None:
                min_vals = final_tensor.min(dim=0, keepdim=True)[0]
                max_vals = final_tensor.max(dim=0, keepdim=True)[0]
            else:
                min_vals = ref['mins']
                max_vals = ref['maxs']

            denom = max_vals - min_vals
            denom[denom == 0] = 1e-8
            final_tensor_scaled = (final_tensor - min_vals) / denom

            tensor_score_ = tensor_score.reshape(P, F, C)
            if verbose:
                print(f'  tensor_score : {tensor_score_.shape}')
                print(f'  tensor       : {final_tensor.shape}')

            if score_csv_path is not None:
                for var in range(tensor_out.shape[2]):
                    header = ['x', 'y', 'z'] + [
                        f'case_{c}' for c in range(tensor_flcc.shape[0])
                    ]
                    df = pd.DataFrame(
                        torch.cat(
                            (tensor_ptos, tensor_score_[:, :, var]), dim=1
                        )
                    )
                    df.to_csv(
                        f'{score_csv_path}/tensor_score_var_{var}.csv',
                        header=header, index=False,
                    )
                print(f'Exported tensor_score CSVs to {score_csv_path}/')

            return {
                'tensor':        final_tensor,
                'scaled':        final_tensor_scaled,
                'mins':          min_vals.squeeze(),
                'maxs':          max_vals.squeeze(),
                'score':         torch.from_numpy(tensor_score.numpy().copy()),
                'score_array':   tensor_score_,
                'n_vec':         torch.tensor(np.array(n_vec, dtype=np.float32)),
                'bin_vec':       torch.tensor(np.array(bin_vec, dtype=np.int64)),
                'score_per_bin': torch.tensor(
                    np.array(score_per_bin_vec, dtype=np.float32)
                ),
                'info': {
                    'ninputs':  tensor_ptos.shape[1] + tensor_flcc.shape[1],
                    'noutputs': tensor_out.shape[2],
                },
            }

        @staticmethod
        def concatenate_sets(sets: tuple, ref: int = 0, score: bool = False):
            """
            Concatenate multiple joint-tensor result dicts along the row axis.

            Parameters
            ----------
            sets : tuple of dict
                Result dicts produced by ``create_final_tensor`` or
                ``create_final_tensor_scored``.
            ref : int
                Index of the dict whose 'mins', 'maxs' and 'info' are used
                in the output.
            score : bool
                Not yet implemented for scored datasets.

            Returns
            -------
            dict: Merged result dict.

            Examples
            --------
            ::

                merged = SAM.Gardener.concatenate_sets((result_a, result_b))
            """
            if score:
                raise NotImplementedError(
                    "concatenate_sets is not yet implemented for scored datasets."
                )

            if ref >= len(sets):
                raise ValueError(
                    f"ref ({ref}) must be a valid index in sets "
                    f"(0 to {len(sets) - 1})."
                )

            new_set = dict(sets[0])
            for one_set in sets[1:]:
                for key in ('tensor', 'scaled'):
                    new_set[key] = torch.cat(
                        (new_set[key], one_set[key]), dim=0
                    )
            for key in ('mins', 'maxs', 'info'):
                new_set[key] = sets[ref][key]

            return new_set

        @staticmethod
        def reduce_dataset_per_frequency(
            dataset: dict,
            lim: int,
            reduce_factor: float = 0.8,
            ref=None,
            plot_path=None,
        ) -> dict:
            """
            Remove a fraction of rows that fall into over-represented value
            bins of the output distribution.

            Parameters
            ----------
            dataset : dict
                Result from ``create_final_tensor_scored``.
            lim : int
                Frequency threshold above which a bin is considered
                over-represented.
            reduce_factor : float
                Fraction of over-represented rows to remove. Default 0.8.
            ref : dict or None
                Reference normalisation dict. If None, recomputed from data.
            plot_path : str or None
                If provided, saves diagnostic histograms to this directory.

            Returns
            -------
            dict: Filtered result dict with updated 'score', 'n_vec',
            'bin_vec' and normalisation.
            """
            n_vec = dataset['n_vec'][0]
            bins_to_check = torch.where(n_vec > lim)[0]
            ind_bin = torch.nonzero(
                torch.isin(dataset['bin_vec'][0], bins_to_check), as_tuple=True
            )[0]

            ind_perm = ind_bin[torch.randperm(ind_bin.shape[0])]
            to_remove = ind_perm[:int(ind_bin.shape[0] * reduce_factor)]

            final_mask = torch.ones_like(dataset['bin_vec'][0], dtype=torch.bool)
            final_mask[to_remove] = False

            final_tensor = dataset['tensor'][final_mask, :]

            if ref is None:
                min_vals = final_tensor.min(dim=0, keepdim=True)[0]
                max_vals = final_tensor.max(dim=0, keepdim=True)[0]
            else:
                min_vals = ref['mins']
                max_vals = ref['maxs']

            denom = max_vals - min_vals
            denom[denom == 0] = 1e-8
            final_tensor_scaled = (final_tensor - min_vals) / denom

            print(f'Filtered dataset shape: {final_tensor.shape}')
            n_outputs = dataset['score'].shape[1]

            n_vec_new, bin_vec_new = [], []
            tensor_score = torch.zeros((final_tensor.shape[0], n_outputs))

            for c in range(n_outputs):
                n, bins = np.histogram(final_tensor[:, -c].numpy(), bins=100)
                inv_freq = 1 / np.maximum(n, 1)
                scale_v = np.linspace(0, 1, len(inv_freq))
                spb = inv_freq * scale_v
                rng = spb.max() - spb.min()
                if rng > 0:
                    spb = (spb - spb.min()) / rng
                bin_idx = np.clip(
                    np.digitize(final_tensor[:, -c].numpy(), bins, right=True) - 1,
                    0, len(n) - 1,
                )
                n_vec_new.append(n)
                bin_vec_new.append(bin_idx)
                tensor_score[:, c] = torch.from_numpy(spb[bin_idx].T)

            if plot_path is not None:
                fig, ax = plt.subplots(2, 2, figsize=(20, 20))
                ax = ax.flatten()
                for i, (data_hist, title) in enumerate([
                    (final_tensor[:, -1],       'Histogram cp (filtered)'),
                    (dataset['tensor'][:, -1],  'Histogram cp (original)'),
                    (tensor_score,              'Score (filtered)'),
                    (dataset['score'],          'Score (original)'),
                ]):
                    ax[i].hist(data_hist, bins=100, color='steelblue',
                               alpha=0.7, edgecolor='black')
                    ax[i].set(title=title, xlabel='Value', ylabel='Frequency')
                    ax[i].set_yscale('log')
                    ax[i].grid(True, linestyle='--', alpha=0.5)
                fig.savefig(f"{plot_path}/hist_dataset_reduced.jpg")
                plt.show()

            return {
                'tensor': final_tensor,
                'scaled': final_tensor_scaled,
                'mins':   min_vals.squeeze(),
                'maxs':   max_vals.squeeze(),
                'score':  tensor_score,
                'n_vec':  torch.from_numpy(np.array(n_vec_new)),
                'bin_vec': torch.from_numpy(np.array(bin_vec_new)),
                'info':   dataset['info'],
            }

    # =========================================================================
    # HDF5READER
    # =========================================================================
    class HDF5reader:
        """
        Thin wrapper around h5py for reading datasets as numpy arrays or
        torch tensors.

        Parameters
        ----------
        file_path : str
            Path to the .h5 file.
        verbose : bool
            If True, prints all dataset paths found in the file.

        Examples
        --------
        ::

            reader = SAM.HDF5reader('results.h5')
            arr = reader.load_to_numpy('group/dataset')
            t   = reader.load_to_tensor('group/dataset')
        """

        def __init__(self, file_path: str, verbose: bool = False):
            self.file_path = file_path
            self.labels = []
            self._explore(verbose)

            def _filter(lst, char="/"):
                return [x for x in lst if char in x]

            self.labels = _filter(self.labels)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

        def _explore(self, verbose: bool):
            with h5py.File(self.file_path, 'r') as f:
                def collect(name, obj):
                    self.labels.append(name)
                    if verbose:
                        print(name)
                    for k, v in obj.attrs.items():
                        if verbose:
                            print(f"    {k}: {v}")
                f.visititems(collect)

        def print_keys(self):
            """Print all dataset paths in the file."""
            with h5py.File(self.file_path, 'r') as f:
                def printname(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(name)
                f.visititems(printname)

        def load_to_numpy(self, key: str, show_data: bool = False) -> np.ndarray:
            """
            Load a dataset as a float32 numpy array.

            Parameters
            ----------
            key : str
                HDF5 dataset path (e.g. 'group/subgroup/dataset').
            show_data : bool
                If True, prints the array shape.
            """
            with h5py.File(self.file_path, 'r') as f:
                arr = np.array(f[key][:], dtype=np.float32)
            if show_data:
                print(f"Dataset '{key}': shape={arr.shape}")
            return arr

        def load_to_tensor(self, key: str, show_data: bool = False) -> torch.Tensor:
            """
            Load a dataset as a float32 torch tensor.

            Parameters
            ----------
            key : str
                HDF5 dataset path.
            show_data : bool
                If True, prints the tensor shape.
            """
            with h5py.File(self.file_path, 'r') as f:
                t = torch.tensor(f[key][:], dtype=torch.float32)
            if show_data:
                print(f"Dataset '{key}': shape={t.shape}")
            return t

    # =========================================================================
    # BACKPACK
    # =========================================================================
    class Backpack:

        @staticmethod
        def find_files(
            path: str,
            file_end: str,
            infile: Union[str, None] = None,
            notinfile: Union[str, None] = None,
            verbose: bool = True,
        ) -> list:
            """
            Return a sorted list of files in *path* whose names end with
            *file_end*.

            Parameters
            ----------
            path : str
                Directory to search.
            file_end : str
                Required filename suffix.
            infile : str or None
                If given, only include files that contain this substring.
            notinfile : str or None
                If given, exclude files that contain this substring.
            verbose : bool
                If True, warn when no files are found.

            Returns
            -------
            list[str]: Sorted absolute file paths.

            Examples
            --------
            ::

                h5_files = SAM.Backpack.find_files('/data/sim', '.h5')
                surface_files = SAM.Backpack.find_files(
                    '/data/sim', '.vtu', infile='surface'
                )
            """
            files = sorted([
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(file_end)
                and (infile is None or infile in f)
                and (notinfile is None or notinfile not in f)
            ])
            if not files and verbose:
                print(
                    f"WARNING: No files found in '{path}' "
                    f"with ending '{file_end}'."
                )
            return files

        @staticmethod
        def read_cfd_times(case_path: str, verbose: bool = True) -> dict:
            """
            Extract wall-clock timing information from a CODA ``-out.txt``
            file.

            Parameters
            ----------
            case_path : str
                Simulation folder path.
            verbose : bool
                Warn if the file is missing or ambiguous.

            Returns
            -------
            dict with keys:
                'start_time', 'end_time', 'total_duration',
                'stage_times_hours', 'stage_total_hours'.
            Returns None if no file is found or multiple files exist.

            Examples
            --------
            ::

                info = SAM.Backpack.read_cfd_times('/data/sim/aoa_3.0_mach_0.75')
                print(info['stage_total_hours'])
            """
            files = SAM.Backpack.find_files(case_path, "-out.txt", verbose=False)
            if not files:
                if verbose:
                    print(f"WARNING: No -out.txt file in {case_path}.")
                return None
            if len(files) > 1:
                if verbose:
                    print(f"WARNING: Multiple -out.txt files in {case_path}.")
                return None

            with open(files[0], 'r') as fh:
                content = fh.read()

            date_pat = r'\w{3} +(\w{3} +\d{1,2} +\d{2}:\d{2}:\d{2}) +CEST +(\d{4})'
            matches = re.findall(date_pat, content)
            if len(matches) < 2:
                raise ValueError("Insufficient time stamps found.")

            fmt = "%b %d %H:%M:%S %Y"
            start_time = datetime.strptime(f"{matches[0][0]} {matches[0][1]}", fmt)
            end_time   = datetime.strptime(f"{matches[-1][0]} {matches[-1][1]}", fmt)

            stage_pat = (
                r'TimeIntegration::Iterate\(\)\s+([\d.]+)\s+\[(h|min|days)\] '
                r'\(wall clock time\)'
            )
            stage_hours = []
            for value, unit in re.findall(stage_pat, content):
                t = float(value)
                if unit == 'min':
                    t /= 60.0
                elif unit == 'days':
                    t *= 24.0
                stage_hours.append(t)

            return {
                'start_time':       start_time,
                'end_time':         end_time,
                'total_duration':   end_time - start_time,
                'stage_times_hours': stage_hours,
                'stage_total_hours': sum(stage_hours),
            }

        @staticmethod
        def same_columns(
            array: np.ndarray,
            atol: float = 1e-6,
            rtol: float = 1e-5,
        ) -> bool:
            """
            Check whether all "slices" along the first axis of *array* are
            numerically identical.

            Parameters
            ----------
            array : np.ndarray, shape (n_cases, n_points, n_dim)
                Stack of arrays to compare.
            atol : float
                Absolute tolerance. Default 1e-6.
            rtol : float
                Relative tolerance. Default 1e-5.

            Returns
            -------
            bool: True if all slices agree within tolerance.

            Examples
            --------
            ::

                stack = np.stack([mesh_a, mesh_b], axis=0)
                assert SAM.Backpack.same_columns(stack)
            """
            base = array[0]
            equal = True
            for i in range(1, array.shape[0]):
                if not np.allclose(base, array[i], atol=atol, rtol=rtol):
                    equal = False
                    diff_abs = np.abs(base - array[i])
                    diff_rel = diff_abs / (np.abs(base) + rtol)
                    n_diff = np.count_nonzero(
                        np.any((diff_abs > atol) & (diff_rel > rtol), axis=1)
                    )
                    print(
                        f"\nSlice {i} differs from slice 0 in {n_diff} points."
                    )
            return equal

        @staticmethod
        def get_unified_connectivity(mesh: pv.UnstructuredGrid) -> np.ndarray:
            """
            Build a padded connectivity array from a PyVista mesh that may
            contain mixed element types.

            Cells with fewer nodes than the maximum are padded with ``-1``.

            Parameters
            ----------
            mesh : pv.UnstructuredGrid

            Returns
            -------
            np.ndarray, shape (n_cells, max_nodes_per_cell)

            Examples
            --------
            ::

                conec = SAM.Backpack.get_unified_connectivity(mesh)
            """
            cell_dict = mesh.cells_dict
            max_nodes = max(arr.shape[1] for arr in cell_dict.values())
            total_cells = sum(arr.shape[0] for arr in cell_dict.values())
            connectivity = np.full((total_cells, max_nodes), -1, dtype=int)
            start = 0
            for _, cells in cell_dict.items():
                n = cells.shape[0]
                connectivity[start:start + n, :cells.shape[1]] = cells
                start += n
            return connectivity

        @staticmethod
        def ensure_cell_data(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
            """
            Convert all point_data arrays to cell_data if the mesh only has
            point data.

            Parameters
            ----------
            mesh : pv.UnstructuredGrid

            Returns
            -------
            pv.UnstructuredGrid: Mesh with cell_data guaranteed.
            """
            if mesh.point_data:
                converted = mesh.point_data_to_cell_data()
                for name, arr in converted.cell_data.items():
                    if name not in mesh.cell_data:
                        mesh.cell_data[name] = arr
            return mesh

        @staticmethod
        def create_tensors_from_h5(file_path: str, stage: int = 0) -> dict:
            """
            Load coordinates, sorting indices, flight conditions and variables
            from a FRODO-format HDF5 file.

            Parameters
            ----------
            file_path : str
                Path to the .h5 file.
            stage : int
                Stage number to load variables from. Default 0.

            Returns
            -------
            dict keyed by CADGroup name, each value containing 'Coord',
            'idx_sort', 'flcc', 'Vars'.
            """
            results = {}
            with h5py.File(file_path, 'r') as hf:
                for cad_key in hf:
                    if cad_key == "sim_metadata":
                        continue
                    cg = hf[cad_key]
                    mesh = cg["Mesh"]
                    vars_dict = {
                        vname: cg["Vars"][str(stage)][vname][()]
                        for vname in cg["Vars"][str(stage)]
                    } if "Vars" in cg else {}
                    results[cad_key] = {
                        "Coord":    mesh["Coord"][()],
                        "idx_sort": mesh["idx_sort"][()],
                        "flcc":     cg["FlCc"][()],
                        "Vars":     vars_dict,
                    }
            return results

        @staticmethod
        def get_df_from_csv(files_list: list) -> pd.DataFrame:
            """
            Read one or more CODA-format CSV monitor files into a single
            DataFrame.

            The files must use the CODA two-line header format (comment line
            followed by quoted column names).

            Parameters
            ----------
            files_list : list[str]
                Paths to the CSV files.

            Returns
            -------
            pd.DataFrame: Concatenated data with an additional 'total_iter'
                counter column prepended.

            Examples
            --------
            ::

                df = SAM.Backpack.get_df_from_csv([
                    'output_0__monitors_TimeIntegration.dat'
                ])
            """
            dfs = []
            for file_path in files_list:
                with open(file_path, 'r') as fh:
                    fh.readline()
                    header_line = fh.readline()
                col_names = [
                    n.replace('"', '')
                    for n in re.findall(r'"(.*?")', header_line)
                ]
                df = pd.read_csv(
                    file_path,
                    delim_whitespace=True,
                    skiprows=2,
                    names=col_names,
                    dtype=np.float64,
                )
                dfs.append(df)

            df = pd.concat(dfs, ignore_index=True)
            df_iter = pd.DataFrame(
                {"total_iter": np.arange(df.shape[0])}
            )
            return pd.concat([df_iter, df], axis=1)

        @staticmethod
        def folder_fmt_to_pattern(folder_fmt: str) -> re.Pattern:
            """
            Convert a folder format string such as ``'aoa_{}_mach_{}'`` into
            a compiled regex that matches corresponding folder names.

            ``{}`` placeholders are replaced with ``[-\\d\\.]+`` to capture
            any signed decimal number.

            Parameters
            ----------
            folder_fmt : str
                Format string where ``{}`` denotes a numeric placeholder.

            Returns
            -------
            re.Pattern

            Examples
            --------
            ::

                pat = SAM.Backpack.folder_fmt_to_pattern('aoa_{}_mach_{}')
                assert pat.match('aoa_-3.50_mach_0.750')
            """
            parts = folder_fmt.split("_")
            regex_parts = [
                r"[-\d\.]+" if "{" in p and "}" in p else re.escape(p)
                for p in parts
            ]
            return re.compile(rf"^{'_'.join(regex_parts)}$")

        @staticmethod
        def _fornberg_weights(x, x0, m):
            """
            Compute finite-difference weights using Fornberg's algorithm.

            Parameters
            ----------
            x : (n,) tensor
                Stencil coordinates.
            x0 : float
                Evaluation point.
            m : int
                Maximum derivative order.

            Returns
            -------
            c : tensor (m+1,n)
                c[k,j] are weights for the k-th derivative.
            """

            n = len(x)
            c = torch.zeros((m + 1, n), dtype=torch.float64, device=x.device)

            c1 = 1.0
            c4 = x[0] - x0
            c[0, 0] = 1.0

            for i in range(1, n):

                mn = min(i, m)

                c2 = 1.0
                c5 = c4
                c4 = x[i] - x0

                for j in range(i):

                    c3 = x[i] - x[j]
                    c2 *= c3

                    if j == i - 1:

                        for k in range(mn, 0, -1):

                            c[k, i] = (
                                c1
                                * (
                                    k * c[k - 1, i - 1]
                                    - c5 * c[k, i - 1]
                                )
                                / c2
                            )

                        c[0, i] = -c1 * c5 * c[0, i - 1] / c2

                    for k in range(mn, 0, -1):

                        c[k, j] = (
                            (c4 * c[k, j] - k * c[k - 1, j])
                            / c3
                        )

                    c[0, j] = c4 * c[0, j] / c3

                c1 = c2

            return c
        
    # =========================================================================
    # WEAPONS
    # =========================================================================
    class Weapons:

        @staticmethod
        def sort_by_centroid(points: np.ndarray):
            """
            Sort an unordered point cloud by polar angle around its centroid,
            computed via PCA on the 2-D principal plane.

            Falls back to lexsort when the second principal component is
            negligible (degenerate / collinear point sets).

            Parameters
            ----------
            points : np.ndarray, shape (N, D)

            Returns
            -------
            sorted_points : np.ndarray, shape (N, D)
            order : np.ndarray[int32], shape (N,)

            Examples
            --------
            ::

                pts_sorted, idx = SAM.Weapons.sort_by_centroid(pts)
            """
            centroid = points.mean(axis=0)
            shifted  = points - centroid
            N, D = shifted.shape

            try:
                _, s, vt = np.linalg.svd(shifted, full_matrices=False)
            except np.linalg.LinAlgError:
                idx = np.lexsort(
                    tuple(points[:, i] for i in reversed(range(D)))
                )
                return points[idx], idx.astype(np.int32)

            if s.size < 2 or s[1] < 1e-8 * max(s[0], 1.0):
                idx = np.lexsort(
                    tuple(points[:, i] for i in reversed(range(D)))
                )
                return points[idx], idx.astype(np.int32)

            proj = shifted @ vt[:2].T
            idx  = np.argsort(np.arctan2(proj[:, 1], proj[:, 0]))
            return points[idx], idx.astype(np.int32)

        @staticmethod
        def sort_lexsort(points: np.ndarray):
            """
            Sort points lexicographically (last column is the primary key).

            Parameters
            ----------
            points : np.ndarray, shape (N, D)

            Returns
            -------
            sorted_points : np.ndarray, shape (N, D)
            order : np.ndarray[int], shape (N,)
            """
            idx = np.lexsort(
                tuple(points[:, i] for i in reversed(range(points.shape[1])))
            )
            return points[idx], idx

        @staticmethod
        def sort_closed_curve_by_kdtree(
            points: np.ndarray,
            k: int = 10,
            start_index: int = None,
            alpha: float = 0.7,
        ):
            """
            Order points sampled on a closed 1-D curve (2-D or 3-D) using a
            nearest-neighbour chain with smoothed tangent guidance.

            Parameters
            ----------
            points : np.ndarray, shape (N, D)
            k : int
                Number of nearest neighbours. Recommended 8–12. Default 10.
            start_index : int or None
                Starting point index. If None, the point most isolated from
                its neighbours is used.
            alpha : float
                Tangent smoothing weight in [0, 1]. Higher values preserve
                direction more strongly. Default 0.7.

            Returns
            -------
            ordered_points : np.ndarray, shape (N, D)
            order : np.ndarray[int], shape (N,)

            Examples
            --------
            ::

                pts_ord, idx = SAM.Weapons.sort_closed_curve_by_kdtree(
                    airfoil_pts, k=10, alpha=0.8
                )
            """
            from scipy.spatial import KDTree

            points = np.asarray(points)
            N = len(points)
            if N < 3:
                return points.copy(), np.arange(N)

            tree = KDTree(points)
            _, neighbors = tree.query(points, k=min(k, N))

            if start_index is None:
                mean_dist = np.mean(
                    np.linalg.norm(
                        points[neighbors[:, 1:]] - points[:, None], axis=2
                    ),
                    axis=1,
                )
                current = int(np.argmax(mean_dist))
            else:
                current = int(start_index)

            visited = np.zeros(N, dtype=bool)
            order   = np.empty(N, dtype=int)

            def unit(v):
                n = np.linalg.norm(v)
                return v / n if n > 0 else v

            visited[current] = True
            order[0] = current

            nxt = min(
                neighbors[current][1:],
                key=lambda j: np.linalg.norm(points[j] - points[current]),
            )
            prev_dir = unit(points[nxt] - points[current])
            current  = nxt
            visited[current] = True
            order[1] = current

            for i in range(2, N):
                candidates = [
                    j for j in neighbors[current]
                    if not visited[j] and j != current
                ]
                if not candidates:
                    remaining = np.where(~visited)[0]
                    dists = np.linalg.norm(
                        points[remaining] - points[current], axis=1
                    )
                    nxt = remaining[np.argmin(dists)]
                else:
                    def cost(j):
                        v = unit(points[j] - points[current])
                        dot = np.dot(v, prev_dir)
                        if dot < 0.0:
                            return np.inf
                        return (
                            np.arccos(np.clip(dot, -1.0, 1.0))
                            + 0.2 * np.linalg.norm(points[j] - points[current])
                        )
                    nxt = min(candidates, key=cost)

                new_dir  = unit(points[nxt] - points[current])
                prev_dir = unit(alpha * prev_dir + (1 - alpha) * new_dir)
                current  = nxt
                visited[current] = True
                order[i] = current

            return points[order], order

        @staticmethod
        def sort_points_by_hull_projection(points: np.ndarray, alpha: float = 1.5):
            """
            Order all points by projecting them onto the concave hull of the
            point cloud and sorting by curvilinear abscissa.

            Parameters
            ----------
            points : np.ndarray, shape (N, 2)
            alpha : float
                Circumradius threshold for the concave-hull algorithm.
                Smaller values produce tighter hulls. Default 1.5.

            Returns
            -------
            ordered_points : np.ndarray, shape (N, 2)
            order : np.ndarray[int], shape (N,)
            """
            def _concave_hull_indices(pts: np.ndarray, a: float) -> np.ndarray:
                tri = Delaunay(pts)
                edge_count = {}
                for simplex in tri.simplices:
                    for i, j in [(0, 1), (1, 2), (2, 0)]:
                        p1 = pts[simplex[i]]
                        p2 = pts[simplex[j]]
                        p3 = pts[simplex[3 - i - j]]
                        ab = np.linalg.norm(p1 - p2)
                        bc = np.linalg.norm(p2 - p3)
                        ca = np.linalg.norm(p3 - p1)
                        s = (ab + bc + ca) / 2.0
                        area = max(s * (s - ab) * (s - bc) * (s - ca), 0.0)
                        if area == 0.0:
                            continue
                        R = ab * bc * ca / (4.0 * np.sqrt(area))
                        if R < a:
                            edge = tuple(sorted((simplex[i], simplex[j])))
                            edge_count[edge] = edge_count.get(edge, 0) + 1

                bnd = [e for e, c in edge_count.items() if c == 1]
                adj = {}
                for i, j in bnd:
                    adj.setdefault(i, []).append(j)
                    adj.setdefault(j, []).append(i)

                start = min(adj, key=lambda i: pts[i, 0])
                path, prev, cur = [start], None, start
                while True:
                    nbrs = adj[cur]
                    nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
                    if nxt == start:
                        break
                    path.append(nxt)
                    prev, cur = cur, nxt
                path.append(start)
                return np.array(path)

            hull_idx = _concave_hull_indices(points, alpha)
            hull_pts = points[hull_idx]
            diffs = np.diff(hull_pts, axis=0)
            seg_len = np.linalg.norm(diffs, axis=1)
            s_hull = np.concatenate(([0.0], np.cumsum(seg_len)))

            s_proj = np.zeros(len(points))
            for i, p in enumerate(points):
                best_d, best_s = np.inf, 0.0
                for j in range(len(hull_pts) - 1):
                    a, b = hull_pts[j], hull_pts[j + 1]
                    ab = b - a
                    denom = np.dot(ab, ab)
                    if denom == 0:
                        continue
                    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
                    d = np.linalg.norm(p - (a + t * ab))
                    if d < best_d:
                        best_d = d
                        best_s = s_hull[j] + t * seg_len[j]
                s_proj[i] = best_s

            order = np.argsort(s_proj)
            return points[order], order

        @staticmethod
        def surface_derivative(
            X,
            f,
            order: int = 1,
            min_ds: float = 1e-6,
            stencil_width: int = 1,
            poly_order: int = None,
        ):
            """
            Compute first- or second-order derivatives of a scalar field
            with respect to the arc length of a discrete curve or surface
            contour.

            Two differentiation modes are available based on ``stencil_width``:

            * ``stencil_width == 1``: standard finite-difference gradients
              via ``numpy.gradient`` / ``torch.gradient`` (O(h²) central).
            * ``stencil_width > 1``: local polynomial regression (Savitzky-
              Golay style) evaluated analytically at the local centre, which
              reduces sensitivity to noise and handles irregular spacing.

            Parameters
            ----------
            X : np.ndarray or torch.Tensor, shape (N, D)
                Ordered point coordinates defining the curve.
            f : np.ndarray or torch.Tensor
                Scalar field evaluated at the curve points.
                * 1-D ``(N,)``        – single snapshot, returns 1-D derivative.
                * 2-D ``(N, ncases)`` – multiple snapshots, returns 2-D derivative.
            order : int
                Derivative order.  1 = first, 2 = second. Default 1.
            min_ds : float
                Minimum arc-length increment; points closer than this are
                removed to avoid division-by-zero. Default 1e-6.
            stencil_width : int
                Half-width of the local polynomial neighbourhood.
                ``stencil_width=1`` activates finite differences.
                Default 1.
            poly_order : int or None
                Degree of the local polynomial (only used when
                ``stencil_width > 1``).  Must satisfy
                ``order ≤ poly_order ≤ 2 × stencil_width``.
                If None, set to ``min(max(order, 2), 2×stencil_width)``.

            Returns
            -------
            np.ndarray or torch.Tensor matching the dtype and device of *f*.

            Examples
            --------
            ::

                # First derivative of Cp along arc length (numpy)
                dCp_ds = SAM.Weapons.surface_derivative(coords, Cp)

                # Second derivative with local polynomial smoothing
                d2Cp_ds2 = SAM.Weapons.surface_derivative(
                    coords, Cp, order=2, stencil_width=5, poly_order=3
                )
            """
            max_poly = 2 * stencil_width
            min_poly = max(order, 1)

            if poly_order is None:
                poly_order = min(max(min_poly, 2), max_poly)
            else:
                if poly_order < min_poly:
                    raise ValueError(
                        f"poly_order={poly_order} must be >= order={order}."
                    )
                if poly_order > max_poly:
                    raise ValueError(
                        f"poly_order={poly_order} exceeds maximum "
                        f"{max_poly} for stencil_width={stencil_width}."
                    )

            use_ls = stencil_width > 1

            def _arc(Xa, is_torch):
                if is_torch:
                    dX = Xa[1:] - Xa[:-1]
                    ds = torch.sqrt(torch.sum(dX**2, dim=1))
                    ds = torch.clamp(ds, min=min_ds)

                    s = torch.zeros(
                        Xa.shape[0],
                        dtype=torch.float64,
                        device=Xa.device,
                    )
                    s[1:] = torch.cumsum(ds, dim=0)

                else:
                    dX = Xa[1:] - Xa[:-1]
                    ds = np.sqrt(np.sum(dX**2, axis=1))
                    ds = np.maximum(ds, min_ds)

                    s = np.zeros(Xa.shape[0], dtype=np.float64)
                    s[1:] = np.cumsum(ds)

                return s

            def _ls_np(s_arr, f_col, sw, pord, dord):

                N = len(s_arr)
                out = np.zeros(N)

                for i in range(N):

                    lo = max(0, i - sw)
                    hi = min(N - 1, i + sw)

                    s_loc = s_arr[lo:hi + 1] - s_arr[i]
                    y = f_col[lo:hi + 1]

                    # Número de puntos disponibles
                    npts = len(s_loc)

                    # Reducir automáticamente el grado si hace falta
                    degree = min(pord, npts - 1)

                    # Escalado local para mejorar el condicionamiento
                    scale = np.max(np.abs(s_loc))

                    if scale < min_ds:
                        scale = 1.0

                    s_scaled = s_loc / scale

                    coeffs = np.polyfit(s_scaled, y, degree)

                    p = np.poly1d(coeffs)

                    out[i] = p.deriv(dord)(0.0) / scale**dord

                return out

            def _ls_torch(s_arr, f_col, sw, pord, dord):
                return torch.tensor(
                    _ls_np(s_arr.cpu().numpy(), f_col.cpu().numpy(), sw, pord, dord),
                    dtype=torch.float64, device=s_arr.device,
                )

            is_torch = torch.is_tensor(X)

            if is_torch:
                X = X.to(torch.float64)
                f = f.to(torch.float64)
                squeeze = f.ndim == 1
                if squeeze:
                    f = f[:, None]
                
                s = _arc(X, True)

                if not use_ls:
                    df = torch.gradient(f, spacing=(s,), dim=0)[0]
                    if order == 1:
                        return df.squeeze() if squeeze else df
                    d2f = torch.gradient(df, spacing=(s,), dim=0)[0]
                    return d2f.squeeze() if squeeze else d2f
                else:
                    nc = f.shape[1]
                    fn = _ls_torch
                    if order == 1:
                        out = torch.stack(
                            [fn(s, f[:, c], stencil_width, poly_order, 1)
                             for c in range(nc)], dim=1
                        )
                    else:
                        out = torch.stack(
                            [fn(s, f[:, c], stencil_width, poly_order, 2)
                             for c in range(nc)], dim=1
                        )
                    return out.squeeze() if squeeze else out

            else:
                X = np.asarray(X, dtype=np.float64)
                f = np.asarray(f, dtype=np.float64)
                squeeze = f.ndim == 1
                if squeeze:
                    f = f[:, None]
                
                s = _arc(X, False)

                if not use_ls:
                    df = np.gradient(f, s, axis=0)
                    if order == 1:
                        return df.squeeze() if squeeze else df
                    d2f = np.gradient(df, s, axis=0)
                    return d2f.squeeze() if squeeze else d2f
                else:
                    nc = f.shape[1]
                    if order == 1:
                        out = np.stack(
                            [_ls_np(s, f[:, c], stencil_width, poly_order, 1)
                             for c in range(nc)], axis=1
                        )
                    else:
                        out = np.stack(
                            [_ls_np(s, f[:, c], stencil_width, poly_order, 2)
                             for c in range(nc)], axis=1
                        )
                    return out.squeeze() if squeeze else out

        @staticmethod
        def finite_diff_derivative(
            X: torch.Tensor,
            f: torch.Tensor,
            order: int = 1,
        ) -> torch.Tensor:
            """
            Compute the *n*-th order derivative of *f* with respect to each
            spatial dimension in *X* using finite differences.

            Interior points use second-order central differences; boundary
            points use second-order one-sided formulas.

            Parameters
            ----------
            X : torch.Tensor, shape (N, D)
                Independent variable coordinates.
            f : torch.Tensor, shape (N,)
                Scalar field values.
            order : int
                Derivative order (≥ 1). Default 1.

            Returns
            -------
            torch.Tensor, shape (N, D)
                Derivatives with respect to each coordinate dimension.

            Examples
            --------
            ::

                grad = SAM.Weapons.finite_diff_derivative(coords, cp, order=1)
            """
            if order < 1:
                raise ValueError("order must be >= 1.")

            X = X.to(torch.float64)
            f = f.to(torch.float64)
            N, D = X.shape
            derivs = torch.zeros((N, D), dtype=torch.float64, device=X.device)
            eps = 1e-14

            for d in range(D):
                xd = X[:, d]
                g  = f.clone()
                for _ in range(order):
                    new_g = torch.zeros_like(g)
                    dx = xd[2:] - xd[:-2]
                    dx = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
                    new_g[1:-1] = (g[2:] - g[:-2]) / dx

                    # Second-order one-sided at left boundary
                    h0 = (xd[1] - xd[0]).clamp(min=eps)
                    h1 = (xd[2] - xd[0]).clamp(min=eps)
                    new_g[0] = (
                        -g[2] * h0 ** 2
                        + g[1] * h1 ** 2
                        - g[0] * (h1 ** 2 - h0 ** 2)
                    ) / (h0 * h1 * (h1 - h0))

                    # Second-order one-sided at right boundary
                    hm1 = (xd[-1] - xd[-2]).clamp(min=eps)
                    hm2 = (xd[-1] - xd[-3]).clamp(min=eps)
                    new_g[-1] = (
                        g[-3] * hm1 ** 2
                        - g[-2] * hm2 ** 2
                        + g[-1] * (hm2 ** 2 - hm1 ** 2)
                    ) / (hm1 * hm2 * (hm2 - hm1))

                    g = new_g
                derivs[:, d] = g

            return derivs

        @staticmethod
        def finite_diff_derivative_Fornberg(
            X,
            f,
            order=1,
            stencil_width=2,
        ):
            """
            Finite differences on arbitrarily spaced nodes using
            Fornberg weights.

            Parameters
            ----------
            X : (N,D)
            f : (N,)
            order : int
            stencil_width : int

                Number of neighbours on each side.

                stencil_width=2

                -> 5-point stencil

                stencil_width=3

                -> 7-point stencil

            """

            X = X.to(torch.float64)
            f = f.to(torch.float64)

            N, D = X.shape

            deriv = torch.zeros((N, D),
                                dtype=torch.float64,
                                device=X.device)

            for d in range(D):

                xd = X[:, d]

                for i in range(N):

                    lo = max(0, i - stencil_width)
                    hi = min(N, i + stencil_width + 1)

                    # if close to boundary enlarge on opposite side

                    if hi - lo < 2 * stencil_width + 1:

                        if lo == 0:
                            hi = min(N, 2 * stencil_width + 1)

                        if hi == N:
                            lo = max(0, N - (2 * stencil_width + 1))

                    xs = xd[lo:hi]

                    weights = SAM.Backpack._fornberg_weights(xs, xd[i], order)

                    deriv[i, d] = torch.dot(
                        weights[order],
                        f[lo:hi],
                    )

            return deriv

        @staticmethod
        def build_element_neighbors(
            connectivity: torch.Tensor,
        ) -> torch.Tensor:
            """
            Build a face-neighbour table for a triangular mesh.

            Parameters
            ----------
            connectivity : torch.Tensor, shape (Ne, 3)
                Triangle vertex indices.

            Returns
            -------
            torch.Tensor, shape (Ne, 3)
                ``neighbors[e, f]`` is the index of the element sharing face
                *f* with element *e*, or ``-1`` for boundary faces.
            """
            Ne = connectivity.shape[0]
            neighbors = torch.full((Ne, 3), -1, dtype=torch.long)

            edges = torch.stack([
                connectivity[:, [0, 1]],
                connectivity[:, [1, 2]],
                connectivity[:, [2, 0]],
            ], dim=1)
            edges_sorted, _ = edges.sort(dim=2)
            edges_flat = edges_sorted.view(-1, 2)
            elem_ids = torch.arange(Ne).repeat_interleave(3)
            face_ids = torch.arange(3).repeat(Ne)

            edge_dict = {}
            for e, f, edge in zip(
                elem_ids.tolist(), face_ids.tolist(), edges_flat.tolist()
            ):
                key = tuple(edge)
                if key in edge_dict:
                    e2, f2 = edge_dict[key]
                    neighbors[e, f] = e2
                    neighbors[e2, f2] = e
                else:
                    edge_dict[key] = (e, f)

            return neighbors

        @staticmethod
        def compute_3dgrad_greengauss(
            nodes: torch.Tensor,
            connectivity: torch.Tensor,
            neighbors: torch.Tensor,
            tensor_out: torch.Tensor,
            var_index: int = 0,
            chunk_size: int = 8,
            export_vtk: bool = False,
            vtk_filename: str = "grad.vtu",
            device: torch.device = None,
        ) -> torch.Tensor:
            """
            Compute the Green-Gauss gradient of a scalar field defined at
            triangle cell centres.

            Parameters
            ----------
            nodes : torch.Tensor, shape (Nv, 3)
                Vertex coordinates.
            connectivity : torch.Tensor, shape (Ne, 3)
                Triangle connectivity.
            neighbors : torch.Tensor, shape (Ne, 3)
                Neighbour table from ``build_element_neighbors``.
            tensor_out : torch.Tensor, shape (Ne, Nc, n_vars)
                Field data at element centres.
            var_index : int
                Which variable channel to differentiate. Default 0.
            chunk_size : int
                Cases processed per batch (memory/speed trade-off). Default 8.
            export_vtk : bool
                If True, writes a ``<vtk_filename>`` VTK file with the field
                and its gradient.
            vtk_filename : str
                Output VTK filename. Default 'grad.vtu'.
            device : torch.device or None
                Compute device. Defaults to tensor_out's device.

            Returns
            -------
            torch.Tensor, shape (Ne, Nc, 3): gradient vector per element
                per case.
            """
            if device is None:
                device = tensor_out.device

            dtype = torch.float32
            nodes        = nodes.to(device=device, dtype=dtype)
            connectivity = connectivity.to(device=device).long()
            neighbors    = neighbors.to(device=device)
            tensor_out   = tensor_out.to(device=device, dtype=dtype)

            cp_elem = tensor_out[:, :, var_index]
            Ne, Nc  = cp_elem.shape

            tri = nodes[connectivity]
            v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
            normal_elem = torch.cross(v1 - v0, v2 - v0, dim=1)
            normal_elem = normal_elem / torch.linalg.norm(
                normal_elem, dim=1, keepdim=True
            )

            edges = [(v0, v1), (v1, v2), (v2, v0)]
            face_normals, face_lengths = [], []
            for a, b in edges:
                e = b - a
                L = torch.linalg.norm(e, dim=1)
                face_normals.append(torch.cross(normal_elem, e, dim=1))
                face_lengths.append(L)

            area = 0.5 * torch.linalg.norm(
                torch.cross(v1 - v0, v2 - v0, dim=1), dim=1
            )

            grad_chunks = []
            for c0 in range(0, Nc, chunk_size):
                c1   = min(c0 + chunk_size, Nc)
                grad = torch.zeros((Ne, c1 - c0, 3), device=device, dtype=dtype)
                cp_c = cp_elem[:, c0:c1]

                for f_idx in range(3):
                    nb   = neighbors[:, f_idx]
                    cp_nb = torch.where(
                        nb[:, None] >= 0,
                        cp_elem[nb, c0:c1],
                        cp_c,
                    )
                    cp_face = 0.5 * (cp_c + cp_nb)
                    grad += (
                        cp_face[:, :, None]
                        * face_normals[f_idx][:, None, :]
                        * face_lengths[f_idx][:, None, None]
                    )

                grad /= area[:, None, None].clamp_min(1e-14)
                grad_chunks.append(grad.cpu())

            grad = torch.cat(grad_chunks, dim=1)

            if export_vtk:
                pts  = nodes.cpu().numpy()
                cells = np.hstack([
                    np.full((Ne, 1), 3), connectivity.cpu().numpy()
                ]).astype(np.int64)
                celltypes = np.full(Ne, pv.CellType.TRIANGLE, dtype=np.uint8)
                mesh = pv.UnstructuredGrid(cells, celltypes, pts)
                for k in range(cp_elem.shape[-1]):
                    mesh.cell_data[f"x_{k}"]      = cp_elem[:, k].cpu().numpy()
                    mesh.cell_data[f"grad_x_{k}"] = grad[:, k, :].numpy()
                mesh.save(vtk_filename)

            return grad

        @staticmethod
        def GMM(
            df_data: pd.DataFrame,
            BIC_study: bool = False,
            groupby: Union[str, list, tuple] = None,
            nclusters: int = 2,
            features: list = None,
            save_pictures: bool = True,
            folder_to_save: str = './GMM_study/',
            format_to_save: Literal['csv', 'hdf', 'pkl'] = 'csv',
            n_components_range: range = range(1, 7),
            random_state: int = 42,
            return_metrics_table: bool = False,
            plot_global_analysis: bool = True,
            verbose: bool = False,
            **kwargs,
        ) -> Union[pd.DataFrame, tuple]:
            """
            Apply Gaussian Mixture Model (GMM) clustering with optional
            BIC/AIC model-selection study.

            Parameters
            ----------
            df_data : pd.DataFrame
                Input data.  Must contain all columns listed in *features*.
            BIC_study : bool
                If True, fit GMMs for each number of components in
                *n_components_range* and record BIC/AIC scores.
            groupby : str, list or None
                Column(s) used to split df_data into independent groups.
                Each group is clustered independently. If None, the entire
                DataFrame is treated as one group.
            nclusters : int
                Number of mixture components for the final model. Default 2.
            features : list[str]
                Feature columns used for clustering. Mandatory.
            save_pictures : bool
                Save BIC curves and scatter plots to *folder_to_save*.
            folder_to_save : str
                Output directory. Default './GMM_study/'.
            format_to_save : str
                Format for saving df_result: 'csv', 'hdf' or 'pkl'.
            n_components_range : range
                Range of component counts for BIC sweep. Default range(1, 7).
            random_state : int
                Random seed for reproducibility. Default 42.
            return_metrics_table : bool
                If True, returns (df_result, df_metrics) instead of only
                df_result. df_metrics is empty when BIC_study=False.
            plot_global_analysis : bool
                Generate global BIC/AIC boxplots and heatmaps (requires
                BIC_study=True and groupby with two columns).
            verbose : bool
                Print progress information.

            Returns
            -------
            pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
                *df_data* augmented with a 'clusters_GMM' column, and
                optionally a metrics DataFrame.

            Examples
            --------
            ::

                df_clustered = SAM.Weapons.GMM(
                    df_data,
                    BIC_study=True,
                    groupby=['AoA', 'Mach'],
                    nclusters=3,
                    features=['x', 'z', 'cp'],
                    verbose=True,
                )
            """
            from sklearn.mixture import GaussianMixture
            from sklearn.decomposition import PCA

            if features is None:
                raise ValueError(
                    "The 'features' list (numerical columns for GMM) "
                    "must be specified."
                )

            if verbose:
                print('\n ── Starting GMM ──────────────────────────────────\n')
                print(f'  Features   : {features}')
                print(f'  N clusters : {nclusters}')
                print(f'  Output dir : {folder_to_save}\n')

            def fmt(x, _):
                return f"{x:.2f}"

            df_result = df_data.copy()
            df_result["clusters_GMM"] = -1

            if save_pictures:
                os.makedirs(
                    os.path.join(folder_to_save, 'pictures_case'), exist_ok=True
                )

            group_iter = (
                df_result.groupby(groupby)
                if groupby is not None
                else [(None, df_result)]
            )

            metrics_records = []

            for group_key, grp in group_iter:
                if verbose and hasattr(group_iter, "set_postfix"):
                    group_iter.set_postfix(case=group_key)

                X = StandardScaler().fit_transform(grp[features].values)

                if X.shape[0] < nclusters:
                    print(
                        f"[WARN] Group {group_key}: {X.shape[0]} samples "
                        f"< {nclusters} clusters. Skipping."
                    )
                    continue

                if BIC_study:
                    bics, aics = [], []
                    for n in n_components_range:
                        gmm_test = GaussianMixture(
                            n_components=n,
                            covariance_type=kwargs.get("covariance_type", "diag"),
                            max_iter=kwargs.get("max_iter", 200),
                            init_params=kwargs.get("init_params", "kmeans"),
                            reg_covar=kwargs.get("reg_covar", 1e-6),
                            random_state=random_state,
                        )
                        try:
                            gmm_test.fit(X)
                            bic_val = gmm_test.bic(X)
                            aic_val = gmm_test.aic(X)
                        except ValueError:
                            bic_val = aic_val = np.inf
                        bics.append(bic_val)
                        aics.append(aic_val)
                        metrics_records.append({
                            "group": group_key,
                            "n_components": n,
                            "BIC": bic_val,
                            "AIC": aic_val,
                        })

                    best_bic_n = n_components_range[np.argmin(bics)]
                    best_aic_n = n_components_range[np.argmin(aics)]
                    df_result.loc[grp.index, "BIC_best_n"] = best_bic_n
                    df_result.loc[grp.index, "AIC_best_n"] = best_aic_n

                gmm_final = GaussianMixture(
                    n_components=nclusters,
                    covariance_type=kwargs.get("covariance_type", "diag"),
                    max_iter=kwargs.get("max_iter", 200),
                    init_params=kwargs.get("init_params", "kmeans"),
                    reg_covar=kwargs.get("reg_covar", 1e-6),
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items()
                       if k not in ("covariance_type", "max_iter",
                                    "init_params", "reg_covar")},
                )
                labels = gmm_final.fit_predict(X)
                df_result.loc[grp.index, "clusters_GMM"] = labels

                if BIC_study and save_pictures:
                    min_bic = min(bics)
                    best_n  = n_components_range[np.argmin(bics)]
                    df_result.loc[grp.index, "BIC_min"]   = min_bic
                    df_result.loc[grp.index, "BIC_opt_n"] = best_n

                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].plot(
                        list(n_components_range), bics,
                        marker="o", color="steelblue",
                    )
                    axes[0].set(
                        title=f"BIC ({group_key})" if group_key else "BIC",
                        xlabel="Components", ylabel="BIC",
                    )
                    axes[0].xaxis.set_major_formatter(
                        mticker.FuncFormatter(fmt)
                    )
                    axes[0].grid(True)

                    X_plot = X
                    if len(features) == 1:
                        xlabel = features[0]
                        ylabel = "Density"
                    elif len(features) == 2:
                        xlabel, ylabel = features[0], features[1]
                    elif len(features) > 2:
                        pca    = PCA(n_components=2)
                        X_plot = pca.fit_transform(X)
                        xlabel = (
                            f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
                        )
                        ylabel = (
                            f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
                        )

                    
                    if len(features) == 1:
                        plt.yticks([])
                        sc = axes[1].scatter(
                            X_plot[:, 0], np.zeros_like(X_plot[:, 0]),
                            c=labels, cmap="viridis", s=30, edgecolor="k",
                        )
                        plt.colorbar(sc, label="Cluster ID")
                        axes[1].set(xlabel=xlabel, ylabel=ylabel)
                        
                    elif len(features) > 1:
                        sc = plt.scatter(
                            X_plot[:, 0], X_plot[:, 1],
                            c=labels, cmap="viridis", s=30, edgecolor="k",
                        )
                        plt.colorbar(sc, label="Cluster ID")
                        plt.xlabel(xlabel)
                        plt.ylabel(ylabel)
                        
                    plt.grid(True)
                    group_str = (
                        "_".join(f"{x:.2f}" for x in group_key)
                        if group_key is not None else "global"
                    )
                    plt.title(f"GMM Clusters ({group_str})")
                    fig.suptitle(f"GMM Study — Group: {group_str}")
                    plt.savefig(
                        os.path.join(
                            folder_to_save, "pictures_case",
                            f"GMM_{group_str}.png",
                        ),
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close(fig)

            print("GMM clustering completed.")

            df_metrics = pd.DataFrame(metrics_records).replace(
                [np.inf, -np.inf], np.nan
            ) if metrics_records else pd.DataFrame()

            if (
                BIC_study
                and plot_global_analysis
                and not df_metrics.empty
            ):
                n_values = sorted(df_metrics["n_components"].unique())
                bic_data = [
                    df_metrics.loc[
                        df_metrics["n_components"] == n, "BIC"
                    ].values
                    for n in n_values
                ]
                aic_data = [
                    df_metrics.loc[
                        df_metrics["n_components"] == n, "AIC"
                    ].values
                    for n in n_values
                ]

                bic_means = np.array([np.mean(b) for b in bic_data])
                aic_means = np.array([np.mean(a) for a in aic_data])
                best_bic_n = n_values[np.argmin(bic_means)]
                best_aic_n = n_values[np.argmin(aic_means)]

                fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
                for ax, data, best, color, label in [
                    (axes[0], bic_data, best_bic_n, "steelblue", "BIC"),
                    (axes[1], aic_data, best_aic_n, "orange",    "AIC"),
                ]:
                    ax.boxplot(
                        data, positions=n_values, widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color="black", linewidth=2),
                    )
                    ax.axvline(
                        best, color="red", linestyle="--", linewidth=2,
                        label=f"Best mean {label} = {best}",
                    )
                    ax.legend()
                    ax.set(
                        title=f"{label} distribution vs components",
                        xlabel="Components", ylabel=label,
                    )
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))
                    ax.grid(True, linestyle="--", alpha=0.4)

                fig.suptitle("Global GMM model selection (BIC / AIC)", fontsize=13)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(folder_to_save, 'global_BIC_AIC_boxplot.png'),
                    dpi=150, bbox_inches="tight",
                )
                plt.show()

                df_summary = (
                    df_metrics.groupby("n_components")
                    .agg({
                        "BIC": ["mean", "median", "std", "min", "max"],
                        "AIC": ["mean", "median", "std", "min", "max"],
                    })
                    .reset_index()
                )
                df_summary.columns = [
                    "n_components",
                    "BIC_mean", "BIC_median", "BIC_std", "BIC_min", "BIC_max",
                    "AIC_mean", "AIC_median", "AIC_std", "AIC_min", "AIC_max",
                ]
                df_summary.to_csv(
                    os.path.join(folder_to_save, "GMM_BIC_AIC_summary.csv"),
                    sep=';', index=False,
                )

                plt.figure(figsize=(6, 4))
                sns.histplot(
                    df_metrics["BIC"], bins=30, kde=True,
                    color="steelblue", alpha=0.7,
                )
                plt.title("Global BIC distribution")
                plt.xlabel("BIC")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(folder_to_save, 'BIC_distribution.png')
                )
                plt.show()

                if (
                    groupby
                    and isinstance(groupby, (list, tuple))
                    and len(groupby) == 2
                ):
                    df_opt = (
                        df_result
                        .groupby(groupby)[["BIC_best_n"]]
                        .first()
                        .reset_index()
                        .pivot(
                            index=groupby[0],
                            columns=groupby[1],
                            values="BIC_best_n",
                        )
                    )
                    from matplotlib.patches import Patch
                    cluster_vals = np.sort(
                        df_opt.stack().dropna().unique()
                    )
                    boundaries = np.concatenate([
                        cluster_vals - 0.5, [cluster_vals[-1] + 0.5]
                    ])
                    cmap = plt.cm.get_cmap("tab10", len(cluster_vals))
                    norm = mcolors.BoundaryNorm(
                        boundaries=boundaries, ncolors=len(cluster_vals)
                    )

                    plt.figure(figsize=(9, 7))
                    step = max(1, len(df_opt.columns) // 15)
                    ax = sns.heatmap(
                        df_opt, annot=(df_opt.size <= 225), fmt=".0f",
                        cmap=cmap, norm=norm, cbar=False,
                        linewidths=0, annot_kws={"size": 8, "color": "black"},
                    )
                    ax.legend(
                        handles=[
                            Patch(facecolor=cmap(i), edgecolor="black",
                                  label=f"k = {int(k)}")
                            for i, k in enumerate(cluster_vals)
                        ],
                        title="Optimal clusters (BIC)",
                        loc="upper left", bbox_to_anchor=(1.02, 1),
                    )
                    ax.set_xticks(
                        np.arange(0, len(df_opt.columns), step) + 0.5
                    )
                    ax.set_xticklabels(
                        [f"{float(df_opt.columns[i]):.2f}"
                         for i in range(0, len(df_opt.columns), step)],
                        rotation=45, ha="right",
                    )
                    ax.set_yticks(
                        np.arange(0, len(df_opt.index), step) + 0.5
                    )
                    ax.set_yticklabels(
                        [f"{float(df_opt.index[i]):.2f}"
                         for i in range(0, len(df_opt.index), step)],
                        rotation=45, ha="right",
                    )
                    plt.title("Heatmap – Optimal number of clusters (BIC)")
                    plt.xlabel(groupby[1])
                    plt.ylabel(groupby[0])
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            folder_to_save, 'heatmap_optimal_clusters.png'
                        )
                    )
                    plt.show()

            # ── Save results ─────────────────────────────────────────────────
            if format_to_save == 'csv':
                df_result.to_csv(
                    os.path.join(folder_to_save, 'df_data_complete.csv'),
                    sep=';', index=True,
                )
            elif format_to_save == 'hdf':
                df_result.to_hdf(
                    os.path.join(folder_to_save, 'df_data_complete.h5'),
                    key=f'df_n_{nclusters}', mode='a', complevel=0, index=True,
                )
            elif format_to_save == 'pkl':
                df_result.to_pickle(
                    os.path.join(folder_to_save, 'df_data_complete.pkl')
                )
            else:
                print(
                    "DataFrame not saved: unsupported format_to_save value."
                )

            if return_metrics_table:
                if not BIC_study:
                    print(
                        "WARNING: df_metrics is empty because BIC_study=False."
                    )
                else:
                    df_metrics.to_csv(
                        os.path.join(folder_to_save, 'df_metrics.csv'),
                        sep=';', index=True,
                    )
                return df_result, df_metrics

            return df_result

        # ── Interpolation helpers ─────────────────────────────────────────────

        @staticmethod
        def _interpolate_idw(
            coord_src: np.ndarray,
            var_src: np.ndarray,
            coord_dst: np.ndarray,
            k: int = 4,
            eps: float = 1e-12,
        ) -> np.ndarray:
            """
            Inverse-distance weighting interpolation (builds KDTree internally).

            Parameters
            ----------
            coord_src : np.ndarray, shape (N_src, D)
            var_src   : np.ndarray, shape (N_src,) or (N_src, n_cols)
            coord_dst : np.ndarray, shape (N_dst, D)
            k : int
                Nearest neighbours. Default 4.
            eps : float
                Stability constant. Default 1e-12.

            Returns
            -------
            np.ndarray, shape (N_dst,) or (N_dst, n_cols)
            """
            from scipy.spatial import cKDTree
            tree = cKDTree(coord_src)
            dist, idx = tree.query(coord_dst, k=k)
            w = 1.0 / (dist + eps)
            w /= w.sum(axis=1, keepdims=True)
            if var_src.ndim == 1:
                return np.sum(w * var_src[idx], axis=1)
            return np.sum(w[..., None] * var_src[idx], axis=1)

        @staticmethod
        def _interpolate_idw_tree(
            tree,
            coord_src: np.ndarray,
            var_src: np.ndarray,
            coord_dst: np.ndarray,
            k: int = 4,
            eps: float = 1e-12,
        ) -> np.ndarray:
            """
            IDW interpolation using a pre-built cKDTree (avoids rebuilding).

            Parameters
            ----------
            tree : scipy.spatial.cKDTree
                KDTree built from coord_src.
            coord_src, var_src, coord_dst, k, eps : see ``_interpolate_idw``.
            """
            dist, idx = tree.query(coord_dst, k=k)
            w = 1.0 / (dist + eps)
            w /= w.sum(axis=1, keepdims=True)
            if var_src.ndim == 1:
                return np.sum(w * var_src[idx], axis=1)
            return np.sum(w[..., None] * var_src[idx], axis=1)

        @staticmethod
        def _interpolate_griddata(
            coord_src: np.ndarray,
            var_src: np.ndarray,
            coord_dst: np.ndarray,
            method: str = "linear",
        ) -> np.ndarray:
            """
            Scattered data interpolation via ``scipy.interpolate.griddata``.

            Parameters
            ----------
            coord_src : np.ndarray, shape (N_src, D)
            var_src   : np.ndarray, shape (N_src,) or (N_src, n_cols)
            coord_dst : np.ndarray, shape (N_dst, D)
            method : str
                Interpolation method passed to griddata.  Default 'linear'.
            """
            from scipy.interpolate import griddata
            return griddata(coord_src, var_src, coord_dst, method=method)

        @staticmethod
        def _build_pyvista_grid(
            coord: np.ndarray,
            conec: np.ndarray,
        ) -> pv.UnstructuredGrid:
            """
            Build a PyVista tetrahedral UnstructuredGrid.

            Parameters
            ----------
            coord : np.ndarray, shape (N, 3)
            conec : np.ndarray, shape (Ne, 4)  – tetrahedra connectivity

            Returns
            -------
            pv.UnstructuredGrid
            """
            n_cells = conec.shape[0]
            cells = np.hstack([
                np.full((n_cells, 1), 4), conec
            ]).astype(np.int64).ravel()
            celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
            return pv.UnstructuredGrid(cells, celltypes, coord)

        @staticmethod
        def _interpolate_pyvista(
            coord_src: np.ndarray,
            conec_src: np.ndarray,
            var_src_stack: np.ndarray,
            coord_dst: np.ndarray,
            conec_dst: np.ndarray,
        ) -> np.ndarray:
            """
            Mesh-to-mesh interpolation using PyVista's ``sample`` probe.

            Parameters
            ----------
            coord_src, conec_src : source mesh geometry.
            var_src_stack        : np.ndarray, shape (N_src, n_cols).
            coord_dst, conec_dst : target mesh geometry.

            Returns
            -------
            np.ndarray, shape (N_dst, n_cols)
            """
            mesh_src = SAM.Weapons._build_pyvista_grid(coord_src, conec_src)
            mesh_dst = SAM.Weapons._build_pyvista_grid(coord_dst, conec_dst)
            mesh_src.cell_data["values"] = var_src_stack
            return mesh_dst.sample(mesh_src).cell_data["values"]

    class DifferentialOperators:
        @staticmethod
        def _polynomial_basis(
            DX,
            poly_order: int = 2,
        ):
            """
            Parameters
            ----------
            DX : ndarray
                shape (nstencil, ndim)

            Returns
            -------
            A : ndarray
                shape (nstencil, nterms)
            """

            ndim = DX.shape[1]

            if poly_order not in (1, 2):
                raise ValueError(
                    "Only poly_order=1 or 2 supported."
                )

            terms = [np.ones(DX.shape[0])]

            terms.extend(
                DX[:, d]
                for d in range(ndim)
            )

            if poly_order == 2:

                terms.extend(
                    DX[:, d] ** 2
                    for d in range(ndim)
                )

                terms.extend(
                    DX[:, i] * DX[:, j]
                    for i in range(ndim)
                    for j in range(i + 1, ndim)
                )

            return np.column_stack(terms)
        
        @staticmethod
        def _build_gradient_operators(
            X,
            radius=None,
            stencil_width=20,
            poly_order=2,
        ):
            """
            Precompute MLS differentiation operators.

            Returns
            -------
            operators : list

                operators[ip]

                shape:
                    (ndim, nstencil)

            neighbors : list
            """

            from scipy.spatial import cKDTree

            X = np.asarray(X)

            npoints, ndim = X.shape

            tree = cKDTree(X)

            operators = []
            neighbors = []

            for ip in range(npoints):

                if radius is not None:

                    idx = tree.query_ball_point(
                        X[ip],
                        r=radius,
                    )

                else:

                    _, idx = tree.query(
                        X[ip],
                        k=min(
                            stencil_width,
                            npoints,
                        ),
                    )

                idx = np.asarray(idx)

                DX = X[idx] - X[ip]

                A = (
                    SAM.DifferentialOperators
                    ._polynomial_basis(
                        DX,
                        poly_order,
                    )
                )

                r = np.linalg.norm(
                    DX,
                    axis=1,
                )

                h = np.max(r) + 1e-12

                w = np.exp(
                    -(r / h) ** 2
                )

                W = np.diag(w)

                ATA = A.T @ W @ A

                P = np.linalg.pinv(ATA) @ A.T @ W

                G = P[1 : 1 + ndim]

                operators.append(G)
                neighbors.append(idx)

            return operators, neighbors

        @staticmethod
        def gradient(
            X,
            f,
            radius=None,
            stencil_width=20,
            poly_order=2,
        ):
            """
            Gradient of scalar field.

            Parameters
            ----------
            f

                shape:
                    (npoints,)

                or

                    (npoints, ncases)

            Returns
            -------
            grad

                shape:
                    (ndim, npoints, ncases)
            """

            X = np.asarray(X)
            f = np.asarray(f)

            if f.ndim == 1:
                f = f[:, None]

            npoints = X.shape[0]
            ncases = f.shape[1]
            ndim = X.shape[1]

            operators, neighbors = (
                SAM.DifferentialOperators
                ._build_gradient_operators(
                    X,
                    radius,
                    stencil_width,
                    poly_order,
                )
            )

            grad = np.empty(
                (
                    ndim,
                    npoints,
                    ncases,
                ),
                dtype=f.dtype,
            )

            for ip in range(npoints):

                idx = neighbors[ip]

                G = operators[ip]

                grad[:, ip, :] = (
                    G @ f[idx]
                )

            return grad

        @staticmethod
        def jacobian(
            X,
            U,
            radius=None,
            stencil_width=20,
            poly_order=2,
        ):
            """
            Parameters
            ----------
            U

                shape:
                    (ncomponents,
                    npoints,
                    ncases)

            Returns
            -------
            J

                shape:
                    (ncomponents,
                    ndim,
                    npoints,
                    ncases)
            """

            U = np.asarray(U)

            ncomponents = U.shape[0]
            ndim = X.shape[1]
            npoints = U.shape[1]
            ncases = U.shape[2]

            J = np.empty(
                (
                    ncomponents,
                    ndim,
                    npoints,
                    ncases,
                ),
                dtype=U.dtype,
            )

            for icomp in range(ncomponents):

                J[icomp] = (
                    SAM.DifferentialOperators
                    .gradient(
                        X,
                        U[icomp],
                        radius,
                        stencil_width,
                        poly_order,
                    )
                )

            return J
        
        @staticmethod
        def divergence(
            X,
            U,
            radius=None,
            stencil_width=20,
            poly_order=2,
        ):
            """
            Parameters
            ----------
            U

                shape:
                    (ndim,
                    npoints,
                    ncases)

            Returns
            -------
            div

                shape:
                    (npoints,
                    ncases)
            """

            U = np.asarray(U)

            ndim = X.shape[1]

            if U.shape[0] != ndim:

                raise ValueError(
                    f"Expected {ndim} components, "
                    f"got {U.shape[0]}"
                )

            J = (
                SAM.DifferentialOperators
                .jacobian(
                    X,
                    U,
                    radius,
                    stencil_width,
                    poly_order,
                )
            )

            div = np.zeros(
                (
                    U.shape[1],
                    U.shape[2],
                ),
                dtype=U.dtype,
            )

            for d in range(ndim):

                div += J[d, d]

            return div

    # =========================================================================
    # DICT VISUALIZER
    # =========================================================================
    class DictVisualizer:

        @staticmethod
        def _simplify(obj):
            if isinstance(obj, torch.Tensor):
                return f"Torch Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype})"
            elif isinstance(obj, np.ndarray):
                return f"Numpy Array(shape={tuple(obj.shape)}, dtype={obj.dtype})"
            elif isinstance(obj, pd.DataFrame):
                return f"DataFrame(shape={obj.shape}, cols={list(obj.columns)})"
            elif isinstance(obj, pd.Series):
                return f"Series(len={len(obj)}, dtype={obj.dtype})"
            elif isinstance(obj, pv.UnstructuredGrid):
                return (
                    f"UnstructuredGrid("
                    f"points={obj.points.shape}, cells={obj.cells.shape})"
                )
            elif isinstance(obj, pv.pyvista_ndarray):
                return f"PyVistaArray(shape={obj.shape}, dtype={obj.dtype})"
            elif isinstance(obj, set):
                return {SAM.DictVisualizer._simplify(v) for v in obj}
            elif isinstance(obj, dict):
                return {
                    k: SAM.DictVisualizer._simplify(v) for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [SAM.DictVisualizer._simplify(v) for v in obj]
            return str(obj)

        @staticmethod
        def pretty_print(d: dict, depth: int = 2, output_file: str = None):
            """
            Pretty-print a (possibly nested) dictionary, replacing large
            arrays with concise shape/dtype summaries.

            Parameters
            ----------
            d : dict
            depth : int
                pprint depth. Default 2.
            output_file : str or None
                If given, write to file instead of printing to stdout.
            """
            from pprint import pformat
            formatted = pformat(SAM.DictVisualizer._simplify(d), depth=depth)
            if not output_file:
                print(formatted)
            else:
                with open(output_file, "w", encoding="utf-8") as fh:
                    fh.write(formatted)

        @staticmethod
        def rich_tree(d: dict):
            """
            Print a nested dictionary as a Rich tree in the terminal.

            Parameters
            ----------
            d : dict
            """
            from rich.tree import Tree
            from rich.console import Console

            def build(data, node):
                for k, v in data.items():
                    branch = node.add(str(k))
                    if isinstance(v, dict):
                        build(v, branch)
                    else:
                        branch.add(str(v))

            root = Tree("root")
            build(SAM.DictVisualizer._simplify(d), root)
            Console().print(root)

        @staticmethod
        def plot_graph(d: dict):
            """
            Render a nested dictionary as a directed graph with networkx.

            Parameters
            ----------
            d : dict
            """
            import networkx as nx

            def to_graph(data, G=None, parent=None):
                if G is None:
                    G = nx.DiGraph()
                for k, v in data.items():
                    G.add_node(k)
                    if parent:
                        G.add_edge(parent, k)
                    if isinstance(v, dict):
                        to_graph(v, G, k)
                    else:
                        node_val = str(v)
                        G.add_node(node_val)
                        G.add_edge(k, node_val)
                return G

            G = to_graph(SAM.DictVisualizer._simplify(d))
            plt.figure(figsize=(8, 6))
            nx.draw(
                G, with_labels=True, font_size=8,
                node_size=2000, node_color="lightblue",
            )
            plt.show()
