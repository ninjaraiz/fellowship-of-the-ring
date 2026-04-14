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

from scipy.spatial import Delaunay

import seaborn as sns
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

from .EarendilsLight import EarendilsLight
class SAM():
    
    """
    SAM – Simulations & Analytics Module
    --------------------------------

    The steadfast companion to FRODO.

    SAM is a utility module designed to assist with machine learning workflows, 
    data transformation, and surrogate model preparation for CFD simulations 
    and other engineering data pipelines.

    From loading datasets in PyTorch, parsing structured HDF5 files, to organizing 
    and manipulating numerical arrays, SAM provides essential tools that augment 
    the capabilities of FRODO with a focus on:

        • Reading & extracting simulation data from .h5 files
        • Managing datasets for surrogate model training
        • Tensor transformations for ML frameworks like PyTorch
        • Utility operations for batch processing and preprocessing

    Whether you're climbing the mountain of model training or just organizing 
    your features, SAM is here to carry the weight.

    Use SAM alongside FRODO — because no great adventure is completed alone.
    """

    light = EarendilsLight(__name__)

    @classmethod
    def some_light(cls, name=None):
        """Atajo a Eärendil's Light."""
        return cls.light.help(name)

    class Gardener:
        
        def create_final_tensor(
            tensor_ptos: Union[torch.Tensor, np.ndarray],
            tensor_flcc: Union[torch.Tensor, np.ndarray],
            tensors_out: list[Union[torch.Tensor, np.ndarray]],
            tensors_aux: list[Union[torch.Tensor, np.ndarray]] = None,
            sol:Union[list[int], tuple[int], int, 'all'] = 'all',
            idx_flcc: Union[list[int], tuple[int], 'all'] = 'all',
            ref:Union[dict, None] = None,
            verbose: bool = False
            ):
            # --- Defensive defaults ---
            if tensors_aux is None:
                tensors_aux = []

            def ensure_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.clone()
                elif isinstance(x, np.ndarray):
                    return torch.from_numpy(x).clone()
                else:
                    raise TypeError(
                        "All inputs, outputs and aux tensors must be torch.Tensor or np.ndarray."
                    )

            tensor_ptos = ensure_tensor(tensor_ptos)
            tensor_flcc = ensure_tensor(tensor_flcc)
            tensors_out = [ensure_tensor(t) for t in tensors_out]
            tensors_aux = [ensure_tensor(t) for t in tensors_aux]

                    
            nptos = tensor_ptos.shape[0]
            ncases = tensor_flcc.shape[0]

            # --- Ensure basic shapes (cols) ---
            if tensor_ptos.dim() == 1:
                tensor_ptos = tensor_ptos.unsqueeze(1)
            if tensor_flcc.dim() == 1:
                tensor_flcc = tensor_flcc.unsqueeze(1)

            # --- Normalize tensors_out: ensure every out is 3D [nptos, ncases, nout] ---
            for i, out in enumerate(tensors_out):
                if out.dim() == 1:
                    raise ValueError(f'tensors_out[{i}] must be at least 2D (found 1D).')
                if out.dim() == 2:
                    # convert (nptos, ncases) -> (nptos, ncases, 1)
                    tensors_out[i] = out.unsqueeze(2)
                elif out.dim() > 3:
                    raise ValueError(f'tensors_out[{i}] has too many dims ({out.dim()}). Expected 2 or 3.')

            # --- Normalize tensors_aux: allow 1D (nptos,), 2D (nptos,ncases) or 3D (nptos,ncases,naux) ---
            for i, aux in enumerate(tensors_aux):
                if aux.dim() == 1:
                    # keep as-is (point-only feature)
                    continue
                elif aux.dim() == 2:
                    # expected (nptos, ncases)
                    continue
                elif aux.dim() == 3:
                    # expected (nptos, ncases, naux)
                    continue
                else:
                    raise ValueError(f'tensors_aux[{i}] has unsupported dim {aux.dim()}')

            # --- Optional case selection via idx_flcc ---
            if idx_flcc == 'all':
                idx_selected = torch.arange(ncases)
            else:
                if not isinstance(idx_flcc, (list, tuple)):
                    raise TypeError("idx_flcc must be list, tuple or None")

                idx_selected = torch.tensor(idx_flcc, dtype=torch.long)

                if torch.any(idx_selected >= ncases) or torch.any(idx_selected < 0):
                    raise IndexError("idx_flcc contains indices out of range")

            # Apply selection
            tensor_flcc = tensor_flcc[idx_selected, :]
            ncases = tensor_flcc.shape[0]

            # Outputs: (nptos, ncases, nout)
            tensors_out = [
                out[:, idx_selected, :] if out.dim() >= 2 else out
                for out in tensors_out
            ]

            # Aux tensors
            new_aux = []
            for aux in tensors_aux:
                if aux.dim() == 1:
                    new_aux.append(aux)
                elif aux.dim() == 2:
                    new_aux.append(aux[:, idx_selected])
                elif aux.dim() == 3:
                    new_aux.append(aux[:, idx_selected, :])
            tensors_aux = new_aux

            if sol == 'all':
                pass  # keep all channels as they are
            elif isinstance(sol, int):
                # For outputs: keep last dim as single channel after slicing
                tensors_out = [out[:, :, sol:sol+1] if out.shape[2] > 1 else out for out in tensors_out]
                # For aux: only slice if aux is 3D and has >1 in last dim
                tensors_aux = [
                    (aux[:, :, sol:sol+1] if (aux.dim()==3 and aux.shape[2] > 1) else aux)
                    for aux in tensors_aux
                ]
            elif isinstance(sol, (list, tuple)):
                # For outputs: select specified channels, keep as multi-channel if len(sol)>1
                tensors_out = [out[:, :, sol] if out.shape[2] > 1 else out for out in tensors_out]
                # For aux: only slice if aux is 3D and has >1 in last dim
                tensors_aux = [
                    (aux[:, :, sol] if (aux.dim()==3 and aux.shape[2] > 1) else aux)
                    for aux in tensors_aux
                ]
            else:
                raise TypeError('sol must be int, list of ints, or "all".')
            
            # --- Expand points and flcc to row-wise dataset: (nptos*ncases, ...) ---
            points_repeated = tensor_ptos.repeat(ncases, 1)             # (nptos*ncases, ncoord)
            flcc_expanded = tensor_flcc.repeat_interleave(nptos, dim=0)  # (nptos*ncases, nflcc)

            if verbose:
                print(f'points_repeated: {points_repeated.shape}')
                print(f'flcc_expanded:   {flcc_expanded.shape}')

            # --- Helpers: flatten keeping ordering case-major (case blocks, each with nptos rows) ---
            def flatten_aux(aux):
                if aux.dim() == 1:
                    # repeat whole vector for each case -> [p0.., p0.., ...] matching points_repeated
                    return aux.repeat(ncases).unsqueeze(1)
                elif aux.dim() == 2:
                    # aux: (nptos, ncases) -> permute to (ncases, nptos) then flatten rows -> (nptos*ncases, 1)
                    return aux.permute(1, 0).reshape(-1, 1)
                elif aux.dim() == 3:
                    # (nptos, ncases, naux) -> (ncases, nptos, naux) -> flatten to (nptos*ncases, naux)
                    return aux.permute(1, 0, 2).reshape(-1, aux.shape[2])
                else:
                    raise RuntimeError('unexpected aux dim')

            def flatten_out(out):
                # out guaranteed 3D here: (nptos, ncases, nout)
                return out.permute(1, 0, 2).reshape(-1, out.shape[2])  # (ncases*nptos, nout)

            # --- Flatten aux ---
            aux_flattened = []
            for aux in tensors_aux:
                flat = flatten_aux(aux)
                if verbose:
                    print(f'aux original dim {aux.dim():d} -> flattened {flat.shape}')
                aux_flattened.append(flat)

            # --- Flatten outputs ---
            out_flattened = []
            for out in tensors_out:
                flat = flatten_out(out)
                if verbose:
                    print(f'out original {out.shape} -> flattened {flat.shape}')
                out_flattened.append(flat)

            # --- Concatenate horizontally ---
            concat_list = [points_repeated, flcc_expanded] + aux_flattened + out_flattened
            final_tensor = torch.cat(concat_list, dim=1)

            if verbose:
                print(f'Final tensor shape: {final_tensor.shape}')

            # --- Normalización robusta (evitar división por cero) ---
            
            if ref == None:
                min_vals = final_tensor.min(dim=0, keepdim=True)[0]
                max_vals = final_tensor.max(dim=0, keepdim=True)[0]
            else:
                min_vals = ref['mins']
                max_vals = ref['maxs']
            
            denom = (max_vals - min_vals)
            denom[denom == 0] = 1e-8  # evita NaNs
            final_tensor_scaled = (final_tensor - min_vals) / denom
                
                
            # --- Información ---
            aux_aport = 0
            for aux in tensors_aux:
                if aux.dim() == 1:
                    aux_aport += 1
                elif aux.dim() == 2:
                    aux_aport += 1
                else:
                    aux_aport += aux.shape[2]

            noutputs = sum(out.shape[2] for out in tensors_out)

            return {
                'tensor': final_tensor,
                'scaled': final_tensor_scaled,
                'mins': min_vals.squeeze(),
                'maxs': max_vals.squeeze(),
                'info': {
                    'ninputs': tensor_ptos.shape[1] + tensor_flcc.shape[1] + aux_aport,
                    'noutputs': noutputs
                }
            }
            
        def create_final_tensor_ant(
            tensor_ptos: Union[torch.Tensor, np.ndarray],
            tensor_flcc: Union[torch.Tensor, np.ndarray],
            tensors_out: list[Union[torch.Tensor, np.ndarray]],
            tensors_aux: list[Union[torch.Tensor, np.ndarray]] = None,
            sol='all',
            n=None,
            ref:Union[dict, None] = None,
            verbose: bool = False
            ):
            # --- Defensive defaults ---
            if tensors_aux is None:
                tensors_aux = []

            def ensure_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.clone()
                elif isinstance(x, np.ndarray):
                    return torch.from_numpy(x).clone()
                else:
                    raise TypeError(
                        "All inputs, outputs and aux tensors must be torch.Tensor or np.ndarray."
                    )

            tensor_ptos = ensure_tensor(tensor_ptos)
            tensor_flcc = ensure_tensor(tensor_flcc)
            tensors_out = [ensure_tensor(t) for t in tensors_out]
            tensors_aux = [ensure_tensor(t) for t in tensors_aux]

                    
            nptos = tensor_ptos.shape[0]
            ncases = tensor_flcc.shape[0]

            # --- Ensure basic shapes (cols) ---
            if tensor_ptos.dim() == 1:
                tensor_ptos = tensor_ptos.unsqueeze(1)
            if tensor_flcc.dim() == 1:
                tensor_flcc = tensor_flcc.unsqueeze(1)

            # --- Normalize tensors_out: ensure every out is 3D [nptos, ncases, nout] ---
            for i, out in enumerate(tensors_out):
                if out.dim() == 1:
                    raise ValueError(f'tensors_out[{i}] must be at least 2D (found 1D).')
                if out.dim() == 2:
                    # convert (nptos, ncases) -> (nptos, ncases, 1)
                    tensors_out[i] = out.unsqueeze(2)
                elif out.dim() > 3:
                    raise ValueError(f'tensors_out[{i}] has too many dims ({out.dim()}). Expected 2 or 3.')

            # --- Normalize tensors_aux: allow 1D (nptos,), 2D (nptos,ncases) or 3D (nptos,ncases,naux) ---
            for i, aux in enumerate(tensors_aux):
                if aux.dim() == 1:
                    # keep as-is (point-only feature)
                    continue
                elif aux.dim() == 2:
                    # expected (nptos, ncases)
                    continue
                elif aux.dim() == 3:
                    # expected (nptos, ncases, naux)
                    continue
                else:
                    raise ValueError(f'tensors_aux[{i}] has unsupported dim {aux.dim()}')

            # --- Optional trimming in n (number of cases) ---
            if n is not None:
                tensor_flcc = tensor_flcc[:n, :]
                ncases = tensor_flcc.shape[0]
                tensors_out = [out[:, :n, :] if out.dim() >= 2 else out for out in tensors_out]
                new_aux = []
                for aux in tensors_aux:
                    if aux.dim() == 1:
                        new_aux.append(aux)
                    elif aux.dim() >= 2:
                        new_aux.append(aux[:, :n] if aux.dim()==2 else aux[:, :n, :])
                tensors_aux = new_aux

            # --- Selection of solution index (keep last dim) ---
            if sol != 'all':
                if not isinstance(sol, int):
                    raise TypeError('sol must be integer index or "all"')
                # For outputs: keep last dim as single channel after slicing
                tensors_out = [out[:, :, sol:sol+1] if out.shape[2] > 1 else out for out in tensors_out]
                # For aux: only slice if aux is 3D and has >1 in last dim
                tensors_aux = [
                    (aux[:, :, sol:sol+1] if (aux.dim()==3 and aux.shape[2] > 1) else aux)
                    for aux in tensors_aux
                ]

            # --- Expand points and flcc to row-wise dataset: (nptos*ncases, ...) ---
            points_repeated = tensor_ptos.repeat(ncases, 1)             # (nptos*ncases, ncoord)
            flcc_expanded = tensor_flcc.repeat_interleave(nptos, dim=0)  # (nptos*ncases, nflcc)

            if verbose:
                print(f'points_repeated: {points_repeated.shape}')
                print(f'flcc_expanded:   {flcc_expanded.shape}')

            # --- Helpers: flatten keeping ordering case-major (case blocks, each with nptos rows) ---
            def flatten_aux(aux):
                if aux.dim() == 1:
                    # repeat whole vector for each case -> [p0.., p0.., ...] matching points_repeated
                    return aux.repeat(ncases).unsqueeze(1)
                elif aux.dim() == 2:
                    # aux: (nptos, ncases) -> permute to (ncases, nptos) then flatten rows -> (nptos*ncases, 1)
                    return aux.permute(1, 0).reshape(-1, 1)
                elif aux.dim() == 3:
                    # (nptos, ncases, naux) -> (ncases, nptos, naux) -> flatten to (nptos*ncases, naux)
                    return aux.permute(1, 0, 2).reshape(-1, aux.shape[2])
                else:
                    raise RuntimeError('unexpected aux dim')

            def flatten_out(out):
                # out guaranteed 3D here: (nptos, ncases, nout)
                return out.permute(1, 0, 2).reshape(-1, out.shape[2])  # (ncases*nptos, nout)

            # --- Flatten aux ---
            aux_flattened = []
            for aux in tensors_aux:
                flat = flatten_aux(aux)
                if verbose:
                    print(f'aux original dim {aux.dim():d} -> flattened {flat.shape}')
                aux_flattened.append(flat)

            # --- Flatten outputs ---
            out_flattened = []
            for out in tensors_out:
                flat = flatten_out(out)
                if verbose:
                    print(f'out original {out.shape} -> flattened {flat.shape}')
                out_flattened.append(flat)

            # --- Concatenate horizontally ---
            concat_list = [points_repeated, flcc_expanded] + aux_flattened + out_flattened
            final_tensor = torch.cat(concat_list, dim=1)

            if verbose:
                print(f'Final tensor shape: {final_tensor.shape}')

            # --- Normalización robusta (evitar división por cero) ---
            
            if ref == None:
                min_vals = final_tensor.min(dim=0, keepdim=True)[0]
                max_vals = final_tensor.max(dim=0, keepdim=True)[0]
            else:
                min_vals = ref['mins']
                max_vals = ref['maxs']
            
            denom = (max_vals - min_vals)
            denom[denom == 0] = 1e-8  # evita NaNs
            final_tensor_scaled = (final_tensor - min_vals) / denom
                
                
            # --- Información ---
            aux_aport = 0
            for aux in tensors_aux:
                if aux.dim() == 1:
                    aux_aport += 1
                elif aux.dim() == 2:
                    aux_aport += 1
                else:
                    aux_aport += aux.shape[2]

            noutputs = sum(out.shape[2] for out in tensors_out)

            return {
                'tensor': final_tensor,
                'scaled': final_tensor_scaled,
                'mins': min_vals.squeeze(),
                'maxs': max_vals.squeeze(),
                'info': {
                    'ninputs': tensor_ptos.shape[1] + tensor_flcc.shape[1] + aux_aport,
                    'noutputs': noutputs
                }
            }
           
        def create_final_tensor_scored(
            tensor_ptos: torch.Tensor,
            tensor_flcc: torch.Tensor,
            tensor_out: torch.Tensor,
            score_law:str,
            sol='all',
            score_csv_path = None,
            n=None,
            ref=None,
            verbose:bool = False
            ):
            """
            Recibe tres tensores de torch para crear un tensor final, con el que se podrá hacer un dataset completo
            y además un tensor de score para cada punto.

            Args:
                tensor_ptos (torch.Tensor): Tensor de puntos.
                tensor_flcc (torch.Tensor): Tensor de condiciones de vuelo.
                tensor_out (torch.Tensor): Tensor de salida.
                score_law (str): Ley de normalización del score. Opciones: 'log10', 'inv', 'sqrt_inv', 'exp_inv', 'linear'.
                sol (int, optional): Índice de la solución a extraer. Por defecto 'all'.
                score_csv_path (str, optional): Ruta para guardar el CSV de scores. Por defecto None.
                n (int, optional): Número de casos a extraer. Por defecto None.
                ref (dict, optional): Diccionario con valores de referencia para la normalización. Por defecto None.

            Returns:
                dict: Diccionario con los tensores finales, normalizados y con score.
            """
            if len(tensor_out.shape) == 2:
                print('WARNING: tensor_out must be 3D. Adding a third dimension.')
                tensor_out = tensor_out.unsqueeze(2)
            elif len(tensor_out.shape) == 1:
                raise ValueError('ERROR: tensor_out must be 3D, but received a 1D tensor.')

            if n is not None:
                tensor_flcc = tensor_flcc[:n, :]
                tensor_out = tensor_out[:, :n, :]

            if sol != 'all':
                tensor_out = tensor_out[:, :, sol]

            if verbose:
                print('CREANDO SET DE DATOS SESGADO\n')
                print(f'    Ptos: {tensor_ptos.shape}')
                print(f'    FLCC: {tensor_flcc.shape}')
                print(f'    out: {tensor_out.shape}')

            ### CALCULANDO GRADIENTES
            diff_ptos = tensor_ptos
            P, F, C = tensor_out.shape

            points_repeated = tensor_ptos.repeat(tensor_flcc.size(0), 1)
            flcc_expanded = tensor_flcc.repeat_interleave(tensor_ptos.size(0), dim=0)
            out_flattened = tensor_out.permute(1, 0, 2).reshape(-1, tensor_out.size(2))

            tensor_score = torch.zeros_like(out_flattened)
            n_vec = []
            bin_vec = []
            score_per_bin_vec = []

            for c in range(out_flattened.shape[1]):
                freq, bins = np.histogram(out_flattened[:, c].numpy(), bins=100)

                if score_law == 'log10':
                    # log_n = np.where(freq > 0, np.log10(freq), 1)
                    inv_freq = np.where(freq > 0, 1 / np.log10(freq), 0)

                elif score_law == 'inv':
                    inv_freq = np.where(freq > 1, 1 / freq, 1)

                elif score_law == 'sqrt_inv':
                    inv_freq = np.where(freq > 0, 1 / np.sqrt(freq), 0)

                elif score_law == 'exp_inv':
                    max_n = np.max(freq) if np.max(freq) > 0 else 1
                    inv_freq = np.exp(-freq / max_n)

                elif score_law == 'linear':
                    max_n = np.max(freq) if np.max(freq) > 0 else 1
                    inv_freq = 1 - (freq / max_n)
                    
                else:
                    raise ValueError(f"ERROR: score_law '{score_law}' no reconocido.")

                scale_vector = np.ones(len(inv_freq)) #np.linspace(0, 1, len(inv_freq))
                score_per_bin = inv_freq * scale_vector

                score_per_bin = np.nan_to_num(score_per_bin, nan=0.0)
                score_per_bin = (score_per_bin - score_per_bin.min()) / (score_per_bin.max() - score_per_bin.min())

                bin_indices = np.digitize(out_flattened[:, c].numpy(), bins, right=True) - 1
                bin_indices = np.clip(bin_indices, 0, len(freq) - 1)
                
                n_vec.append(freq)
                bin_vec.append(bin_indices)
                score_per_bin_vec.append(score_per_bin)
                tensor_score[:, c] = torch.from_numpy(score_per_bin[bin_indices])

            final_tensor = torch.cat((points_repeated, flcc_expanded, out_flattened), dim=1)

            if ref is None:
                min_vals = final_tensor.min(dim=0, keepdim=True)[0]
                max_vals = final_tensor.max(dim=0, keepdim=True)[0]
            else:
                min_vals = ref['mins']
                max_vals = ref['maxs']
            
        
            final_tensor_scaled = (final_tensor - min_vals) / (max_vals - min_vals)

            tensor_score_ = tensor_score.reshape(P, F, C)
            if verbose:
                print(f'    tensor_score: {tensor_score_.shape}\n')
                print(f'    set: {final_tensor.shape}\n')

            if score_csv_path is not None:
                for var in range(tensor_out.shape[2]):
                    header = ['x', 'y', 'z']
                    casos = []
                    for c in range(tensor_flcc.shape[0]):
                        casos.append(f'case_{c}')

                    header.extend(casos)
                    df = pd.DataFrame(torch.cat((tensor_ptos, tensor_score_[:,:,var]), dim=1))
                    df.to_csv(f'{score_csv_path}/tensor_score_var_{var}.csv', header=header, index=False)
                print(f'Exported tensor_score to csv file in {score_csv_path}/')
                
            return {
                'tensor': final_tensor,
                'scaled': final_tensor_scaled,
                'mins': min_vals.squeeze(),
                'maxs': max_vals.squeeze(),
                'score': torch.from_numpy(tensor_score.numpy().copy()),
                'score_array': tensor_score_,
                'n_vec': torch.tensor(np.array(n_vec, dtype=np.float32)),
                'bin_vec': torch.tensor(np.array(bin_vec, dtype=np.int64)),
                'score_per_bin': torch.tensor(np.array(score_per_bin_vec, dtype=np.float32)),
                'info': {'ninputs': tensor_ptos.shape[1] + tensor_flcc.shape[1], 'noutputs': tensor_out.shape[2]}
            }
        
        def concatenate_sets(sets:tuple, ref:int=0,  score:bool = False):
            if score:
                raise NotImplementedError("Método todavía no implementado para datasets con scores")
            else:
                if ref > len(sets):
                    raise ValueError(f"ref debe corresponder a un índice válido en sets (0 a {len(sets)-1})")
                new_set = sets[0]
                for one_set in sets[1:]:
                    for key in ['tensor', 'scaled']:
                        new_set[key] = torch.concatenate((new_set[key], one_set[key]), axis=0)

                for key in ['mins', 'maxs', 'info']:
                    new_set[key]=sets[ref][key]
                    
            return new_set
        
        def reduce_dataset_per_frequency(
            dataset,
            lim:int,
            reduce_factor:float = 0.8,
            ref = None,
            plot_path = None,
            # geometry_path:bool = False
            ):
            """
            Dado un dataset, filtra la salida del mismo por un valor de frecuencia en el histograma, una cantidad proporcional al reduce_factor.
            """
            n_vec = dataset['n_vec'][0]
            bins_to_check = torch.where(n_vec > lim)[0] # bines con más frecuencia que lim
            ind_bin = torch.nonzero(torch.isin(dataset['bin_vec'][0], bins_to_check), as_tuple = True)[0] # filas del dataset que están en los bines malos
            print(f'Bins to reduce: {bins_to_check}\n')
            #proceso para elegir el reduce_factor %
            ind_bin_permuted = ind_bin[torch.randperm(ind_bin.shape[0])]
            to_remove = ind_bin_permuted[:int(ind_bin.shape[0]*reduce_factor)]

            final_mask = torch.ones_like(dataset['bin_vec'][0], dtype=torch.bool)
            final_mask[to_remove]=False
            #obtenida máscara booleana de filtrado. Aplicar a todos los tensores de dataset

            final_tensor = dataset['tensor'][final_mask,:]

            if ref is None:
                min_vals = final_tensor.min(dim=0, keepdim=True)[0]
                max_vals = final_tensor.max(dim=0, keepdim=True)[0]
            else:
                min_vals = ref['mins']
                max_vals = ref['maxs']

            final_tensor_scaled = (final_tensor - min_vals) / (max_vals - min_vals)

            print(f'Filtered dataset shape: {final_tensor.shape}')
            n_outputs = dataset['score'].shape[1]

            n_vec=[]
            bin_vec=[]
            tensor_score=torch.zeros((final_tensor.shape[0], n_outputs))
            for c in range(n_outputs):
                n, bins = np.histogram(final_tensor[:,-c].numpy(), bins=100)
                inv_freq = 1 / n
                # Crear un vector de normalización con paso fijo (0 a 1)
                scale_vector = np.linspace(0, 1, len(inv_freq))

                # Multiplicación para mantener la relación inversa con la frecuencia
                score_per_bin = inv_freq * scale_vector

                # Normalización para asegurar el rango [0,1]
                score_per_bin = (score_per_bin - score_per_bin.min()) / (score_per_bin.max() - score_per_bin.min())

                # Indexar los valores en los bins
                bin_indices = np.digitize(final_tensor[:,-c].numpy(), bins, right=True) - 1
                bin_indices = np.clip(bin_indices, 0, len(n) - 1)
                n_vec.append(n)
                bin_vec.append(bin_indices)
                tensor_score[:,c]=torch.from_numpy(np.transpose(score_per_bin[bin_indices]))

            # if geometry_path is not None:
            #     os.makedirs(geometry_path, exist_ok=True)

            #     # Extraer X, Y, Z y Score
            #     original_geom = torch.cat((dataset['tensor'][:, :3], dataset['score']), dim=1).numpy()
            #     reduced_geom = torch.cat((final_tensor[:, :3], tensor_score), dim=1).numpy()

            #     # Guardar en CSV
            #     np.savetxt(f"{geometry_path}/geometry_original.csv", original_geom, delimiter=",", header="X,Y,Z,Score", comments="")
            #     np.savetxt(f"{geometry_path}/geometry_reduced.csv", reduced_geom, delimiter=",", header="X,Y,Z,Score", comments="")

            #     print(f"Geometría exportada en {geometry_path}")

            if plot_path is not None:
                fig = plt.figure(figsize=(20,20))
                ax1 = fig.add_subplot(221)
                hist1 = ax1.hist(final_tensor[:,-1], bins=100, color='blue', alpha=0.7, edgecolor='black')
                # tam: numero de repeticiones que hay en el bin correspondiente
                # bins: delimitaciones del bin. Tiene de longitud len(tam) + 1, porque cuenta con ambos lados.
                # patches: es el objeto rectágulo de matplotlib, sin importancia ahora.
                ax1.set_title(f"Histograma cP capado.", fontsize=16)
                ax1.set_xlabel("cP", fontsize=14)
                ax1.set_ylabel("Frecuencia", fontsize=14)
                ax1.set_yscale('log')
                ax1.grid(True, linestyle="--", alpha=0.5)

                ax2 = fig.add_subplot(222)
                hist2 = ax2.hist(dataset['tensor'][:,-1], bins=100, color='blue', alpha=0.7, edgecolor='black')
                # tam: numero de repeticiones que hay en el bin correspondiente
                # bins: delimitaciones del bin. Tiene de longitud len(tam) + 1, porque cuenta con ambos lados.
                # patches: es el objeto rectágulo de matplotlib, sin importancia ahora.
                ax2.set_title(f"Histograma cP.", fontsize=16)
                ax2.set_xlabel("cP", fontsize=14)
                ax2.set_ylabel("Frecuencia", fontsize=14)
                ax2.set_yscale('log')
                ax2.grid(True, linestyle="--", alpha=0.5)

                max_y = max(max(hist1[0]), max(hist2[0]))  # Get the max frequency from both histograms
                ax1.set_ylim(1, max_y)  # Set same y-axis limits
                ax2.set_ylim(1, max_y)
                
                ax3 = fig.add_subplot(223)
                hist3 = ax3.hist(tensor_score, bins=100, color='blue', alpha=0.7, edgecolor='black')
                # tam: numero de repeticiones que hay en el bin correspondiente
                # bins: delimitaciones del bin. Tiene de longitud len(tam) + 1, porque cuenta con ambos lados.
                # patches: es el objeto rectágulo de matplotlib, sin importancia ahora.
                ax3.set_title(f"Histograma notas capado.", fontsize=16)
                ax3.set_xlabel("Nota", fontsize=14)
                ax3.set_ylabel("Frecuencia", fontsize=14)
                ax3.set_yscale('log')
                ax3.grid(True, linestyle="--", alpha=0.5)

                ax4 = fig.add_subplot(224)
                hist4 = ax4.hist(dataset['score'], bins=100, color='blue', alpha=0.7, edgecolor='black')
                # tam: numero de repeticiones que hay en el bin correspondiente
                # bins: delimitaciones del bin. Tiene de longitud len(tam) + 1, porque cuenta con ambos lados.
                # patches: es el objeto rectágulo de matplotlib, sin importancia ahora.
                ax4.set_title(f"Histograma notas.", fontsize=16)
                ax4.set_xlabel("Nota", fontsize=14)
                ax4.set_ylabel("Frecuencia", fontsize=14)
                ax4.set_yscale('log')
                ax4.grid(True, linestyle="--", alpha=0.5)
                
                fig.savefig(plot_path + "/hist_dataset_reduced.jpg")
                plt.show()

            return dict({
                'tensor': final_tensor,
                'scaled': final_tensor_scaled,
                'mins': min_vals.squeeze(),
                'maxs': max_vals.squeeze(),
                'score': tensor_score,
                'n_vec': torch.from_numpy(np.array(n_vec)),
                'bin_vec': torch.from_numpy(np.array(bin_vec)),
                'info': dataset['info']
            })
            
    class HDF5reader:
                
        def __init__(self, file_path, verbose=False):
            self.file_path = file_path
            self.labels = []  # Lista para guardar las etiquetas
            self.explore_file(verbose)  # Pasar verbose al explorador

            def filter_list_with_character(lst, char="/"):
                return [item for item in lst if char in item]
            self.labels = filter_list_with_character(self.labels)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"El archivo no existe: {file_path}")
        
        def explore_file(self, verbose):
            with h5py.File(self.file_path, 'r') as f:
                def collect_attrs(name, obj):
                    self.labels.append(name)
                    if verbose:
                        print(name)
                    for key, val in obj.attrs.items():
                        if verbose:
                            print(f"    {key}: {val}")
                f.visititems(collect_attrs)
        
        def print_keys(self):
            with h5py.File(self.file_path, 'r') as f:
                def printname(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(name)
                f.visititems(printname)

        def load_to_numpy(self, key, show_data=False):
            with h5py.File(self.file_path, 'r') as f:
                data = f[key][:]
                array = np.array(data, dtype=np.float32)
                if show_data:
                    print(f"Datos del dataset '{key}':")
                    print(array.shape)
                return array
            
        def load_to_tensor(self, key, show_data=False):
            with h5py.File(self.file_path, 'r') as f:
                data = f[key][:]
                tensor = torch.tensor(data, dtype=torch.float32)
                if show_data:
                    print(f"Datos del dataset '{key}':")
                    print(tensor.shape)
                return tensor
    
    class Backpack:
        
        @staticmethod
        def reset_and_change_node_from_some_cases(
            db: 'FRODO',
            new_node: str,
            case_type: Literal["non-converged", "non-started"],    
            ):
            """
            Reset and modify compute node assignment for non-converged or non-started cases.

            Args:
                db (FRODO): FRODO instance for metadata access.
                new_node (str): Name of the new compute node to assign.
                case_type (Literal["non-converged","non-started"]): Type of cases to modify.

            Returns:
                None. Updates run.sh files in place and prints modifications made.
            """

            res = FRODO.residuals(db = db)
            df = res.get_all_final_residuals(
                scaled=True, verbose=False, only_finished=False)
            if case_type == "non-converged":
                mask = (
                    (df.iloc[:, 4] > 1e-5) &
                    (df.iloc[:, 5] > 1e-5) &
                    (df.iloc[:, 6] > 1e-5)
                )
            elif case_type == "non-started":
                mask = (
                    (df.iloc[:, 4].isna()) &
                    (df.iloc[:, 5].isna()) &
                    (df.iloc[:, 6].isna())
                )
            cases_selected = df[mask]

            if cases_selected.empty:
                print("No hay casos seleccionados.")
                return

            for row in cases_selected.itertuples():
                case_name = 'aoa_{:.4f}_mach_{:.4f}'.format(row[1], row[2])
                print(f"Empezando carpeta {case_name}")
                for file in os.listdir(db.sim_metadata[case_name]['path']):
                    if file.startswith("output") and case_type == "non-converged":
                        # os.remove(os.path.join(db.sim_metadata[case_name]['path'], file))
                        print(f"hubiese borrado {file}")
                    elif file == "run.sh":
                        run_file_path = os.path.join(db.sim_metadata[case_name]['path'], file)
                        with open(run_file_path, 'r') as f:
                            lineas = f.readlines()
                        
                        for i, linea in enumerate(lineas):
                            if linea.startswith("#SBATCH --nodelist="):
                                print(f"Modificado run.sh, en línea {linea}, poniendo el nodo {new_node}")
                                lineas[i] = f"#SBATCH --nodelist={new_node}\t \t # node name\n"
                        with open(run_file_path, 'w') as f:
                            f.writelines(lineas)                
            
        @staticmethod
        def find_files(path: str, file_end: str, infile:Union[str, None] = None, notinfile:Union[str, None] = None, verbose: bool = True):
            """
            Find all files in a directory that end with a specific suffix.

            Args:
                path (str): Directory to search in.
                infile (Union[str, None]): String that must be included in the file name. If None, this condition is ignored. Default is None.
                notinfile (Union[str, None]): String that must NOT be included in the file name. If None, this condition is ignored. Default is None.
                file_end (str): String suffix that files must end with.
                verbose (bool): If True, print a warning when no files are found.

            Returns:
                list[str]: Sorted list of file paths matching the suffix.
            """

            files = []

            for file in os.listdir(path):
                if file.endswith(file_end) and (infile is None or infile in file) and (notinfile is None or notinfile not in file):
                    files.append(os.path.join(path, file))
                    
            files.sort()
            if len(files) == 0:
                if verbose:
                    print(f"WARNING: No files found in {path} with the ending {file_end}")
            return files
  
        @staticmethod
        def read_cfd_times(case_path, verbose: bool = True):
            """
            Read CFD timing information from a -out.txt file.

            Args:
                case_path (str): Path to the simulation folder.
                verbose (bool): If True, print warnings about missing/duplicate files.

            Returns:
                dict: Timing information with start/end times and per-stage durations in hours.
            """

            file = SAM.Backpack.find_files(case_path, "-out.txt", verbose=False)
            if len(file) == 0:
                if verbose:
                    print(f"WARNING: No files found in {case_path} with the ending -out.txt")
                return None
            elif len(file) > 1:
                if verbose:
                    print(f"WARNING: More than one file found in {case_path} with the ending -out.txt")
                return None
            else:
                file = file[0]

            with open(file, 'r') as f:
                content = f.read()

            # 1. Extraer fechas (inicio y fin)
            date_pattern = r'\w{3} +(\w{3} +\d{1,2} +\d{2}:\d{2}:\d{2}) +CEST +(\d{4})'
            # date_pattern = r'\w{3} +\w{3} +\d{1,2} +\d{2}:\d{2}:\d{2} +CEST +\d{4}'
            matches = re.findall(date_pattern, content)

            if len(matches) < 2:
                raise ValueError("No se encontraron suficientes marcas de tiempo.")

            parse_format = "%b %d %H:%M:%S %Y"
            datetime_strings = [f"{month_day_time} {year}" for (month_day_time, year) in matches]

            start_time = datetime.strptime(datetime_strings[0], parse_format)
            end_time = datetime.strptime(datetime_strings[-1], parse_format)
            total_duration = end_time - start_time

            # 2. Extraer tiempos por etapa en [h] o [min]
            stage_pattern = r'TimeIntegration::Iterate\(\)\s+([\d.]+)\s+\[(h|min|days)\] \(wall clock time\)'
            stage_matches = re.findall(stage_pattern, content)

            stage_times_hours = []
            for value, unit in stage_matches:
                time = float(value)
                if unit == 'min':
                    time /= 60.0  # convertir minutos a horas
                elif unit == 'days':
                    time *= 24.0  # convertir días a horas
                stage_times_hours.append(time)

            return {
                'start_time': start_time,
                'end_time': end_time,
                'total_duration': total_duration,
                'stage_times_hours': stage_times_hours,
                'stage_total_hours': sum(stage_times_hours)
            }
            
        @staticmethod
        def same_columns(array, atol=1e-6, rtol=1e-5):
            """
            Check if all arrays along the first axis have identical values within tolerances.

            Args:
                array (np.ndarray): Array of shape (n_cases, n_points, n_dim) to compare.
                atol (float): Absolute tolerance for comparison.
                rtol (float): Relative tolerance for comparison.

            Returns:
                bool: True if all arrays are equal within tolerances, False otherwise.
            """

            base = array[0]
            iguales = True

            for i in range(1, array.shape[0]):
                malla_i = array[i]
                # Comprobación similar a torch.allclose
                if not np.allclose(base, malla_i, atol=atol, rtol=rtol):
                    iguales = False

                    # Diferencia absoluta y relativa
                    diferencia_absoluta = np.abs(base - malla_i)
                    diferencia_relativa = diferencia_absoluta / (np.abs(base) + rtol)

                    # Máscara de puntos distintos
                    mask_dif = (diferencia_absoluta > atol) & (diferencia_relativa > rtol)
                    puntos_distintos = np.any(mask_dif, axis=1)  # (P,)
                    indices_diferentes = np.nonzero(puntos_distintos)[0]  # Devuelve los índices

                    print(f"\nMalla {i} difiere de la malla 0 en {indices_diferentes.size} puntos.")

            return iguales
        
        @staticmethod
        def get_unified_connectivity(mesh):
            """
            Generate a unified connectivity array from a pyvista mesh.

            Supports arbitrary VTK cell types.
            Pads with -1 up to the maximum number of nodes per cell.
            """

            # Obtener todos los tipos presentes
            cell_dict = mesh.cells_dict

            # Determinar máximo número de nodos por celda
            max_nodes = max(arr.shape[1] for arr in cell_dict.values())

            # Contar total de celdas
            total_cells = sum(arr.shape[0] for arr in cell_dict.values())

            # Inicializar con -1
            connectivity = np.full((total_cells, max_nodes), -1, dtype=int)

            start = 0
            for _, cells in cell_dict.items():
                n = cells.shape[0]
                connectivity[start:start+n, :cells.shape[1]] = cells
                start += n

            return connectivity

        @staticmethod
        def get_unified_connectivity_ant(mesh):
            """
            Generate a unified 4-column connectivity array from a pyvista mesh.

            - Triangles (VTK type 5) fill first 3 columns, last column = -1.
            - Quads (VTK type 9) fill all 4 columns.

            Args:
                mesh (pyvista.UnstructuredGrid): Mesh with cells_dict containing 5 (triangles) and 9 (quads).

            Returns:
                np.ndarray: Connectivity array of shape (n_cells, 4) with -1 padding for triangles.
            """

            triangles = mesh.cells_dict.get(5, np.empty((0, 3), dtype=int))
            quads = mesh.cells_dict.get(9, np.empty((0, 4), dtype=int))

            n_tris = len(triangles)
            n_quads = len(quads)

            # Crear matriz con -1 (relleno)
            connectivity = np.full((n_tris + n_quads, 4), -1, dtype=int)

            # Triángulos ocupan 3 columnas
            connectivity[:n_tris, :3] = triangles

            # Cuadriláteros ocupan 4 columnas
            connectivity[n_tris:, :] = quads

            return connectivity
        
        @staticmethod
        def ensure_cell_data(mesh):
            """
            Ensure all point_data arrays are converted and added to cell_data in the mesh.

            Args:
                mesh (pyvista.UnstructuredGrid): The mesh to modify.

            Returns:
                pyvista.UnstructuredGrid: The mesh with cell_data guaranteed.
            """
            if mesh.point_data:
                converted = mesh.point_data_to_cell_data()
                for name, arr in converted.cell_data.items():
                    if name not in mesh.cell_data:
                        mesh.cell_data[name] = arr
            return mesh
        
        @staticmethod
        def create_tensors_from_h5(file_path:str, stage:int = 0):
            
            results = {}
            with h5py.File(file_path, 'r') as h5file:
                for cad_key in h5file:
                    if cad_key == "sim_metadata":
                        continue
                    cad_group = h5file[cad_key]
                    mesh = cad_group["Mesh"]
                    
                    coord = mesh["Coord"][()]
                    idx_sort = mesh["idx_sort"][()]
                    
                    flcc = cad_group["FlCc"][()]
                    vars_dict = {}
                    if "Vars" in cad_group:
                        for var_name in cad_group["Vars"][str(stage)]:
                            vars_dict[var_name] = cad_group["Vars"][str(stage)][var_name][()]
                    
                    results[cad_key] = {
                        "Coord": coord,
                        "idx_sort": idx_sort,
                        "flcc": flcc,
                        "Vars": vars_dict,
                }
                
            return results
        
        @staticmethod       
        def get_df_from_csv(
            files_list
            ):
            """
            Read CODA CSV files into a pandas DataFrame. It must be wrote in CODA format.

            Args:
                files_list (list[str]): List of file paths to residual CSVs.

            Returns:
                pd.DataFrame: Combined DataFrame with iteration numbers and residual values.
            """

            df_case = []

            for file_path in files_list:
                with open(file_path, 'r') as f:
                    f.readline()
                    header_line = f.readline()
                column_names = re.findall(r'"(.*?")', header_line)
                column_names = [name.replace('"', '') for name in column_names]
                df = pd.read_csv(
                    file_path, delim_whitespace=True,
                    skiprows=2, names=column_names,
                    dtype = np.float64,
                    )
                df_case.append(df)
            df = pd.concat(df_case, ignore_index=True)
            
            df_iter = pd.DataFrame({"total_iter": np.arange(0,df.shape[0])})
            df = pd.concat([df_iter, df], axis=1)
            
            return df
        
        @staticmethod
        def folder_fmt_to_pattern(folder_fmt: str) -> re.Pattern:
            """
            Convierte un folder_fmt del tipo nombre_{}_nombre_{} en una regex
            sustituyendo {} por números genéricos [-\\d\\.]+
            """
            parts = folder_fmt.split("_")
            regex_parts = []

            for part in parts:
                if "{" in part and "}" in part:
                    regex_parts.append(r"[-\d\.]+")
                else:
                    regex_parts.append(re.escape(part))

            regex = "_".join(regex_parts)
            return re.compile(rf"^{regex}$")
        
    class Weapons:
        
        @staticmethod
        def sort_by_centroid(points:np.ndarray):
            centroid = points.mean(axis=0)
            shifted = points - centroid
            
            N, D = shifted.shape
            
            # PCA (SVD) para obtener el plano principal donde proyectar
            try:
                _, s, vt = np.linalg.svd(shifted, full_matrices=False)
            except np.linalg.LinAlgError:
                order_ptos = np.lexsort(tuple(points[:, i] for i in reversed(range(D))))
                return points[order_ptos], order_ptos.astype(np.int32)

            # Si la segunda componente principal es despreciable -> fallback lexsort
            if s.size < 2 or s[1] < 1e-8 * max(s[0], 1.0):
                order_ptos = np.lexsort(tuple(points[:, i] for i in reversed(range(D))))
                return points[order_ptos], order_ptos.astype(np.int32)

            basis = vt[:2]           # dos vectores principales (2, D)
            proj = shifted @ basis.T # (N, 2)

            # Calcular ángulo en ese plano principal y ordenar por él
            angles = np.arctan2(proj[:, 1], proj[:, 0])
            order_ptos = np.argsort(angles)

            return points[order_ptos], order_ptos.astype(np.int32)

        @staticmethod
        def sort_lexsort(points:np.ndarray):
           
            idx = np.lexsort(tuple(points[:, i] for i in reversed(range(points.shape[1]))))
            return points[idx], idx
            
        @staticmethod
        def sort_closed_curve_by_kdtree(points:np.ndarray, k=10, start_index:int=None, alpha:float=0.7):
            """
            Ordena puntos muestreados sobre una curva cerrada 1D (2D o 3D)
            usando vecinos cercanos y seguimiento por tangente suavizada.

            Parámetros
            ----------
            points : ndarray, shape (N, D)
                Coordenadas de los puntos (D = 2 o 3).
            k : int
                Número de vecinos cercanos (8–12 recomendado).
            start_index : int o None
                Punto inicial. Si None, se elige automáticamente.
            alpha : float
                Peso de la tangente previa (0.6–0.9 recomendado).

            Retorna
            -------
            ordered_points : ndarray, shape (N, D)
                Puntos ordenados.
            order : ndarray, shape (N,)
                Índices originales en el orden seguido.
            """

            from scipy.spatial import KDTree
            
            points = np.asarray(points)
            N = len(points)

            if N < 3:
                return points.copy(), np.arange(N)

            # --------------------------------------------------
            # KDTree y vecinos
            # --------------------------------------------------
            tree = KDTree(points)
            _, neighbors = tree.query(points, k=min(k, N))

            # --------------------------------------------------
            # Selección de punto inicial
            # (punto "extremo" local: mayor distancia media a vecinos)
            # --------------------------------------------------
            if start_index is None:
                mean_dist = np.mean(
                    np.linalg.norm(points[neighbors[:, 1:]] - points[:, None], axis=2),
                    axis=1,
                )
                current = int(np.argmax(mean_dist))
            else:
                current = int(start_index)

            visited = np.zeros(N, dtype=bool)
            order = np.empty(N, dtype=int)

            def unit(v):
                n = np.linalg.norm(v)
                return v / n if n > 0 else v

            # --------------------------------------------------
            # Inicialización: elegir primer vecino coherente
            # --------------------------------------------------
            visited[current] = True
            order[0] = current

            # vecino más cercano para arrancar
            first_neighbors = neighbors[current][1:]
            nxt = min(
                first_neighbors,
                key=lambda j: np.linalg.norm(points[j] - points[current])
            )

            prev_dir = unit(points[nxt] - points[current])
            current = nxt
            visited[current] = True
            order[1] = current

            # --------------------------------------------------
            # Recorrido principal
            # --------------------------------------------------
            for i in range(2, N):
                candidates = [
                    j for j in neighbors[current]
                    if not visited[j] and j != current
                ]

                if not candidates:
                    # fallback: el no visitado más cercano
                    remaining = np.where(~visited)[0]
                    dists = np.linalg.norm(points[remaining] - points[current], axis=1)
                    nxt = remaining[np.argmin(dists)]
                else:
                    def cost(j):
                        v = unit(points[j] - points[current])
                        dot = np.dot(v, prev_dir)

                        # prohibir retroceso
                        if dot < 0.0:
                            return np.inf

                        angle = np.arccos(np.clip(dot, -1.0, 1.0))
                        dist = np.linalg.norm(points[j] - points[current])

                        return angle + 0.2 * dist

                    nxt = min(candidates, key=cost)

                # actualizar tangente suavizada
                new_dir = unit(points[nxt] - points[current])
                prev_dir = unit(alpha * prev_dir + (1 - alpha) * new_dir)

                current = nxt
                visited[current] = True
                order[i] = current

            return points[order], order
           
        @staticmethod
        def sort_points_by_hull_projection(points, alpha:float = 1.5):
            """
            Orders all points by projecting them onto the concave hull
            and sorting by curvilinear abscissa.
            """
            def sort_profile_by_concave_hull(points:np.ndarray, alpha:float=3.0):
      
                tri = Delaunay(points)
            
                edge_count = {}
                for simplex in tri.simplices:
                    for i, j in [(0,1), (1,2), (2,0)]:
                        p1 = points[simplex[i]]
                        p2 = points[simplex[j]]
                        a = np.linalg.norm(p1 - p2)
                        p3 = points[simplex[3 - i - j]]
                        b = np.linalg.norm(p2 - p3)
                        c = np.linalg.norm(p3 - p1)
            
                        s = (a + b + c) / 2.0
                        area = max(s * (s - a) * (s - b) * (s - c), 0.0)
                        if area == 0.0:
                            continue
            
                        circum_r = a * b * c / (4.0 * np.sqrt(area))
            
                        if circum_r < alpha:
                            edge = tuple(sorted((simplex[i], simplex[j])))        # pipeline.model.save(model_file)
                            edge_count[edge] = edge_count.get(edge, 0) + 1
            
                boundary_edges = [e for e, c in edge_count.items() if c == 1]
            
                adjacency = {}
                for i, j in boundary_edges:
                    adjacency.setdefault(i, []).append(j)
                    adjacency.setdefault(j, []).append(i)
            
                start = min(adjacency.keys(), key=lambda i: points[i, 0])
                path = [start]
                prev = None
                current = start
            
                while True:
                    neighbors = adjacency[current]
                    next_pt = neighbors[0] if neighbors[0] != prev else neighbors[1]
                    if next_pt == start:
                        break
                    path.append(next_pt)
                    prev, current = current, next_pt
            
                path.append(start)
                return np.array(path)
            
            hull_indices = sort_profile_by_concave_hull(points, alpha=1.5)
            hull_points = points[hull_indices]
            # Compute cumulative arc-length of the hull
            diffs = np.diff(hull_points, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            s_hull = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        
            # For each point, find closest hull segment
            s_proj = np.zeros(len(points))
        
            for i, p in enumerate(points):
                min_dist = np.inf
                best_s = 0.0
        
                for j in range(len(hull_points) - 1):
                    a = hull_points[j]
                    b = hull_points[j + 1]
                    ab = b - a
                    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
                    proj = a + t * ab
                    d = np.linalg.norm(p - proj)
        
                    if d < min_dist:
                        min_dist = d
                        best_s = s_hull[j] + t * seg_lengths[j]
        
                s_proj[i] = best_s
        
            order = np.argsort(s_proj)
            
            return points[order], order

        @staticmethod
        def finite_diff_derivative(
                X: torch.Tensor,
                f: torch.Tensor,
                order: int = 1) -> torch.Tensor:

            if order < 1:
                raise ValueError("El orden de la derivada debe ser >= 1")

            # forzar doble precisión
            X = X.to(torch.float64)
            f = f.to(torch.float64)

            N, D = X.shape
            derivs = torch.zeros((N, D), dtype=torch.float64, device=X.device)

            eps = 1e-14

            for d in range(D):

                xd = X[:, d]
                g = f.clone()

                for _ in range(order):

                    new_g = torch.zeros_like(g)

                    dx_forward  = xd[2:] - xd[1:-1]
                    dx_backward = xd[1:-1] - xd[:-2]
                    dx = dx_forward + dx_backward

                    dx = torch.where(torch.abs(dx) < eps, eps, dx)

                    new_g[1:-1] = (g[2:] - g[:-2]) / dx

                    dx_left = xd[1] - xd[0]
                    if torch.abs(dx_left) < eps:
                        dx_left = eps

                    new_g[0] = (g[1] - g[0]) / dx_left

                    dx_right = xd[-1] - xd[-2]
                    if torch.abs(dx_right) < eps:
                        dx_right = eps

                    new_g[-1] = (g[-1] - g[-2]) / dx_right

                    g = new_g

                derivs[:, d] = g

            return derivs

        @staticmethod
        def surface_derivative(X, f, order=1):
            """
            Derivadas sobre una curva usando longitud de arco.

            Parameters
            ----------
            X : (N,D) torch.Tensor or np.ndarray
                Coordenadas ordenadas de la curva.
            f : (N,) o (N,M)
                Valores de la función en la curva (M casos opcional).
            order : {1,2}

            Returns
            -------
            df : misma shape que f
                Derivada respecto a la longitud de arco.
            """

            import numpy as np
            import torch

            is_torch = torch.is_tensor(X)

            if is_torch:

                X = X.to(torch.float64)
                f = f.to(torch.float64)

                if f.ndim == 1:
                    f = f[:, None]

                # --- longitud de arco
                dX = X[1:] - X[:-1]
                ds = torch.sqrt(torch.sum(dX**2, dim=1))

                eps = 1e-14
                ds = torch.clamp(ds, min=eps)

                s = torch.zeros(X.shape[0], dtype=torch.float64, device=X.device)
                s[1:] = torch.cumsum(ds, dim=0)

                # --- primera derivada
                df = torch.gradient(f, spacing=(s,), dim=0)[0]

                if order == 1:
                    return df.squeeze()

                # --- segunda derivada
                d2f = torch.gradient(df, spacing=(s,), dim=0)[0]

                return d2f.squeeze()

            else:

                X = np.asarray(X, dtype=np.float64)
                f = np.asarray(f, dtype=np.float64)

                if f.ndim == 1:
                    f = f[:, None]

                # --- longitud de arco
                dX = X[1:] - X[:-1]
                ds = np.sqrt(np.sum(dX**2, axis=1))

                eps = 1e-14
                ds[ds < eps] = eps

                s = np.zeros(X.shape[0])
                s[1:] = np.cumsum(ds)

                # --- primera derivada
                df = np.gradient(f, s, axis=0)

                if order == 1:
                    return df.squeeze()

                # --- segunda derivada
                d2f = np.gradient(df, s, axis=0)

                return d2f.squeeze()
            
        @staticmethod
        def finite_diff_surface_derivative_ant(
                X: torch.Tensor,
                f: torch.Tensor,
                order: int = 1):

            if order not in (1, 2):
                raise ValueError("order must be 1 or 2")

            X = X.to(torch.float64)
            f = f.to(torch.float64)

            N = X.shape[0]
            eps = 1e-14

            # --- longitud de arco ---
            dX = X[1:] - X[:-1]
            ds = torch.sqrt(torch.sum(dX**2, dim=1))

            ds = torch.where(ds < eps, eps, ds)

            s = torch.zeros(N, dtype=torch.float64, device=X.device)
            s[1:] = torch.cumsum(ds, dim=0)

            # --- primera derivada ---
            df = torch.zeros(N, dtype=torch.float64, device=X.device)

            ds_forward  = s[2:] - s[1:-1]
            ds_backward = s[1:-1] - s[:-2]

            ds_tot = ds_forward + ds_backward
            ds_tot = torch.where(ds_tot < eps, eps, ds_tot)

            df[1:-1] = (f[2:] - f[:-2]) / ds_tot

            df[0]  = (f[1] - f[0]) / (s[1] - s[0] + eps)
            df[-1] = (f[-1] - f[-2]) / (s[-1] - s[-2] + eps)

            if order == 1:
                return df
            
            else:

                # --- segunda derivada ---
                d2f = torch.zeros(N, dtype=torch.float64, device=X.device)

                d2f[1:-1] = (
                    2 * (
                        (f[2:] - f[1:-1]) / (ds_forward + eps)
                        - (f[1:-1] - f[:-2]) / (ds_backward + eps)
                    ) / (ds_forward + ds_backward + eps)
                )

                d2f[0] = d2f[1]
                d2f[-1] = d2f[-2]

                return df, d2f

        @staticmethod
        def finite_diff_derivative_ant(
            X: torch.Tensor,
            f: torch.Tensor,
            order: int = 1) -> torch.Tensor:
            """
            Calcula derivadas n-ésimas de f respecto a cada dimensión de X mediante diferencias finitas.
            
            Args:
                X (torch.Tensor): Tensor (N, D) con las variables independientes (ej: [x, z]).
                f (torch.Tensor): Tensor (N,) con la variable dependiente (ej: cp).
                order (int): Orden de la derivada (1 = primera, 2 = segunda, ...).
                
            Returns:
                torch.Tensor: Tensor (N, D) con las derivadas de orden `order` en cada dimensión.
            """
            if order < 1:
                raise ValueError("El orden de la derivada debe ser >= 1")
            
            N, D = X.shape
            derivs = torch.zeros((N, D), dtype=f.dtype, device=f.device)
            
            for d in range(D):
                xd = X[:, d]

                # Paso uniforme aproximado (si no es uniforme, ajusta en cada punto)
                dx = xd[1] - xd[0]

                g = f.clone()
                for _ in range(order):
                    new_g = torch.zeros_like(g)

                    # diferencias centradas en interior
                    new_g[1:-1] = (g[2:] - g[:-2]) / (2 * dx)

                    # forward / backward en los bordes
                    new_g[0]  = (g[1] - g[0]) / dx
                    new_g[-1] = (g[-1] - g[-2]) / dx

                    g = new_g  # para siguiente iteración

                derivs[:, d] = g
            
            return derivs
        
        @staticmethod
        def build_element_neighbors(connectivity):
            """
            Devuelve:
            neighbors[e, f] = índice del elemento vecino por la cara f
                            o -1 si es frontera
            """
            Ne = connectivity.shape[0]
            neighbors = torch.full((Ne, 3), -1, dtype=torch.long)

            # aristas normalizadas (min, max)
            edges = torch.stack([
                connectivity[:, [0, 1]],
                connectivity[:, [1, 2]],
                connectivity[:, [2, 0]],
            ], dim=1)  # [Ne, 3, 2]

            edges_sorted, _ = edges.sort(dim=2)
            edges_flat = edges_sorted.view(-1, 2)

            elem_ids = torch.arange(Ne).repeat_interleave(3)
            face_ids = torch.arange(3).repeat(Ne)

            edge_dict = {}

            for e, f, edge in zip(elem_ids.tolist(),
                                face_ids.tolist(),
                                edges_flat.tolist()):
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
            nodes: torch.Tensor,          # [Nv, 3]
            connectivity: torch.Tensor,   # [Ne, 3]
            neighbors: torch.Tensor,      # [Ne, 3]  (-1 en frontera)
            tensor_out: torch.Tensor,     # [Ne, Nc, 2]
            var_index: int = 0,
            chunk_size: int = 8,
            export_vtk: bool = False,
            vtk_filename: str = "grad.vtu",
            device: torch.device | None = None,
            ):
            """
            Calculates the gradient of a scalar field defined at element centers over a triangular mesh using the Green-Gauss theorem.
            
            Args:
                nodes (torch.Tensor): Vertex coordinates, shape (Nv, 3).
                connectivity (torch.Tensor): Element connectivity, shape (Ne, 3).
                neighbors (torch.Tensor): Neighboring elements per face, shape (Ne, 3), with -1 for boundary faces.
                tensor_out (torch.Tensor): Scalar field at element centers, shape (Ne, Nc, 2).
                var_index (int): Index of the variable in tensor_out to compute the gradient for.
                chunk_size (int): Number of variables to process in each chunk.
                export_vtk (bool): If True, exports the gradient field to a VTK file.
                vtk_filename (str): Filename for the VTK export.
                device (torch.device | None): Device to perform computations on. If None, uses tensor_out's device.
            Returns:
                torch.Tensor: Gradient of the scalar field at element centers, shape (Ne, Nc, 3).
            """
            
            if device is None:
                device = tensor_out.device

            dtype = torch.float32

            nodes = nodes.to(device=device, dtype=dtype)
            connectivity = connectivity.to(device=device).long()
            neighbors = neighbors.to(device=device)
            tensor_out = tensor_out.to(device=device, dtype=dtype)

            cp_elem = tensor_out[:, :, var_index]   # [Ne, Nc]

            Ne, Nc = cp_elem.shape

            tri = nodes[connectivity]     # [Ne, 3, 3]
            v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]

            edges = [(v0, v1), (v1, v2), (v2, v0)]

            # normales de cara (perpendiculares a la arista, en el plano)
            face_normals = []
            face_lengths = []

            normal_elem = torch.cross(v1 - v0, v2 - v0, dim=1)
            normal_elem = normal_elem / torch.linalg.norm(normal_elem, dim=1, keepdim=True)

            for a, b in edges:
                edge_vec = b - a
                L = torch.linalg.norm(edge_vec, dim=1)
                n_f = torch.cross(normal_elem, edge_vec, dim=1)
                face_normals.append(n_f)
                face_lengths.append(L)

            area = 0.5 * torch.linalg.norm(
                torch.cross(v1 - v0, v2 - v0, dim=1), dim=1
            )

            grad_chunks = []

            for c0 in range(0, Nc, chunk_size):
                c1 = min(c0 + chunk_size, Nc)

                grad = torch.zeros((Ne, c1 - c0, 3), device=device, dtype=dtype)

                cp_c = cp_elem[:, c0:c1]

                for f in range(3):
                    nb = neighbors[:, f]
                    cp_nb = torch.where(
                        nb[:, None] >= 0,
                        cp_elem[nb, c0:c1],
                        cp_c
                    )

                    cp_face = 0.5 * (cp_c + cp_nb)

                    grad += cp_face[:, :, None] * face_normals[f][:, None, :] * face_lengths[f][:, None, None]

                grad /= area[:, None, None].clamp_min(1e-14)
                grad_chunks.append(grad.cpu())

            grad = torch.cat(grad_chunks, dim=1)
            
            if export_vtk:
                points = nodes.cpu().numpy()
                cells = np.hstack([
                    np.full((Ne, 1), 3),
                    connectivity.cpu().numpy()
                ]).astype(np.int64)

                celltypes = np.full(Ne, pv.CellType.TRIANGLE, dtype=np.uint8)
                mesh = pv.UnstructuredGrid(cells, celltypes, points)

                for k in range(cp_elem.shape[-1]):
                    mesh.cell_data[f"x_{k}"] = cp_elem[:, k].cpu().numpy()
                    mesh.cell_data[f"grad_x_{k}"] = grad[:, k, :].numpy()
                    # mesh.cell_data["|grad_cp|"] = np.linalg.norm(
                    #     grad_cp[:, k, :].numpy(), axis=1
                    # )

                mesh.save(vtk_filename)
                
            return grad
        
        @staticmethod
        def GMM(
            df_data: pd.DataFrame,
            BIC_study: bool = False,
            groupby: Union[str, list[str], tuple[str]] = None,
            nclusters: int = 2,
            features: list[str] = None,
            save_pictures:bool = True,
            folder_to_save:str = './GMM_study/',
            format_to_save:Literal['csv', 'hdf', 'pkl'] = 'csv',
            n_components_range: range = range(1, 7),
            random_state: int = 42,
            return_metrics_table: bool = False,
            plot_global_analysis: bool = True,
            verbose:bool = False,
            **kwargs,
            ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
            """
            Aplica un modelo Gaussian Mixture (GMM) con análisis de BIC y AIC global.
            Permite obtener el número de clusters óptimo y visualizar mapas de calor.
            """
            from sklearn.mixture import GaussianMixture
            from sklearn.decomposition import PCA
            if verbose:
                print('\n ---------- Starting GMM algorithm ---------- \n')
                print(f'\tFeatures: {features}')
                print(f'\tN Clusters: {nclusters}')
                print(f'\tFiles saved in {folder_to_save}\n')
            if features is None:
                # raise ValueError("Debe especificarse la lista de 'features' (columnas numéricas para el GMM).")
                raise ValueError("The 'features' list (numerical columns for GMM) must be specified.")

            df_result = df_data.copy()
            df_result["clusters_GMM"] = -1 

            if save_pictures:
                os.makedirs(os.path.join(folder_to_save, 'pictures_case'), exist_ok=True)

            group_iter = (
                df_result.groupby(groupby)
                if groupby is not None
                else [(None, df_result)]
            )

            if verbose:
                group_iter = tqdm(
                    group_iter,
                    total=len(group_iter),
                    desc="GMM cases",
                    unit="case",
                    leave=True
                )


            # --- Registro global de métricas ---
            metrics_records = []
            if BIC_study:
                if verbose:
                    print(f'BIC study on coming per case {groupby}\n')
                if save_pictures:
                    if verbose:
                        print(f'Saving pictures in {os.path.join(folder_to_save, "pictures_case")}')    
        
            for group_key, grp in group_iter:
                if verbose and hasattr(group_iter, "set_postfix"):
                    group_iter.set_postfix(case=group_key)

                X = grp[features].values

                scaler = StandardScaler()
                X = scaler.fit_transform(X)


                if X.shape[0] < nclusters:
                    print(f"[WARN] Group {group_key}: insufficient samples ({X.shape[0]}) for {nclusters} clusters.")
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
                            random_state=random_state
                        )
                        try:
                            gmm_test.fit(X)
                            bic_val = gmm_test.bic(X)
                            aic_val = gmm_test.aic(X)
                        except ValueError:
                            bic_val = np.inf
                            aic_val = np.inf

                        bics.append(bic_val)
                        aics.append(aic_val)

                        metrics_records.append({
                            "group": group_key,
                            "n_components": n,
                            "BIC": bic_val,
                            "AIC": aic_val,
                        })

                    # Identificar mínimos
                    best_bic_n = n_components_range[np.argmin(bics)]
                    best_aic_n = n_components_range[np.argmin(aics)]

                    # Guardar columnas resumen por grupo
                    df_result.loc[grp.index, "BIC_best_n"] = best_bic_n
                    df_result.loc[grp.index, "AIC_best_n"] = best_aic_n

                # --- Entrenamiento final con nclusters ---
                gmm_final = GaussianMixture(
                    n_components=nclusters,
                    covariance_type=kwargs.get("covariance_type", "diag"),
                    max_iter=kwargs.get("max_iter", 200),
                    init_params=kwargs.get("init_params", "kmeans"),
                    reg_covar=kwargs.get("reg_covar", 1e-6),
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items() if k not in ["covariance_type", "max_iter", "init_params", "reg_covar"]}
                )
                labels = gmm_final.fit_predict(X)
                df_result.loc[grp.index, "clusters_GMM"] = labels

                # if plot_each_one:
                #     # Figura con scatter plot de clusters
                #     plt.figure(figsize=(6, 5))
                #     if len(features) >= 2:
                #         scatter = plt.scatter(
                #             X[:, 0], X[:, 1],
                #             c=labels, cmap="viridis", s=30, edgecolor='k'
                #         )
                #         plt.title(f"GMM Clusters ({group_key})" if group_key else "GMM Clusters")
                #         plt.xlabel(features[0])
                #         plt.ylabel(features[1])
                #         plt.colorbar(scatter, label="Cluster ID")
                #         plt.grid(True)
                #         filename = (
                #             f"GMM_clusters_{'_'.join(map(str, group_key))}.png" if group_key is not None else "GMM_clusters_global.png"
                #         )
                #         if save_dir:
                #             plt.savefig(os.path.join('./GMM_study/pictures_case/', filename), dpi=150, bbox_inches="tight")
                #             plt.close()
                #         else:
                #             plt.show()
                #     elif len(features) == 1:
                #         plt.hist(X[:, 0], bins=30, color="steelblue", alpha=0.7, edgecolor='black')
                #         plt.title(f"GMM Clusters ({group_key})" if group_key else "GMM Clusters")
                #         plt.xlabel(features[0])
                #         plt.ylabel("Frequency")
                #         plt.grid(True)
                #         filename = (
                #             f"GMM_clusters_{'_'.join(map(str, group_key))}.png"
                #             if group_key is not None
                #             else "GMM_clusters_global.png"
                #         )
                #         if save_dir:
                #             plt.savefig(os.path.join('./GMM_study/pictures_case/', filename), dpi=150, bbox_inches="tight")
                #             plt.close()
                #         else:
                #             plt.show()
                                
                # --- Guardar resumen BIC por grupo ---
                if BIC_study and save_pictures:
                    min_bic = min(bics)
                    best_n = n_components_range[np.argmin(bics)]
                    df_result.loc[grp.index, "BIC_min"] = min_bic
                    df_result.loc[grp.index, "BIC_opt_n"] = best_n

                    # --- Gráficos BIC + scatter ---
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                    # (1) Curva BIC
                    axes[0].plot(list(n_components_range), bics, marker="o", color="steelblue")
                    axes[0].set_title(f"BIC evolution ({group_key})" if group_key else "BIC evolution")
                    axes[0].set_xlabel("Number of components")
                    axes[0].set_ylabel("BIC")
                    axes[0].grid(True)

                    # (2) Scatter clusters finales
                    if len(features) > 2:                              
                        pca = PCA(n_components=2)
                        X_plot = pca.fit_transform(X)

                        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
                        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"

                    elif len(features) == 2:
                        X_plot = X
                        xlabel, ylabel = features[0], features[1]

                    scatter = plt.scatter(
                        X_plot[:, 0],
                        X_plot[:, 1],
                        c=labels,
                        cmap="viridis",
                        s=30,
                        edgecolor="k"
                    )

                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.title(f"GMM Clusters ({group_key})" if group_key else "GMM Clusters")
                    plt.colorbar(scatter, label="Cluster ID")
                    plt.grid(True)


                    fig.suptitle(f"GMM Study — Group: {group_key}" if group_key else "GMM Study", fontsize=12)

                    filename = (
                        f"GMM_{'_'.join(map(str, group_key))}.png"
                        if group_key is not None
                        else "GMM_global.png"
                    )
                    plt.savefig(os.path.join(folder_to_save, "pictures_case", filename), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                        
            print("GMM clustering completed.")

            # --- Convertir métricas globales a DataFrame ---
            df_metrics = pd.DataFrame(metrics_records) if metrics_records else pd.DataFrame()
            df_metrics = df_metrics.replace([np.inf, -np.inf], np.nan)

            # --- Análisis y gráficos globales ---
            if BIC_study and plot_global_analysis and not df_metrics.empty:

                print("Analyzing global BIC and AIC results...")

                # --- Boxplot global de BIC y AIC (solo matplotlib) ---
                n_values = sorted(df_metrics["n_components"].unique())

                bic_data = [
                    df_metrics.loc[df_metrics["n_components"] == n, "BIC"].values
                    for n in n_values
                ]

                aic_data = [
                    df_metrics.loc[df_metrics["n_components"] == n, "AIC"].values
                    for n in n_values
                ]

                fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

                # --- Estadísticos para anotaciones ---
                bic_means = np.array([np.mean(b) for b in bic_data])
                aic_means = np.array([np.mean(a) for a in aic_data])

                best_bic_n = n_values[np.argmin(bic_means)]
                best_aic_n = n_values[np.argmin(aic_means)]


                # --- BIC boxplot ---
                axes[0].boxplot(
                    bic_data,
                    positions=n_values,
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6),
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor="gray", alpha=0.5)
                )
                axes[0].axvline(
                    best_bic_n,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Best mean BIC = {best_bic_n}"
                )
                axes[0].legend()

                axes[0].set_title("BIC distribution vs number of clusters")
                axes[0].set_xlabel("Number of components")
                axes[0].set_ylabel("BIC")
                axes[0].grid(True, linestyle="--", alpha=0.4)

                # --- AIC boxplot ---
                axes[1].boxplot(
                    aic_data,
                    positions=n_values,
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor="orange", alpha=0.6),
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor="gray", alpha=0.5)
                )
                axes[1].axvline(
                    best_aic_n,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Best mean AIC = {best_aic_n}"
                )
                axes[1].legend()

                axes[1].set_title("AIC distribution vs number of clusters")
                axes[1].set_xlabel("Number of components")
                axes[1].set_ylabel("AIC")
                axes[1].grid(True, linestyle="--", alpha=0.4)

                fig.suptitle("Global GMM model selection (BIC / AIC)", fontsize=13)

                plt.tight_layout()
                plt.savefig(os.path.join(folder_to_save, 'global_BIC_AIC_boxplot.png'), dpi=150, bbox_inches="tight")
                plt.show()

                # --- Tabla resumen BIC/AIC por número de clusters ---
                df_bic_aic_summary = (
                    df_metrics
                    .groupby("n_components")
                    .agg({
                        "BIC": ["mean", "median", "std", "min", "max"],
                        "AIC": ["mean", "median", "std", "min", "max"],
                    })
                    .reset_index()
                )

                # Aplanar nombres de columnas
                df_bic_aic_summary.columns = [
                    "n_components",
                    "BIC_mean", "BIC_median", "BIC_std", "BIC_min", "BIC_max",
                    "AIC_mean", "AIC_median", "AIC_std", "AIC_min", "AIC_max",
                ]


                # Guardar a disco
                df_bic_aic_summary.to_csv(
                    os.path.join(folder_to_save,"GMM_BIC_AIC_summary.csv"),
                    sep=';',
                    index=False
                )


                # --- Distribución global del BIC ---
                plt.figure(figsize=(6, 4))
                sns.histplot(df_metrics["BIC"], bins=30, kde=True, color="steelblue", alpha=0.7)
                plt.title("Global distribution of BIC")
                plt.xlabel("BIC value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(folder_to_save, 'BIC_distribution.png'))
                plt.show()

                # --- Mapa de calor del número óptimo ---
                if groupby and isinstance(groupby, (list, tuple)) and len(groupby) == 2:
                    df_opt = (
                        df_result
                        .groupby(groupby)[["BIC_best_n"]]
                        .first()
                        .reset_index()
                        .pivot(index=groupby[0], columns=groupby[1], values="BIC_best_n")
                    )

                    # --- Ajustes visuales del heatmap ---
                    plt.figure(figsize=(9, 7))
                    n_rows, n_cols = df_opt.shape
                    annot_enable = n_rows * n_cols <= 225  # solo si < 15x15
                    from matplotlib.patches import Patch
                    cluster_vals = np.sort(df_opt.stack().dropna().unique())
                    n_clusters_vals = len(cluster_vals)
                    cmap = plt.cm.get_cmap("tab10", n_clusters_vals)
                    
                    norm = mcolors.BoundaryNorm(
                        boundaries=np.arange(cluster_vals.min() - 0.5, cluster_vals.max() + 1.5),
                        ncolors=n_clusters_vals
                    )
                    ax = sns.heatmap(
                        df_opt,
                        annot=annot_enable,
                        fmt=".0f",
                        cmap=cmap,
                        norm=norm,
                        cbar=False,
                        linewidths=0,
                        linecolor="gray",
                        annot_kws={"size": 8, "color": "black"}
                    )

                    legend_elements = [
                        Patch(facecolor=cmap(i), edgecolor="black", label=f"k = {int(k)}")
                        for i, k in enumerate(cluster_vals)
                    ]

                    ax.legend(
                        handles=legend_elements,
                        title="Optimal clusters (BIC)",
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1),
                        borderaxespad=0.0
                    )

                    plt.title("Heatmap — Optimal number of clusters (BIC)", fontsize=12)
                    plt.xlabel(groupby[1])
                    plt.ylabel(groupby[0])
                    plt.xticks(rotation=45, ha="right")
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    plt.savefig(os.path.join(folder_to_save, 'heatmap_optimal_clusters.png'))
                    plt.show()

            if format_to_save == 'csv':
                df_result.to_csv(
                    os.path.join(folder_to_save, 'df_data_complete.csv'),
                    sep=';', header=True, index=True
                    )
            elif format_to_save == 'hdf':
                df_result.to_hdf(
                    os.path.join(folder_to_save, 'df_data_complete.h5'),
                    key=f'df_n_{nclusters}',
                    mode='a',
                    complevel=0, index=True
                    )
            elif format_to_save == 'pkl':
                df_result.to_pickle(
                    os.path.join(folder_to_save, 'df_data_complete.pkl')
                    )
            else:
                print('Dataframe with clusters not saved because chosen format not supported.')
                
            if return_metrics_table:
                if not BIC_study:
                    print("WARNING: df_metrics only works if BIC_study is activated.")
                else:# 
                    df_metrics.to_csv(os.path.join(folder_to_save, 'df_metrics.csv'), sep=';', header=True, index=True)
                    
                return df_result, df_metrics
            else:
                # df_result.to_csv(os.path.join(folder_to_save, 'df_data_complete.csv'), sep=';', header=True, index=True)
                return df_result

    class DictVisualizer:
        
        @staticmethod
        def _simplify(obj):
            import torch  # se importa aquí para no ensuciar el global
            if isinstance(obj, torch.Tensor):
                return f"Torch Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype})"
            
            elif isinstance(obj, np.ndarray):
                return f"Numpy Array(shape={tuple(obj.shape)}, dtype={obj.dtype})"
            
            elif isinstance(obj, pd.DataFrame):
                return f"DataFrame(shape={obj.shape}, columns={list(obj.columns)})"
            
            elif isinstance(obj, pd.Series):
                return f"Series(length={len(obj)}, dtype={obj.dtype})"
            
            elif isinstance(obj, pv.UnstructuredGrid):
                return f"UnstructuredGrid(points={obj.points.shape}, cells={obj.cells.shape})"
            
            elif isinstance(obj, pv.pyvista_ndarray):
                return f"PyVistaArray(shape={obj.shape}, dtype={obj.dtype})"
            
            elif isinstance(obj, set):
                return {__class__._simplify(v) for v in obj}
            
            elif isinstance(obj, dict):
                return {k: __class__._simplify(v) for k, v in obj.items()}
            
            elif isinstance(obj, list):
                return [__class__._simplify(v) for v in obj]
            
            return str(obj)

        @staticmethod
        def pretty_print(d, depth=2, output_file:Union[str, None]=None):
            """
            Pretty print dictionary.
            
            Parameters
            ----------
            d : dict
            depth : int
                Depth for pprint
            output_file : str or None
                If provided, saves output to this file instead of printing.
            """
            from pprint import pformat
            simplified = SAM.DictVisualizer._simplify(d)
            formatted = pformat(simplified, depth=depth)

            if not bool(output_file):
                print(formatted)
            else:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(formatted)

        @staticmethod
        def rich_tree(d):
            from rich.tree import Tree
            from rich.console import Console

            def build_tree(data, tree):
                for k, v in data.items():
                    branch = tree.add(str(k))
                    if isinstance(v, dict):
                        build_tree(v, branch)
                    else:
                        branch.add(str(v))

            root = Tree("root")
            build_tree(__class__._simplify(d), root)
            console = Console()
            console.print(root)

        @staticmethod
        def plot_graph(d):
            import networkx as nx
            import matplotlib.pyplot as plt

            def dict_to_graph(data, G=None, parent=None):
                if G is None:
                    G = nx.DiGraph()
                for k, v in data.items():
                    G.add_node(k)
                    if parent:
                        G.add_edge(parent, k)
                    if isinstance(v, dict):
                        dict_to_graph(v, G, k)
                    else:
                        node_val = str(v)
                        G.add_node(node_val)
                        G.add_edge(k, node_val)
                return G

            G = dict_to_graph(__class__._simplify(d))
            plt.figure(figsize=(8, 6))
            nx.draw(G, with_labels=True, font_size=8, node_size=2000, node_color="lightblue")
            plt.show()