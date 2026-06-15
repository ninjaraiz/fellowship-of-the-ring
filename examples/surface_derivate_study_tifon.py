pc = 'FotR_ex'
import os
config = {
    'root_dir':{
        'laptop': '/home/migueljaraiz/anaconda3/repos/',
        'cluster': '/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_extended/outputs/',
        'pc_pro': '/home/ninjaraiz/anaconda3/repos/'
    },
    'folder_to_save':{
        'laptop': '/home/migueljaraiz/anaconda3/repos/GMM_TIFON/',
        'cluster': '/home/m.jaraiz/Documentos/GMM/GMM_TIFON/',
        'FotR_ex': './example_GMM/'
    },
    'pylom_path': {
        'laptop': '/home/migueljaraiz/anaconda3/repos/pyLowOrder',
        'cluster': '/home/m.jaraiz/repos/pyLowOrder',
        'pc_pro': '/home/ninjaraiz/anaconda3/repos/pyLowOrder'
    }   
}

for key in ['root_dir', 'pylom_path']:
    config[key]['FotR_ex'] = [path for _, path in config[key].items()]


import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

try:
    import pyLOM
    print('Entorno con pyLOM instalado')
except ImportError as e:
    print(e)
    print('Imported by local repository')

    if pc == 'FotR_ex':
        for path in config['pylom_path'][pc]:
            if os.path.exists(path):
                sys.path.append(path)
                print(f'Found pylom path: {path}')
                break
        print('Not found pylom path')
    else:
        sys.path.append(config['pylom_path'][pc])

from FotR import FRODO
import matplotlib.pyplot as plt

import numpy as np, torch

def symlog(x, linthresh=1e-3):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log10(1 + torch.abs(x) / linthresh)
    elif isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x) / linthresh)

for folder_path in config['root_dir'][pc]:
    if os.path.exists(folder_path):
        config['root_dir'][pc] = folder_path
        print('Folder root_dir found succesfull.')
        break
if isinstance(config['root_dir'][pc], list):
    raise LookupError('Folder root_dir not found with this configuration.')
    
db = FRODO(
    root_dir = config['root_dir'][pc],
    format = 'PYLOM',
    file = 'CADGroup_3_completo_stage_1.h5')

db.extract_inputs(
    keys_inputs={
        'ptos': 'xyz',    # coordenadas del mallado → data_dict['inputs']['ptos']
        'aoa': 'aoa',   # variable paramétrica   → data_dict['inputs']['aoa']
        'mach': 'M',   # variable paramétrica   → data_dict['inputs']['M']
    },
    keys_aux={},
)

db.extract_outputs(
    keys_outputs={
        'cp': 'BoundaryValues_CoefPressure',
        # 'gradrho': 'AugStateGrad_DensityGradient_interp',  # field del Dataset → data_dict['outputs']['cp']
        # 'gradT': 'AugStateGrad_TemperatureGradient_interp',
        'T': 'AugState_Temperature_interp',
        'rho': 'State_Density_interp'
        }
    
)

# Acceder a los datos
xyz = db.sets.get_xyz()           # (npoints, 3)
aoa   = db.sets.get_variable('aoa') # (500,)
mach = db.sets.get_variable('mach')   # (500,)
cp  = db.sets.get_field('cp')     # (npoints, 500)

T = db.sets.get_field('T')
rho = db.sets.get_field('rho')

# gradrhox = db.sets.get_field('gradrho')[0, :, :]#np.linalg.norm(db.sets.get_field('gradrho'), ord=2,axis=0)
# gradTx = db.sets.get_field('gradT')[0, :, :]#np.linalg.norm(db.sets.get_field('gradT'), ord=2,axis=0)


from FotR import SAM
xyz_sort, order_sort = SAM.Weapons.sort_by_centroid(xyz)
cp_sort = cp[order_sort, :]

T_sort = T[order_sort, :]
rho_sort = rho[order_sort, :]

# gradrhox_sort = gradrhox[order_sort, :]
# gradTx_sort = gradTx[order_sort, :]

sep = 1
n_clusters = 2
def segmentar_tensor(tensor, sep):
    return torch.from_numpy(tensor[::sep, :]) if sep != 1 else torch.from_numpy(tensor)

tensor_ptos = segmentar_tensor(xyz_sort, sep)
tensor_cp = segmentar_tensor(cp_sort, sep)

tensor_rho = segmentar_tensor(rho_sort, sep)
tensor_T = segmentar_tensor(T, sep)

# tensor_gradrhox = segmentar_tensor(gradrhox_sort, sep)
# tensor_gradTx = segmentar_tensor(gradTx_sort, sep)

from scipy.signal import savgol_filter

activate_filter = False

window_length = 101   # debe ser impar
polyorder = 2

def SGS(tensor, window_length=21, polyorder=3):

    def _filter(x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()

            if x_np.ndim == 1:
                y = savgol_filter(
                    x_np,
                    window_length=window_length,
                    polyorder=polyorder
                )
            else:
                y = savgol_filter(
                    x_np,
                    window_length=window_length,
                    polyorder=polyorder,
                    axis=0
                )

            return torch.from_numpy(y).to(x.dtype)

        elif isinstance(x, np.ndarray):

            if x.ndim == 1:
                return savgol_filter(
                    x,
                    window_length=window_length,
                    polyorder=polyorder
                )
            else:
                return savgol_filter(
                    x,
                    window_length=window_length,
                    polyorder=polyorder,
                    axis=0
                )

        else:
            raise ValueError(
                "Input must be a torch tensor or numpy array."
            )

    if isinstance(tensor, (list, tuple)):
        return [_filter(t) for t in tensor]
    else:
        return _filter(tensor)

if activate_filter:
    (
        tensor_cp_filtered,
        # tensor_gradrhox_filtered,
        # tensor_gradTx_filtered
        tensor_rho_filtered,
        tensor_T_filtered
    ) = SGS(
        [
            tensor_cp,
            tensor_T,
            tensor_rho,
            # tensor_gradrhox,
            # tensor_gradTx
            ],
        window_length=21,
        polyorder=3
    )
else:
    (
        tensor_cp_filtered,
        tensor_rho_filtered,
        tensor_T_filtered,
        
        # tensor_gradrhox_filtered,
        # tensor_gradTx_filtered
    ) = (
        tensor_cp,
        tensor_T,
        tensor_rho,
        
        # tensor_gradrhox,
        # tensor_gradTx
    )
    
# def derivar_y_filtrar(tensor, tensor_ptos, d_tensor_ds, order, stencil, poly_order):
#     d_tensor_ds = torch.zeros(tensor.shape, dtype=torch.float64)
#     for case in range(tensor.shape[-1]):
#         d_tensor_ds[:, case] = SAM.Weapons.surface_derivative(
#                 X=tensor_ptos,
#                 f=tensor_cp_filtered[:, case],
#                 order=order,
#                 stencil_width=stencil,   
#                 poly_order=poly_order,
#             )
    
scale_log = True
for stencil in range(10, 420, 20):
    # ── derivada por longitud de arco ─────────────────────────────────────────
    dcp_ds = torch.zeros(tensor_cp_filtered.shape, dtype=torch.float64)
    dcp2_ds = torch.zeros(tensor_cp_filtered.shape, dtype=torch.float64)
    
    drho_ds = torch.zeros(tensor_rho_filtered.shape, dtype=torch.float64)
    drho2_ds = torch.zeros(tensor_rho_filtered.shape, dtype=torch.float64)
    
    dT_ds = torch.zeros(tensor_T_filtered.shape, dtype=torch.float64)
    dT2_ds = torch.zeros(tensor_T_filtered.shape, dtype=torch.float64)
    
    for case in range(tensor_cp_filtered.shape[1]):
        dcp_ds[:, case] = SAM.Weapons.surface_derivative(
            X=tensor_ptos,
            f=tensor_cp_filtered[:, case],
            order=1,
            stencil_width=stencil,   
            poly_order=polyorder,
        )
        
        drho_ds[:, case] = SAM.Weapons.surface_derivative(
            X=tensor_ptos,
            f=tensor_rho_filtered[:, case],
            order=1,
            stencil_width=stencil,   
            poly_order=polyorder,
        )
        
        dT_ds[:, case] = SAM.Weapons.surface_derivative(
            X=tensor_ptos,
            f=tensor_T_filtered[:, case],
            order=1,
            stencil_width=stencil,   
            poly_order=polyorder,
        )
        
    dcp_ds_filtered = SGS(dcp_ds, window_length=5, polyorder=2) if activate_filter else dcp_ds
    drho_ds_filtered = SGS(drho_ds, window_length=5, polyorder=2) if activate_filter else drho_ds
    dT_ds_filtered = SGS(dT_ds, window_length=5, polyorder=2) if activate_filter else dT_ds
    
    for case in range(tensor_cp_filtered.shape[1]):
        dcp2_ds[:, case] = SAM.Weapons.surface_derivative(
            X=tensor_ptos,
            f=dcp_ds_filtered[:, case],
            order=1,
            stencil_width=stencil,
            poly_order=polyorder,
        )

        drho2_ds[:, case] = SAM.Weapons.surface_derivative(
            X=tensor_ptos,
            f=drho_ds_filtered[:, case],
            order=1,
            stencil_width=stencil,   
            poly_order=polyorder,
        )
        
        dT2_ds[:, case] = SAM.Weapons.surface_derivative(
            X=tensor_ptos,
            f=dT_ds_filtered[:, case],
            order=1,
            stencil_width=stencil,   
            poly_order=polyorder,
        )
        
    dcp_ds_log = symlog(dcp_ds_filtered, linthresh=1e-4) if scale_log else dcp_ds_filtered
    dcp2_ds_log = symlog(dcp2_ds, linthresh=1e-4) if scale_log else dcp2_ds
    
    drho_ds_log = symlog(drho_ds_filtered, linthresh=1e-4) if scale_log else drho_ds_filtered
    drho2_ds_log = symlog(drho2_ds, linthresh=1e-4) if scale_log else drho2_ds
    
    dT_ds_log = symlog(dT_ds_filtered, linthresh=1e-4) if scale_log else dT_ds_filtered
    dT2_ds = symlog(dT2_ds, linthresh=1e-4) if scale_log else dT2_ds
    
    
    # gradrhox_log = symlog(tensor_gradrhox_filtered, linthresh=1e-4) if scale_log else tensor_gradrhox_filtered
    # gradTx_log = symlog(tensor_gradTx_filtered, linthresh=1e-4) if scale_log else tensor_gradTx_filtered
    
    db_one = db.copy()
    db_one.sets.add_aux(
        array_name = 'dcp_ds_log',
        array = dcp_ds_log.numpy(),
        notes = 'Log dcp_ds')

    db_one.sets.add_aux(
        array_name = 'dcp2_ds_log',
        array = dcp2_ds_log.numpy(),
        notes = 'Log dcp2_ds')

    db_one.sets.add_aux(
        array_name = 'drho_ds_log',
        array = drho_ds_log.numpy(),
        notes = 'Log drho_ds')

    db_one.sets.add_aux(
        array_name = 'drho2_ds_log',
        array = drho2_ds_log.numpy(),
        notes = 'Log drho2_ds')
    
    db_one.sets.add_aux(
        array_name = 'dT_ds_log',
        array = dT_ds_log.numpy(),
        notes = 'Log dT_ds')

    db_one.sets.add_aux(
        array_name = 'dT2_ds',
        array = dT2_ds.numpy(),
        notes = 'Log dT2_ds')
    # db_one.sets.add_aux(
    #     array_name = 'gradrhox_log',
    #     array = gradrhox_log.numpy(),
    #     notes = 'Log gradrhox'
    # )

    # db_one.sets.add_aux(
    #     array_name = 'gradTx_log',
    #     array = gradTx_log.numpy(),
    #     notes = 'Log gradTx'
    # )

    db_one.data_dict['inputs']['ptos'] = tensor_ptos.numpy()
    db_one.data_dict['outputs']['cp'] = tensor_cp_filtered.numpy()
    [db_one.data_dict['outputs'].pop(key, None) for key in ['gradT', 'gradrho']]
    db_one.sets.create_jset(verbose=False)
    # display(db_one.df_data)

    db_one.sets.create_jset(verbose=False)

    features = ['dcp_ds_log', 'dcp2_ds_log', 'drho_ds_log', 'drho2_ds_log', 'dT_ds_log', 'dT2_ds'] # , 'gradrhox_log', 'gradTx_log'
    # features = ['dcp_ds_log', 'dcp2_ds_log']
    folder_name = '_'.join(features)
    df_data_complete, _ = SAM.Weapons.GMM(
        df_data=db_one.df_data,
        BIC_study=True,
        groupby=["aoa", "mach"],
        nclusters=n_clusters,
        features=features,
        save_pictures=True,
        folder_to_save=os.path.join(config['folder_to_save'][pc], f'{folder_name}/sep_{sep}/c_{n_clusters}/s_{stencil}'),
        n_components_range=range(1, 5),
        covariance_type="diag",
        max_iter=300,
        random_state=42,
        return_metrics_table=True,
        plot_global_analysis=True,
        verbose = True
    )
    # case = 20
    # scale = 7
    # markersize_dcp = 1
    # def get_column_from_df(column, case, df):
    #     if isinstance(column, (list, tuple)):
    #         lista = []
    #         for col in column:
    #             lista.append(df.groupby(['aoa', 'mach']).get_group((df['aoa'].unique()[case], df['mach'].unique()[case]))[col])
    #         return lista
        
    #     if isinstance(column, str):
    #         serie = df_data_complete.groupby(['aoa', 'mach']).get_group((df_data_complete['aoa'].unique()[case], df_data_complete['mach'].unique()[case]))[column]
    #         return serie
        
    # [x, z, cp, clusters] = get_column_from_df(['x', 'z', 'cp', 'clusters_GMM'], case, df_data_complete)

    # fig, ax = plt.subplots(2, 1, figsize=(12, 2*6))
    # # ax = ax.flatten()
    # ax[0].scatter(
    #     x, z,
    #     c='black', s=1
    # )
    # ax00 = ax[0].twinx()
    # ax00.scatter(
    #     x, dcp_ds_filtered[:, case], c='blue', s=markersize_dcp
    # )

    # ax01 = ax[0].twinx()
    # ax01.scatter(
    #     x, dcp2_ds[:, case], c='red', s=markersize_dcp
    # )
    # ax[0].set_ylim(bottom = z.min()*scale, top = z.max()*scale)
    # # Poner un tercer eje a la izquierda con cp
    # ax_cp = ax[0].twinx()
    # ax_cp.scatter(
    #     x, tensor_cp_filtered[:, case], c='green', s=markersize_dcp
    # )
    # ax_cp.spines['left'].set_position(('outward', 60))
    # ax_cp.spines['left'].set_color('green')
    # ax_cp.tick_params(axis='y', colors='green')
    # ax_cp.invert_yaxis()
    # # Arreglar ejes de los twinx
    # ax00.set_yscale('log')
    # ax01.set_yscale('log')
    # #separar ejes secundarios y poner del mismo color que los puntos
    # ax00.spines['right'].set_position(('outward', 60))
    # ax01.spines['right'].set_position(('outward', 120))
    # ax00.spines['right'].set_color('blue')
    # ax01.spines['right'].set_color('red')
    # ax00.tick_params(axis='y', colors='blue')
    # ax01.tick_params(axis='y', colors='red')
    # ax[0].set_xlabel('x')
    # ax[0].set_ylabel('z')
    # ax00.set_ylabel('dcp/ds', color='blue')
    # ax01.set_ylabel('d2cp/ds2', color='red')
    # ax[0].set_title(f'Case {case}')

    # ax[1].scatter(
    #     x, z, c='black', s=markersize_dcp, alpha=0.7)
    # ax[1].set_xlabel('x')
    # ax[1].set_ylabel('z')
    # ax[1].set_ylim(z.min() - 0.1, z.max() + 0.1)
    # ax10 = ax[1].twinx()
    # ax10.scatter(
    #     x, cp, c=clusters, s=markersize_dcp, alpha=0.7, cmap='viridis')
    # ax10.set_ylabel('cP')
    # ax10.invert_yaxis()

    # fig.suptitle(f'Case {case} - features: {features} - sep: {sep}')
    # fig.savefig(os.path.join(config['folder_to_save'][pc], f'{folder_name}/sep_{sep}/c_{n_clusters}/example_case20_s_{stencil}.png'))