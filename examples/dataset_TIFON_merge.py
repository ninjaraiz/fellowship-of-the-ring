import sys
try:
    import pyLOM
except ImportError as e:
    print(f'Error importing pyLOM: {e}')
    print('Importing with local repository')
    sys.path.append('/home/m.jaraiz/repos/pyLowOrder/')
from FotR import FRODO
# Anulamos la escritura de los prints en el buffer. El comando "sys.stdout.flush()" vacía el buffer. Puede ser una opción mejor si no se quiere gastar tantos recursos.
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

def read_db_CODA(datafolder, case_idx, interpolate_vol2surf=True):
    db = FRODO(root_dir = datafolder, format = 'CODA', initial_parse = True)
    
    if interpolate_vol2surf:
        for id, type, var_excluded in zip([3, 4], ['surface', 'volume'], [[f'BoundaryValues_CoefSkinFriction{coord}' for coord in ['X', 'Y', 'Z']], []]):
            print(f'\t CAD {id}')
            db.extract_inputs(
                id_groups = (id,),
                cases_idx = case_idx,
                vtu_type=type,
                verbose=False
                )

            for stage in [0, 1]:
                print(f'\n\t\t stage {stage}')
                db.extract_outputs(
                    id_groups=(id,),
                    stage=stage, cases_idx = case_idx,
                    var_name_excluded = var_excluded if stage==0 else [],
                    vtu_type=type,
                    )
        
        db.sets.interpolate_vol2surf(
            vol_group = '4',
            surf_group = '3',
            stage = str(stage),
            vars = 'all',
        )
        
        db.sets.add_to_data_dict(
            arr = db.data_dict['CADGroup_3']['Coord'],
            id_group = 4,
            array_name = 'Airfoil'
        )
        
    else:
        db.extract_inputs(
            id_groups = (3,),
            cases_idx = case_idx,
            vtu_type='surface',
            verbose=False
            )

        for stage in [0, 1]:
            db.extract_outputs(
                id_groups=(3,),
                stage=stage, cases_idx = case_idx,
                var_name_excluded = [f'BoundaryValues_CoefSkinFriction{coord}' for coord in ['X', 'Y', 'Z']] if stage == 0 else [],
                vtu_type='surface',
                )
            
        db.sets.add_to_data_dict(
            arr = db.data_dict['CADGroup_3']['Coord'],
            id_group = 4,
            array_name = 'Airfoil'
        )
        
    return db

from typing import Union
import os
import gc
import pandas as pd

DATA_DIR = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON'

def juntar_db(list_folders:Union[tuple[str], list[str]], case_idx:Union[tuple, list], interpolate_vol2surf:bool = True):
    list_db = []
    
    for folder, cases in zip(list_folders, case_idx):
        print(f'\nLeyendo {folder}')
        db = read_db_CODA(datafolder = folder, case_idx=cases, interpolate_vol2surf=interpolate_vol2surf)
        print('CADGroups: ', db.data_dict.keys())
        print('Matriz FlCc: ', db.data_dict['CADGroup_3']['FlCc'].shape, db.data_dict['CADGroup_4']['FlCc'].shape)
        for g in ('CADGroup_3', 'CADGroup_4'):
            if db.data_dict[g]['FlCc'].shape[-1] > 2:
                db.data_dict[g]['FlCc'] = db.data_dict[g]['FlCc'][:, :2]
        n = db.data_dict['CADGroup_3']['FlCc'].shape[-1]
        db.metadata['design_vars'] = list(db.metadata['design_vars'])[:n]
        list_db.append(db)
    return list_db

def fusionar_y_exportar(list_db:list, cad:int, name:str):
    """
    Junta el CADGroup `cad` de todas las bases, lo exporta a pyLOM y libera la memoria.
    Tras la fusión se borra ese grupo de las bases de origen, así que cada grupo
    solo se puede fusionar una vez.
    """
    root_dir = os.path.join(DATA_DIR, name)
    print(f'\nJuntando CAD{cad} en {root_dir}\n')
    db_merged = FRODO.merge_datasets(
        root_dir=root_dir,
        name = name,
        sources = [(db, str(cad)) for db in list_db],
        new_group_id=f'{cad}_completo',
        k=4,
        mesh_ref=0,
        cache=True,
        get_df_metrics_attr={
            'var_metrics': ['CoefLift', 'CoefDrag', 'CoefMomentY'],
            'iter_var': 1000,
            'save' : False
        }
    )

    # Los datos de este grupo ya están en db_merged: se liberan de las bases de origen
    for db in list_db:
        db.sets.remove_data(id_group=str(cad), stage=None, verbose=False)
    gc.collect()

    print(f'\nExportando CAD{cad} a PYLOM\n')
    for stage in [0, 1]:
        db_merged.sets.create_NN_pylom(id_groups = f'{cad}_completo', stage=str(stage), idx_to_print='all', save_path = os.path.join(root_dir, 'outputs'))
        # El stage ya está en el .h5: se libera antes de construir el siguiente
        db_merged.sets.remove_data(id_group=f'{cad}_completo', stage=str(stage), verbose=False)
        gc.collect()

    del db_merged
    gc.collect()

    df_post = pd.read_csv(
        filepath_or_buffer=os.path.join(root_dir, 'metadata', 'df_post.csv'),
        sep = ',',
        index_col=0
    )
    df_post['h'] = 11000
    #remove the column coef_area
    try:
        df_post = df_post.drop(columns=['coef_area'])
    except:
        pass

    df_post.to_csv(
        os.path.join(root_dir, 'metadata', 'df_post.csv'),
        sep=',',
        index_label = 'index',
        index=True,
    )

isTest = True
print('\nFunciones leídas\n')
if isTest:
    case_idx = list(range(3))
    tuple_cases = (case_idx, 'all', case_idx, case_idx)
    
else:
    # Base de datos original
    case_idx = list(range(100))
    fuera = [64, 79, 87, 88, 94]
    for c in fuera:
        case_idx.remove(c)

    tuple_cases = (case_idx, 'all', 'all', 'all') # original, rest, transonic_1, propose_0

list_folders = [
    os.path.join('/home/m.jaraiz/Documentos/DATASETS/data_TIFON/', folder) for folder in ['rans3_basic', 'rans3_basic_rest', 'rans3_transonic_1', 'rans3_propose_0']
    ]

# list_db = [FRODO(root_dir = root_dir, format = 'CODA', initial_parse = True) for root_dir in list_folders]

# list_db = [
#     read_db_CODA(
#         datafolder = root_dir,
#         case_idx = cases,
#         interpolate_vol2surf=False
#     ) for root_dir, cases in zip(list_folders, tuple_cases)
# ]
name = "rans3_isTest"
list_db = juntar_db(
    list_folders = [
        os.path.join(DATA_DIR, folder) for folder in ['rans3_basic', 'rans3_basic_rest', 'rans3_transonic_1', 'rans3_propose_0']
        ],
    case_idx=tuple_cases,
    interpolate_vol2surf = True
)

# Uno detrás de otro, para que nunca estén en memoria los dos datasets fusionados a la vez
for cad, suf in zip([3, 4], ['_airfoil', '_field']):
    fusionar_y_exportar(list_db, cad=cad, name=name + suf)

del list_db
gc.collect()

# database_lengths = [len(db.df_state) for db in list_db]
# discarded_cases = [[64, 79, 87, 88, 94], [], [], []]
 
# def get_kept_indices(dbs_lns, css_out):
#     kept_indices = []
#     offset = 0
#     for length, discard in zip(dbs_lns, css_out):
#         discard_set = set(discard)
#         kept_indices.extend(offset + i for i in range(length) if i not in discard_set)
#         offset += length
#     return kept_indices
 
# idx_to_print = get_kept_indices(database_lengths, discarded_cases)
# print(f"Total number of kept indices: {len(idx_to_print)}")
# print(f"Kept indices: {idx_to_print}")



# df_post = pd.read_csv(
#     filepath_or_buffer=f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name + suf}/metadata/df_post.csv',
#     sep = ',',
#     index_col=0
# )
# df_post['h'] = 11000
# #remove the column coef_area
# try:
#     df_post = df_post.drop(columns=['coef_area'])
# except:
#     pass

# df_post.to_csv(
#     f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}/metadata/df_post.csv',
#     sep=',',
#     index_label = 'index',
#     index=True,
# )
# Cambiar de nombre un archivo

# for stage in [0, 1]:
#     old_name = f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}/outputs/CADGroup_3_completo_stage_{stage}.h5'
#     new_name = f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}/outputs/CADGroup_3_stage_{stage}.h5'

#     os.rename(old_name, new_name)