import sys
try:
    import pyLOM
except ImportError as e:
    print(f'Error importing pyLOM: {e}')
    print('Importing with local repository')
    sys.path.append('/home/m.jaraiz/repos/pyLowOrder/')
from FotR import FRODO, SAM

def read_db_CODA(datafolder, case_idx, interpolate_vol2surf=True):
    db = FRODO(root_dir = datafolder, format = 'CODA', initial_parse = True)
    
    if interpolate_vol2surf:
        for id, type, var_excluded in zip([3, 4], ['surface', 'volume'], [[f'BoundaryValues_CoefSkinFriction{coord}' for coord in ['X', 'Y', 'Z']], []]):
            db.extract_inputs(
                id_groups = (id,),
                cases_idx = case_idx,
                vtu_type=type,
                verbose=False
                )

            for stage in [0, 1]:
                db.extract_outputs(
                    id_groups=(id,),
                    stage=stage, cases_idx = case_idx,
                    var_name_excluded = var_excluded,
                    vtu_type=type,
                    )
        
        db.sets.interpolate_vol2surf(
            vol_group = '4',
            surf_group = '3',
            stage = str(stage),
            vars = 'all',
        )
        
        db.sets.remove_data(id_group='4', stage=None, verbose=False)
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
                var_name_excluded = [f'BoundaryValues_CoefSkinFriction{coord}' for coord in ['X', 'Y', 'Z']],
                vtu_type='surface',
                )
    return db

from typing import Union
import os
def juntar_db(list_folders:Union[tuple[str], list[str]], case_idx:Union[tuple, list], name: str = None, interpolate_vol2surf:bool = True):
    list_db = []
    
    for folder, cases in zip(list_folders, case_idx):
        db = read_db_CODA(datafolder = folder, case_idx=cases, interpolate_vol2surf=interpolate_vol2surf)
        
        if db.data_dict['CADGroup_3']['FlCc'].shape[-1] > 2:
            flcc = db.data_dict['CADGroup_3']['FlCc'][:, :-1]
            design_vars = db.metadata['design_vars']
            db.metadata['design_vars'] = design_vars[:-1]
            db.data_dict['CADGroup_3']['FlCc'] = flcc
    
        list_db.append(db)
        
    db_full = FRODO.merge_datasets(
        root_dir=f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}',
        name = name,
        sources = [(db, '3') for db in list_db], #[(db_0, '3'), (db_1, '3'), (db_trans, '3')],
        new_group_id='3_completo',
        k=4,
        mesh_ref=0,
        cache=True,
        get_df_metrics_attr={
            'var_metrics': ['CoefLift', 'CoefDrag', 'CoefMomentY'],
            'iter_var': 1000,
            'save' : False
        }
        
    )

    return db_full

isTest = False

if isTest:
    case_idx = list(range(3))
    tuple_cases = (case_idx, 'all', case_idx, case_idx)
    
else:
    # Base de datos original
    case_idx = list(range(100))
    fuera = [64, 79, 87, 88, 94]
    for c in fuera:
        case_idx.remove(c)

    tuple_cases = (case_idx, 'all', 'all', 'all') # original, rest, transonic, propose_0

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

name = "rans3_PRUEBA_2"
db_completo = juntar_db(
    list_folders = [
        os.path.join('/home/m.jaraiz/Documentos/DATASETS/data_TIFON/', folder) for folder in ['rans3_basic', 'rans3_basic_rest', 'rans3_transonic_1', 'rans3_propose_0']
        ],
    case_idx=tuple_cases,
    name = name,
    interpolate_vol2surf=False
)

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
 
path = f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}/outputs'
db_completo.sets.create_NN_pylom(id_groups = '3_completo', stage='0', idx_to_print='all', save_path = path)
db_completo.sets.create_NN_pylom(id_groups = '3_completo', stage='1', idx_to_print='all', save_path = path)

import pandas as pd

df_post = pd.read_csv(
    filepath_or_buffer=f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}/metadata/df_post.csv',
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
    f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}/metadata/df_post.csv',
    sep=',',
    index_label = 'index',
    index=True,
)
# Cambiar de nombre un archivo

for stage in [0, 1]:
    old_name = f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}/outputs/CADGroup_3_completo_stage_{stage}.h5'
    new_name = f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{name}/outputs/CADGroup_3_PRUEBA0_stage_{stage}.h5'

    os.rename(old_name, new_name)