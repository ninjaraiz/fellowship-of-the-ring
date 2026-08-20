import sys
try:
    import pyLOM
except ImportError as e:
    print(f'Error importing pyLOM: {e}')
    print('Importing with local repository')
    sys.path.append('/home/m.jaraiz/repos/pyLowOrder/')
from FotR import FRODO, SAM

def read_db_CODA(datafolder, case_idx):
    db = FRODO(root_dir = datafolder, format = 'CODA', initial_parse = True)
    
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

    return db
from typing import Union
import os
def juntar_db(list_folders:Union[tuple[str], list[str]], case_idx:Union[tuple, list], name: str = None):
    list_db = []
    
    for folder, cases in zip(list_folders, case_idx):
        db = read_db_CODA(datafolder = folder, case_idx=cases)
        
        if db.data_dict['CADGroup_3']['FlCc'].shape[-1] > 2:
            flcc = db.data_dict['CADGroup_3']['FlCc'][:, :-1]
            design_vars = db.metadata['design_vars']
            db.metadata['design_vars'] = design_vars[:-1]
            db.data_dict['CADGroup_3']['FlCc'] = flcc
    
        list_db.append(db)
        
    db_full = FRODO.merge_datasets(
        root_dir='/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_completed',
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
    
db_completo = juntar_db(
    list_folders = [
        os.path.join('/home/m.jaraiz/Documentos/DATASETS/data_TIFON/', folder) for folder in ['rans3_basic', 'rans3_basic_rest', 'rans3_transonic_1', 'rans3_propose_0']
        ],
    case_idx=tuple_cases,
    name = "complete"
)

path = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_completed/outputs'
db_completo.sets.create_NN_pylom(id_groups = '3_completo', stage='0', save_path = path)
db_completo.sets.create_NN_pylom(id_groups = '3_completo', stage='1', save_path = path)

import pandas as pd

df_post = pd.read_csv(
    filepath_or_buffer='/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_completed/metadata/df_post.csv',
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
    '/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_completed/metadata/df_post.csv',
    sep=',',
    index_label = 'index',
    index=True,
)
# Cambiar de nombre un archivo

for stage in [0, 1]:
    old_name = f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_completed/outputs/CADGroup_3_completo_stage_{stage}.h5'
    new_name = f'/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_completed/outputs/CADGroup_3_#280_stage_{stage}.h5'

    os.rename(old_name, new_name)