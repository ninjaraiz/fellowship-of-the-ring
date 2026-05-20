import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
# sys.path.append('/home/m.jaraiz/repos/pyLowOrder/')
from FotR import FRODO

def read_db(datafolder, case_idx):
    db = FRODO(root_dir = datafolder, format = 'CODA', initial_parse = True)
    
    db.extract_inputs(
        id_groups = (3,),
        cases_idx = case_idx,
        vtu_type='surface',
        verbose=False
        )

    db.extract_inputs(
        id_groups = (4,),
        cases_idx = case_idx,
        vtu_type='volume',
        verbose=False
        )
    
    for stage in [0, 1]:
        db.extract_outputs(
            id_groups=(3,),
            stage=stage, cases_idx = case_idx,
            var_name_excluded = [
                'BoundaryValues_CoefSkinFrictionX',
                'BoundaryValues_CoefSkinFrictionY',
                'BoundaryValues_CoefSkinFrictionZ'
                ],
            vtu_type='surface',
            )
        
        db.extract_outputs(
            id_groups=(4,),
            stage=stage, cases_idx = case_idx,
            var_name_excluded = [],
            vtu_type='volume',
            )
    
    return db

# Base de datos original
case_idx = list(range(100))
fuera = [64, 79, 87, 88, 94]
for c in fuera:
    case_idx.remove(c)
# case_idx = list(range(5))
db_0 = read_db(
    datafolder = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3/',
    case_idx = case_idx,
    )

db_1 = read_db(
    datafolder = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_rest/',
    case_idx = 'all',
)

for stage in [0, 1]:
    db_0.sets.interpolate_vol2surf(
        vol_group = '4',
        surf_group = '3',
        stage = str(stage),
        vars = 'all',
    )

    db_1.sets.interpolate_vol2surf(
        vol_group = '4',
        surf_group = '3',
        stage = str(stage),
        vars = 'all',
    )
print(db_0.metadata['design_vars'])
print(db_1.metadata['design_vars'])
flcc = db_1.data_dict['CADGroup_3']['FlCc'][:, :-1]
design_vars = db_1.metadata['design_vars']
db_1.metadata['design_vars'] = design_vars[:-1]
db_1.data_dict['CADGroup_3']['FlCc'] = flcc
print(db_0.metadata['design_vars'])
print(db_1.metadata['design_vars'])

db_completo = FRODO.merge_datasets(
    root_dir='/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_extended',
    sources = [(db_0, '3'), (db_1, '3')],
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

db_completo.summary_data()

import pandas as pd
df_post = pd.read_csv(
    '/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_extended/metadata/df_post.csv',
    sep=',',
    header=0,
)

for stage, factor in zip([0, 1], [1, 1]):
    vars = {
        'aoa' : {'idim':0, 'value':df_post['aoa'].values}, # Angle of attack
        'M'   : {'idim':0, 'value':df_post['mach'].values},   # Mach number
        'Re'  : {'idim':0, 'value':df_post['re'].values},  # Reynolds
        'CL'  : {'idim':0, 'value':df_post[f'coeflift_mean_stage{stage}'].values / factor},  # Mean Lift coefficient
        'CD'  : {'idim':0, 'value':df_post[f'coefdrag_mean_stage{stage}'].values / factor},  # Mean Drag coefficient
        'CMy'  : {'idim':0, 'value':df_post[f'coefmomenty_mean_stage{stage}'].values / factor},  # Mean Momentum coefficient
        'varCL'  : {'idim':0, 'value':df_post[f'coeflift_var_stage{stage}'].values},  # Var Lift coefficient
        'varCD'  : {'idim':0, 'value':df_post[f'coefdrag_var_stage{stage}'].values},  # Var Drag coefficient
        'varCM'  : {'idim':0, 'value':df_post[f'coefmomenty_var_stage{stage}'].values},  # Var Momentum coefficient
    }
    _ = db_completo.sets.create_NN_pylom(
        id_groups=['3_completo',],
        stage=stage, idx_to_print='all',
        external_vars=vars,
        save_path='/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_extended/outputs/',
        nan_policy = 'fill')



import plotly.express as px
import matplotlib.pyplot as plt
fig = px.scatter(
    df_post,
    x='coeflift_mean_stage0',
    y='coeflift_mean_stage1',
    color='aoa',
    hover_data=['case_idx', 'mach', 'dataset']
)
# añadir línea diagonal y=x
fig.add_shape(
    type='line',
    x0=0, y0=0, x1=1, y1=1,
    line=dict(color='LightGray', dash='dash'),
    xref='x', yref='y'
)

plt.savefig('./pictures/coeflift_mean_comparison.png')

fig = px.scatter(
    df_post,
    x='coefdrag_mean_stage0',
    y='coefdrag_mean_stage1',
    color='aoa',
    hover_data=['case_idx', 'mach', 'dataset']
)
fig.add_shape(
    type='line',
    x0=0, y0=0, x1=0.2, y1=0.2,
    line=dict(color='LightGray', dash='dash'),
    xref='x', yref='y'
)

plt.savefig('./pictures/coefdrag_mean_comparison.png')

fig = px.scatter(
    df_post,
    x="aoa",
    y="coeflift_mean_stage1",
    color="mach",
    title="Polar CL vs AoA"
)
plt.savefig('./pictures/polar_cl.png')

fig = px.scatter(
    df_post,
    x="coefdrag_mean_stage1",
    y="coeflift_mean_stage1",
    color="aoa"
)
plt.savefig('./pictures/cl_cd.png')