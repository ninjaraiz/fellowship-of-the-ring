import os, shutil
import pandas as pd, numpy as np
# sys.path.append('/home/m.jaraiz/repos/pyLowOrder/')
import sys
try:
    import pyLOM
    print('Entorno con pyLOM instalado')
except ImportError as e:
    print(f'Error al importar pylom: {e}')
    sys.path.append('/home/m.jaraiz/repos/pyLowOrder')
    print('Importando desde carpeta local')
from FotR import GANDALF

case_name = 'rans5'
try:
    shutil.rmtree(f"/home/m.jaraiz/Documentos/DATASETS/data_TIFON/{case_name}/")
except:
    print('No la encuentro')
    pass

dataset_dir = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON'

gdf = GANDALF(
    os.path.join(dataset_dir, case_name),
    eq_type="rans", 
    num_stages = 2,
    version="flowsimulator2024",
    ring='coda'
)
df_post = pd.read_csv('/home/m.jaraiz/Documentos/DATASETS/data_TIFON/rans3_#280_airfoil/metadata/df_post.csv', sep=',', usecols=['aoa', 'mach', 'h', 'case_idx', 're'])
gdf.define_cases(
    method="external",
    external_dataframe = df_post[['aoa', 'mach']]
    # external_dataframe = pd.DataFrame({ # Caso 79 de df_post.csv
    #     "AoA": df_post['aoa'].loc[cases],
    #     "Mach": df_post['mach'].loc[cases],
    #     "h": df_post['h'].loc[cases]
    # }),
)
geom_csv = 'eta_0_0.csv'

gdf.define_geom_file(
    geom_file_path=os.path.join(dataset_dir, 'airfoil_files', geom_csv), sep=',', decimal='.', header = 'infer',
    cols_idx=[1, 3],
    normalize = False
)

c = gdf.array_ptos[:, 0].max() - gdf.array_ptos[:, 0].min()
cm = -c/4 # la malla lo tiene ya ajustado a x=0 el borde de ataque
print(gdf.df_cases.columns)
from FotR.characters.rings import BaseRing
T, P, rho = BaseRing.Backpack.isa_atmosphere(h=11000)
mu = BaseRing.Backpack.Sutherland_law(mu0 = 1.716e-5, T=T, Treference=255.55)

gdf.compute_param(
    name = 're',
    formula = "L * mach*sqrt(1.4*287*T)*rho/mu",
    externals={
        "rho": rho,
        "mu": mu,
        "L": c,
        "T": T
    }
)
print(gdf.df_cases.columns)

gdf.generate_folders(
    base_files = ["run_sst_v5.py", "run.sh"],
    mesh_path='/home/m.jaraiz/Documentos/DATASETS/data_TIFON/GCI/meshes/mesh_f22_v8_f0.75.msh',
    script_dir=os.path.join(dataset_dir, "sources"),
    folder_fmt = "aoa_{aoa:.4f}_mach_{mach:.4f}",
    overwrite=False,
    update_base_files=True,
    data_to_update={
        "AOA_PLACEHOLDER": "aoa",
        "MACH_PLACEHOLDER": "mach",
        # "ALT_PLACEHOLDER": "h",
        "PYTHON_FILE_PLACEHOLDER": "run_sst_v5.py",
        "CHORD_PLACEHOLDER": str(c),
        "CM_PLACEHOLDER": str(cm)
    }
)
gdf.df_cases['Coef_Area'] = c

gdf.df_cases.to_csv(os.path.join(gdf.root_dir,'metadata', 'df_cases.csv'), index=False)

gdf.assign_jobs(
    file_sh="run.sh",
    nodes=[f'n00{n}' for n in [4, 5, 6, 7]],
    cpus_per_job=48,
    submit=True,
    )

gdf.submit_cases()