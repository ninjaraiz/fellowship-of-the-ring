import os
import pandas as pd
from FotR import GANDALF, FRODO

root = '/home/m.jaraiz/Documentos/COMAC/wind_turbine_ansa'
sources = os.path.join(root, 'sources')
utils = '/home/m.jaraiz/Documentos/utils_H3d'
repo = '/home/m.jaraiz/repos/horses3d-gpu-ad'   # rama arr/actuator_disk (PR #121): disco actuador
R, U = 0.447, 10.0

gdf = GANDALF(root_dir=root, ring='horses3d')
gdf.define_cases(method='external', external_dataframe=pd.DataFrame({'tsr': [6.0]}))
gdf.compute_param(name='omega', formula=f'tsr*{U}/{R}')

# La plantilla es un .control que ya funciona: el ring cambia las claves, no hay marcadores.
# Malla, p, nombre de la solución y escalera de restart (p2 -> p3) los pone el ring.
gdf.generate_folders(
    control_template=os.path.join(sources, 'wind_turbine.control'),
    mesh_path=os.path.join(sources, 'Wind_turbine_disk_fluid_cgns_g2_mesh.h5'),
    tag='wt',
    folder_fmt='tsr_{tsr:.1f}',
    p_levels=[2, 3],
    control_values={'mach number': 0.02934, 'final time': 20.0},
    p_values={3: {'final time': 25.0}},
    extra_dirs=[os.path.join(sources, 'ActuatorDef')],
    data_to_update={'OMEGA_PLACEHOLDER': 'omega'},
    run_script=os.path.join(utils, 'run.sh'),
)

# Primero la prueba FSP (Inflow en todas las fronteras): la malla tiene que conservar la
# corriente libre. Sin dry_run=False solo muestra los comandos.
print(gdf.submit_cases(2, repo, fsp=True))
# gdf.submit_cases(2, repo, fsp=True, dry_run=False)
# gdf.fsp_report(2)                      # cuando termine: la solución no debe moverse (< 1e-6)
# gdf.submit_cases(2, repo, dry_run=False)     # la simulación de verdad

db = FRODO(root_dir=root, format='HORSES3D')
print(db.df_state[['case', 'p_available', 'p_finished', 'state', 'g']])
