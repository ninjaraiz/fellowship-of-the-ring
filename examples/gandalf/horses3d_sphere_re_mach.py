import os
import itertools
import pandas as pd
from FotR import GANDALF, FRODO

root = '/home/m.jaraiz/Documentos/COMAC/my_ESPHERE_3'
reference = '/home/m.jaraiz/Documentos/COMAC/my_ESPHERE_1/outputs/M_0.3_re_400'
utils = '/home/m.jaraiz/Documentos/utils_H3d'
repo = '/home/m.jaraiz/repos/horses3d-gpu'      # el solver, siempre explícito

# Malla completa de Mach y Reynolds: 2 x 2 = 4 casos.
cases = pd.DataFrame(list(itertools.product([0.2, 0.3], [200.0, 400.0])), columns=['M', 'Re'])

gdf = GANDALF(root_dir=root, ring='horses3d')
gdf.define_cases(method='external', external_dataframe=cases)

# Plantilla: un .control que ya ha corrido. Lo que no se nombra aquí se queda como está.
gdf.generate_folders(
    control_template=os.path.join(reference, 'file_control_p2.control'),
    mesh_path=os.path.join(reference, 'MESH', 'my_esphere_v2g2_mesh.h5'),
    tag='sph',
    folder_fmt='M_{M:.2f}_Re_{Re:.0f}',
    p_levels=[2, 3],
    control_values={
        'mach number': 'M',                  # columna de df_cases: un valor por caso
        'Reynolds number': 'Re',             # columna de df_cases: un valor por caso
        'final time': 150.0,                 # valor fijo para todos
        'Number of time steps': 3000000,
        'autosave mode': '"time"',           # las cadenas se escriben tal cual: comillas incluidas
        'autosave interval': 10.0,
        'save gradients with solution': True,  # bool -> .true.
        'write dt restriction': None,        # None borra la clave
    },
    p_values={3: {'final time': 200.0, 'cfl': 0.3, 'dcfl': 0.3}},   # solo el control de p=3
    run_script=os.path.join(utils, 'run.sh'),
)

# p=2 arranca de cero; p=3 reinicia desde la solución de p=2 (restart polorder = 2).
print(gdf.submit_cases(2, repo, fsp=True))          # primero la prueba FSP de la malla
# gdf.submit_cases(2, repo, fsp=True, dry_run=False); gdf.fsp_report(2)
# gdf.submit_cases(2, repo, dry_run=False)          # cuando termine p=2:
# gdf.submit_cases(3, repo, dry_run=False)

db = FRODO(root_dir=root, format='HORSES3D')
print(db.df_state[['case', 'M', 'Re', 'p_available', 'state', 'g']])
