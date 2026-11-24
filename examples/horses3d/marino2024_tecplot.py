import pandas as pd
from FotR import FRODO

root = '/home/m.jaraiz/Documentos/COMAC/wind_turbine_marino2024/model_A'
binary = '/home/m.jaraiz/repos/horses3d/Solver/bin/horses2plt'
script = '/home/m.jaraiz/Documentos/utils_H3d/POST/horses2plt_batch.slurm'

db = FRODO(root_dir=root, format='HORSES3D')

# Qué se convertiría: finales, estadísticas y autoguardados de cada p (sin escribir nada).
plan = db.to_tecplot(binary, script)
print(plan[['p', 'kind', 'hsol', 'hmesh', 'tec_mb', 'status']])

# Solo la solución final y las estadísticas de p2 y p3, en un trabajo de Slurm por caso
# (desde el nodo maestro de flexo o desde nibbler).
done = db.to_tecplot(binary, script, p=[2, 3], kinds=['final', 'stats'], cpus=4,
                     dry_run=False)
print(done[['p', 'kind', 'tec', 'status', 'job_id']])

# Sin Slurm (en un nodo de cálculo con una sesión interactiva): run='local' espera a que acabe.
# db.to_tecplot(binary, script, p=2, kinds=['final'], run='local', dry_run=False)

# Los .tec quedan en outputs/<caso>/POST/tec/p<p>/ y el reader los lista con cada solución.
db.reader.parse_simulation_dirs()
print(pd.Series(db.sim_metadata['marino_A_tsr_6.0']['solutions'][2]['tec']))
