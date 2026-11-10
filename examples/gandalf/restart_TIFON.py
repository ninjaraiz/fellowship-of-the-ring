import os
import sys
import pandas as pd
from FotR import FRODO

case_name = 'rans5'
dataset_dir = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON'

# case_idx de FRODO (columna 'case_idx' de db.metadata['df_cases'], casos ordenados por aoa).
# No es la fila de metadata/df_cases.csv: el script imprime la carpeta de cada uno para comprobarlo.
cases_idx = [203]#, 203]
cpus = 48
node = 'n004'
lanzar = True  # False: solo muestra los cambios. True: los aplica y hace sbatch (solo desde flexo)

db = FRODO(root_dir=os.path.join(dataset_dir, case_name), format='CODA', initial_parse=True)
stage = db.metadata['num_stages'] - 1

df_cases = db.metadata['df_cases'].set_index('case_idx')
desconocidos = [i for i in cases_idx if i not in df_cases.index]
if desconocidos:
    sys.exit(f'case_idx que no existen en {case_name}: {desconocidos}')

print(f'\nCasos pedidos (stage {stage}):')
for i in cases_idx:
    folder = df_cases.loc[i, 'folder']
    monitor = os.path.join(db.reader.output_dir, folder, f'output_{stage}__monitors_TimeIntegration.dat')
    if os.path.exists(monitor):
        ultima = open(monitor).read().rstrip().splitlines()[-1].split()[3]
        print(f'  {i:4d}  {folder}  última iteración del stage {stage}: {ultima}')
    else:
        print(f'  {i:4d}  {folder}  sin output_{stage} (stage sin terminar)')

res = db.residuals.restart_stalled_cases(
    cases_idx,
    cpus=cpus,
    node=node,
    cluster='linux',
    dry_run=not lanzar,
    submit=lanzar,
)
idx_por_carpeta = {f: i for i, f in df_cases['folder'].items()}
res.insert(0, 'case_idx', res['folder'].map(idx_por_carpeta))

pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 120)
print('\n' + res[['case_idx', 'folder', 'status', 'reason', 'job_id']].to_string(index=False))

if not lanzar:
    for _, fila in res[res['status'] == 'planned'].iterrows():
        print(f"\n{fila['case_idx']}  {fila['folder']}")
        for cambio in fila['changes']:
            print(f'    {cambio}')
    print('\nNada modificado. Revisa la tabla y pon lanzar = True para aplicarlo y hacer sbatch.')
