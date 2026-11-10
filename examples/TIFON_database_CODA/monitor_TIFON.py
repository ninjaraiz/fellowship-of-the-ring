import os
from FotR import FRODO

case_name = 'rans5'
dataset_dir = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON'

db = FRODO(root_dir=os.path.join(dataset_dir, case_name), format='CODA', initial_parse=True)

# Uno de los tres: case_idx o carpetas, un subset, o una consulta sobre df_cases.
mon = db.residuals.monitor_cases([219, 228, 243, 245, 257])
# mon = db.residuals.monitor_cases(['aoa_4.3363_mach_0.7247', 'aoa_4.9831_mach_0.8085'])
# mon = db.residuals.monitor_cases(query='mach > 0.78 and aoa > 4')

print(mon)
# VS Code (Remote-SSH) reenvía el puerto solo: la dirección local está en la pestaña PORTS.
# Ventana propia en el PC: brave --app=http://localhost:8765/
# El servidor sigue vivo aunque cierres esto, y se para solo a los 30 min sin la página
# abierta. Para pararlo antes: mon.stop(), o kill <pid>.
