import os
from FotR import LEGOLAS

case = '/home/m.jaraiz/Documentos/COMAC/wind_turbine_marino2024/model_A/outputs/marino_A_tsr_6.0'

# Como lo abre HORSES3D: malla y solución salen del propio control del caso.
lg = LEGOLAS.from_control(os.path.join(case, 'file_control_p3.control'))
print(lg.solution)
print({k: [round(v, 4) for v in r] for k, r in lg.solution.summary()['range'].items()})

# Otra solución de la misma malla (un autoguardado, o en unidades SI).
# lg.solution.load(os.path.join(case, 'RESULTS', 'adA_g2p3_0000500000.hsol'), physical=True)

# Valores en los nodos de cada celda (sin promediar entre elementos) y medias por elemento.
u = lg.solution.node_field('u')            # (38718, 27): 27 nodos de la malla g2 por celda
print(u.shape, lg.solution.cell_field('u').min())

# Al visor: arranca coloreado por Mach; en Color hay un grupo «Solución» con el resto.
h = lg.viewer.show()
print(h.url)

# Para ParaView: un valor por nodo (promedio entre las celdas que lo comparten).
# lg.convert.to_vtu('marino_p3.vtu', point_fields={'u': lg.solution.point_field('u')})
