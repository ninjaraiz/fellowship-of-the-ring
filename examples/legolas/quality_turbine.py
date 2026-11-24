import os
from FotR import LEGOLAS

sources = '/home/m.jaraiz/Documentos/COMAC/wind_turbine_ansa/sources'

# Calidad de la malla g2 de la turbina: tabla, histogramas y avisos.
lg = LEGOLAS(os.path.join(sources, 'Wind_turbine_disk_fluid_cgns_g2_mesh.h5'))
print(lg.quality)

# Las cinco peores celdas por jacobiano; file_cell + 1 es el nº que imprime HORSES3D.
for w in lg.quality.worst('scaled_jacobian', 5):
    print(w['cell'], round(w['value'], 3), w['file_cell'] + 1, [round(c, 3) for c in w['centroid']])

# Los valores por celda, para lo que haga falta (numpy).
jr = lg.quality.field('jacobian_ratio')
print('jacobiano mín/máx (lo que imprime PyHOPE):', jr.min())

# En el visor: Color -> jacobiano escalado, y un clic en las peores celdas.
# lg.viewer.show()
