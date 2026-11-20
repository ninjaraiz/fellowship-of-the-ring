import os
from FotR import LEGOLAS

sources = '/home/m.jaraiz/Documentos/COMAC/wind_turbine_ansa/sources'

# La misma malla g2 de la turbina en los tres formatos que salen de ANSA y PyHOPE.
files = ['Wind_turbine_disk_fluid_cgns_g2_mesh.h5',   # HOPR/PyHOPE (lo que lee HORSES3D)
         'Wind_turbine_disk_fluid_g2.cgns',           # CGNS/HDF5 de ANSA
         'Wind_turbine_disk_fluid_g2.msh']            # gmsh 2.2 binario big-endian de ANSA

for f in files:
    lg = LEGOLAS(os.path.join(sources, f))
    print(lg.census)                                  # tabla: BC, caras, área, caras interiores
    print()

# El resumen también es un dict, para comparar sin mirar tablas.
s = LEGOLAS(os.path.join(sources, files[0])).census.summary()
inner = [b['name'] for b in s['boundaries'] if b['interior_faces']]
print('zonas interiores (caras con celda a cada lado):', inner)

# La g4 en .msh de ANSA trae un orden de nodos propio: LEGOLAS lo dice al leer.
g4 = LEGOLAS(os.path.join(sources, 'Wind_turbine_disk_fluid_g4.msh'))
for w in g4.mesh.meta['warnings']:
    print('aviso:', w)
