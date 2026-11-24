import os
from FotR import LEGOLAS

sources = '/home/m.jaraiz/Documentos/COMAC/wind_turbine_ansa/sources'
out = 'legolas_out'
os.makedirs(out, exist_ok=True)

# 1. La g4 en .msh de ANSA trae su propio orden de nodos: LEGOLAS lo deduce de la
#    geometría, lo corrige y la reescribe en el orden de gmsh, que PyHOPE ya lee.
lg = LEGOLAS(os.path.join(sources, 'Wind_turbine_disk_empty_g4.msh'))
for r in lg.convert.fix_order():
    print('bloque', r['block'], 'celdas mal:', r['fails'], 'acuerdo:', r['agreement'],
          'después:', r['fails_after'], 'aplicado:', r['applied'])
print(lg.convert.write_msh(os.path.join(out, 'Wind_turbine_disk_empty_g4_gmsh.msh')))

# 2. La turbina g2 para ParaView: volumen curvo (Lagrange) con la calidad por celda,
#    y una pieza por frontera dentro de un .vtm.
g2 = LEGOLAS(os.path.join(sources, 'Wind_turbine_disk_fluid_cgns_g2_mesh.h5'))
print(g2.convert.to_vtm(os.path.join(out, 'turbine_g2'), mode='lagrange', quality=True))

# 3. ¿Encontraría HORSES3D todas las estaciones del disco actuador?
dat = os.path.join(sources, 'ActuatorDef', 'Act_ActuatorDiskDef.dat')
r = g2.locate.horses3d_ad(dat, nranks=4)
print('estaciones no encontradas:', int((~r['found']).sum()), '| perdidas dentro del fluido:', len(r['lost']))
