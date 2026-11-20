import os
from FotR import LEGOLAS

sources = '/home/m.jaraiz/Documentos/COMAC/wind_turbine_ansa/sources'
control = os.path.join(sources, 'wind_turbine.control')
ini = os.path.join(sources, 'parameter.ini')

# La malla con el disco relleno ("fluid") frente a los ficheros del caso.
lg = LEGOLAS(os.path.join(sources, 'Wind_turbine_disk_fluid_cgns_g2_mesh.h5'))
print(lg.census.check_report(control=control, ini=ini))
print()

# Los mismos ficheros con la malla del disco hueco ("empty"): sin errores.
empty = LEGOLAS(os.path.join(sources, 'Wind_turbine_disk_empty_cgns_g2_mesh.h5'))
print(empty.census.bcs_report(empty.census.check_bcs(control=control, ini=ini)))
print()

# Nodos repetidos del CGNS elevado: solo de alto orden, sin grietas.
cgns = LEGOLAS(os.path.join(sources, 'Wind_turbine_disk_fluid_g2.cgns'))
d = cgns.census.duplicates()
print(d['extra_nodes'], 'nodos de más,', d['crack_faces'], 'caras en grieta')

# La misma malla a orden 2 y a orden 4: mismas esquinas, áreas curvas casi iguales.
c = lg.census.compare(os.path.join(sources, 'Wind_turbine_disk_fluid_g4.cgns'))
print(c['same_corners'], [(r['name'], round(100 * r['area_rel_diff'], 3))
                          for r in c['boundaries'] if r['area_rel_diff'] > 1e-9])
