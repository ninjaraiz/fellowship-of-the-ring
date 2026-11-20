from FotR import LEGOLAS

mesh = '/home/m.jaraiz/Documentos/COMAC/wind_turbine_ansa/sources/Wind_turbine_disk_fluid_cgns_g2_mesh.h5'

# El visor corre en un proceso aparte y sobrevive a este script; se apaga solo
# 30 minutos después de cerrar la pestaña. Desde VS Code el puerto se reenvía solo.
lg = LEGOLAS(mesh)
h = lg.viewer.show()
print(h.url)

# Una malla grande se dibuja diezmada por encima de max_triangles (1 millón por
# defecto), y sus zonas arrancan ocultas.
# h = lg.viewer.show(max_triangles=3_000_000)

# Para pararlo antes de tiempo:
# h.stop()
