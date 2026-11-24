from FotR import LEGOLAS

mesh = '/home/m.jaraiz/Documentos/COMAC/my_ESPHERE_1/outputs/M_0.3_re_400/MESH/my_esphere_v2g2_mesh.h5'

# Esfera de diámetro 1 a Re = 400 (laminar), corriente según +x, resuelta con p = 3.
lg = LEGOLAS(mesh)
print(lg.resolution.report(walls='esphere_surface', p=3, Re=400,
                           correlation='laminar', flow_dir=[1, 0, 0]))

# Primera celda y primer nodo de solución en la pared, cara a cara (numpy).
w = lg.resolution.wall('esphere_surface', p=3)
print('primer nodo: de', w['y1_dof'].min(), 'a', w['y1_dof'].max())

# Al visor, para colorear por Δ de DG o por y⁺ estimado (selector Color).
fields = lg.resolution.fields(p=3, walls='esphere_surface', Re=400,
                              correlation='laminar', flow_dir=[1, 0, 0])
# lg.viewer.show(fields=fields)
