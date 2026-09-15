## Flujo HORSES3D (parcial)

ES: LES con malla por caso (grado g en el nombre) y soluciones por p.
EN: LES with per-case mesh (grade g in the name) and per-p solutions.

```
casos: 2  design_vars: ['M', 're']
        case p_available  g
M_0.3_re_200         [2]  2
M_0.3_re_400      [2, 4]  2
```

ES: `extract_inputs` agrupa por malla/g; FlCc solo design_vars:
EN: `extract_inputs` groups by mesh/g; FlCc holds design_vars only:
```
grupo: CADGroup_g2_my_esphere_v2g2_mesh
Coord: (359856, 3)  FlCc: [[0.3, 200.0]]
```

ES: `extract_outputs` guarda descriptores lazy (sin leer 56MB):
EN: `extract_outputs` stores lazy descriptors (no 56MB read):
```
Vars['2']: ['rho', 'rhoE', 'rhou', 'rhov', 'rhow']
<HorsesLazyField 'rho' case='M_0.3_re_200' p=2 n_points=359856 file='my_esphere_v2g2p2.hsol'>
```

ES: historial de residuos del caso:
EN: case residual history:
```
filas: 1000001  columnas: ['Iteration', 'Time', 'Total_elapsed_Time(s)', 'Solver_elapsed_Time(s)', 'continuity']
```

ES: pendiente — sets/stats HORSES3D (sin `create_jset` aun).
EN: pending — HORSES3D sets/stats (no `create_jset` yet).
