## Flujo CODA

ES: parseo de la base Cylinder_PINNS (formato CODA, dos stages).
EN: parsing the Cylinder_PINNS base (CODA format, two stages).

```

casos: 31  design_vars: ['M']
```

ES: grupos CAD disponibles en superficie, stage 1:
EN: available surface CAD groups, stage 1:
```
CADGroupID / cell_data summary:
  CADGroupIDs : (2, 3, 4, 5, 6)
  cell_data   : ('BoundaryValues_CoefPressure', 'BoundaryValues_CoefSkinFrictionTangential', 'BoundaryValues_CoefSkinFrictionX', 'BoundaryValues_CoefSkinFrictionY', 'BoundaryValues_CoefSkinFrictionZ', 'BoundaryValues_YPlusFirstCell', 'CADGroupID', 'GlobalNumber')
  Simulations : ['M_0.3000', 'M_0.3200', 'M_0.3400', 'M_0.3500', 'M_0.3600', 'M_0.3800', 'M_0.4000', 'M_0.4200', 'M_0.4400', 'M_0.4500', 'M_0.4600', 'M_0.4800', 'M_0.5000', 'M_0.5200', 'M_0.5400', 'M_0.5500', 'M_0.5600', 'M_0.5800', 'M_0.6000', 'M_0.6200', 'M_0.6400', 'M_0.6500', 'M_0.6600', 'M_0.6800', 'M_0.7000', 'M_0.7200', 'M_0.7400', 'M_0.7500', 'M_0.7600', 'M_0.7800', 'M_0.8000']
```

ES: `extract_inputs` con dos casos del grupo 3:
EN: `extract_inputs` with two cases of group 3:
```
Coord: (220, 3)  FlCc: [[0.30000001192092896], [0.3199999928474426]]
```

ES: `extract_outputs` del stage 1 (primera variable):
EN: `extract_outputs` of stage 1 (first variable):
```
BoundaryValues_CoefPressure: (220, 2)
```

ES: ultimos residuos agregados:
EN: aggregated final residuals:
```
filas: 31  columnas: ['total_iter', 'CFL', 'DensityResidual', 'EnergyStagnationDensityResidual', 'Iteration', 'MomentumResidual']
```

ES: tensor conjunto via `sets.create_jset`:
EN: joint tensor via `sets.create_jset`:
```
tensor: (440, 10)  info: {'ninputs': 4, 'noutputs': 6}
```
