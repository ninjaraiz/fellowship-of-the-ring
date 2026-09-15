## Flujo PYLOM

ES: Dataset pyLOM con variables por caso y fields por punto.
EN: pyLOM Dataset with per-case variables and per-point fields.

```
fields: ['cp']
variables: ['aoa', 'mach']
```

ES: entradas y campo extraidos:
EN: extracted inputs and field:
```
xyz: (13862, 3)  cp: (13862, 280)
df_state: (280, 2)
```

ES: tensor conjunto del formato PYLOM:
EN: PYLOM joint tensor:
```
tensor: (3881360, 6)  info: {'ninputs': 5, 'noutputs': 1}
```
