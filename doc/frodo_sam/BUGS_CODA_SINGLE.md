# BUGS — formato CODA_SINGLE

> Auditoría de `readers/`, `sets/`, `stats/` y `residuals/` para el formato
> `CODA_SINGLE`.
>
> **Estado: todos los bugs están arreglados salvo B12** (dependencia pyLOM no
> declarada, que es de entorno). Los apartados describen el defecto tal y como
> estaba, porque es lo que explica la corrección y lo que hay que evitar
> reintroducir. B10 se resolvió con clases propias — ver
> [`DISENO_SETS_CODA_SINGLE.md`](DISENO_SETS_CODA_SINGLE.md).
>
> Cada bug lleva etiqueta de origen:
> **`[ESPECÍFICO]`** vive en `readers/coda_single.py` · **`[HEREDADO]`** vive en
> `readers/coda.py` o `CODAResiduals` y golpea a CODA_SINGLE al heredarlo (arreglarlo
> toca también a CODA) · **`[DISEÑO]`** es una incompatibilidad estructural, no un
> descuido · **`[ENTORNO]`** depende de la instalación.
>
> Contexto general del framework: [`ANALISIS.md`](ANALISIS.md).

## Dataset de verificación

`/home/m.jaraiz/Documentos/DATASETS/data_TIFON/GCI/` — estudio de convergencia de malla.
Tres raíces FRODO independientes (`aoa_0.022_m_1.071`, `aoa_0.022_m_1.386`,
`aoa_3.303_m_0.541`), cada una con:

```
<root>/
├── metadata/cases_metadata.json     folder_fmt='f{mesh:g}', num_stages=2
│                                    design_vars=['aoa','mach','mesh']
│                                    mesh_map: una entrada por caso → <GCI>/meshes/*.msh
└── outputs/
    ├── f0.8/  mesh_TIFON_v8_f0.8.msh  output_{0,1}_{surface,volume}.vtu  *.dat
    ├── f0.9/  …
    ├── f1/    …
    ├── f2/    …
    ├── f5/    …
    └── f10/   …
```

Dentro de cada raíz **solo varía `mesh`**; `aoa` y `mach` son constantes. Los tamaños de
malla del grupo CAD 3 van de 1082 celdas (`f10`) a 2165 (`f5`).

El número de casos por raíz crece según se van añadiendo refinamientos, así que los
`case_idx` concretos se desplazan. Ninguno de los bugs de abajo depende de eso: todos
salen de la lógica del repo, no de qué casos haya en disco.

Todo lo que sigue está **verificado ejecutando** contra esa raíz con el entorno
`envkan_nvidia`, salvo el apéndice, marcado como lectura de código.

### Hipótesis descartada

Comprobé si leer este dataset con `format="CODA"` — que es lo que hace
`examples/TIFON_database_CODA/coda_single_GCI.ipynb` — rompía el emparejamiento
carpeta↔caso por la distancia euclídea de `CODAReader.parse_simulation_dirs`.
**No rompe**: todas las carpetas se emparejan con su caso correcto. No es un bug y no
aparece abajo.

---

## Resumen

| | bug | origen | qué provoca | estado |
|---|---|---|---|---|
| **B1** | `get_df_metrics` cruza casos | HEREDADO | métricas integrales en la fila equivocada, en silencio | **arreglado** |
| **B2** | `mesh_map` nunca llega al lector | ESPECÍFICO | la identidad de malla declarada se ignora siempre | **arreglado** |
| **B3** | `Conec` inválida | HEREDADO | conectividad inservible (índices y filas equivocados) | **arreglado** (ruta CODA_SINGLE) |
| **B4** | listas desalineadas si un caso falla a mitad | ESPECÍFICO | geometría atribuida a otro caso | **arreglado** |
| **B5** | `extract_inputs` borra `Vars` | ESPECÍFICO | pierde los outputs ya extraídos | **arreglado** |
| **B6** | `df_state` en orden `listdir` | ESPECÍFICO | causa raíz de B1 y B7 | **arreglado** |
| **B7** | `plot_state` etiqueta con la fila, no con `case_idx` | HEREDADO | el número del gráfico no sirve para consultar el caso | **arreglado** |
| **B8** | `plot_integrals_from_case` no arranca | HEREDADO | `TypeError` inmediato | **arreglado** |
| **B9** | `merge_datasets` inusable | DISEÑO | `AttributeError` | **acotado**: error explícito |
| **B10** | `CODASets`/`CODAStats` no aceptan el layout | DISEÑO | los 7 puntos de entrada fallan | **resuelto**: `CODASingleSets`/`CODASingleStats` propios |
| **B11** | `sort_values` por design var constante | HEREDADO | latente: permuta filas con ≳80 casos | **arreglado** |
| **B12** | dependencia ausente reportada como formato no soportado | ENTORNO | enmascara el diagnóstico | sin tocar |
| **B13** | `extract_outputs` no valida `stage` | ESPECÍFICO | error tardío y confuso | **arreglado** |
| **B14** | `df_cases` con columna desfasada mata al lector | HEREDADO | `ValueError` opaco de pandas al añadir un caso | **arreglado** |

**B14** apareció al ejecutar sobre el dataset vivo: al añadir `f0.9`, el ring dejó la
columna `exist` con una entrada menos que el resto, y `pd.DataFrame.from_dict` moría con
`All arrays must be of the same length` desde dentro de pandas, fuera del `except` que
protege la carga del JSON. Ahora las columnas de identidad (design vars, `folder`,
`case_idx`) siguen siendo obligatorias y coherentes —si una está desfasada se lanza un
error que la nombra— mientras que una columna de contabilidad desfasada se descarta con
aviso en vez de bloquear el dataset entero.

### Qué se cambió

| fichero | cambio |
|---|---|
| `readers/coda_single.py` | `__init__` propio que carga `mesh_map`; resolución de malla con fallback; `df_state` ordenado por `case_idx`; `extract_inputs` atómico por caso, con conectividad correcta, extrayendo de los stages que existen y sin pisar `Vars`; `extract_outputs` tolerante a casos saltados y con validación de `stage` |
| `sam.py` | `Backpack.cell_connectivity_in_order` (nuevo); aviso en `get_unified_connectivity` sobre su reagrupación por tipo |
| `readers/coda.py` | `endswith=` en las dos llamadas rotas; orden estable de `df_cases`; `_df_cases_from_dict`; etiquetas de `plot_state` por `case_idx` |
| `residuals/coda.py` | `get_df_metrics` resuelve el caso por `case_idx` y escribe por etiqueta de fila |
| `frodo.py` | `merge_datasets` lanza `NotImplementedError` explicativo si el formato no tiene Sets |
| `tests/test_coda_single.py` | 12 tests nuevos de regresión |
| `sets/base.py` | `_save_result`, `_normalise_cases_idx` y `plot_wall_integrals` subidos desde `CODASets` (estaban duplicados) |
| `sets/coda_single.py`, `stats/coda_single.py` | **nuevos** — B10; ver [`DISENO_SETS_CODA_SINGLE.md`](DISENO_SETS_CODA_SINGLE.md) |
| `tests/test_coda_single_sets.py` | **nuevo** — 32 tests |

---

## B1 · `[HEREDADO]` · CRÍTICO — `get_df_metrics` asigna las métricas al caso equivocado

**Síntoma.** `db.residuals.get_df_metrics(...)` devuelve una tabla en la que las columnas
`<var>_mean_stage<s>` / `<var>_var_stage<s>` pertenecen a un caso distinto del que
describe la fila. Sin error, sin warning.

**Causa raíz.** `residuals/coda.py:589-619`:

```python
for irow in range(len(db.df_state)):          # :589  posición de fila de df_state
    case_name   = db.reader.case_per_idx(irow) # :592  …usada como case_idx
    ...
    df_post.loc[irow, f"{v}_mean_stage{stage}"] = df_tail[v].mean()   # :618
```

`irow` se usa a la vez como **posición de fila de `df_state`** (para escribir) y como
**`case_idx`** (para resolver la carpeta). En CODA esos dos índices coinciden porque
`df_state` y `df_cases` se ordenan ambos por `design_vars[0]`. En CODA_SINGLE no: `df_state`
se construye recorriendo `sorted(os.listdir(...))` (ver **B6**) y `df_cases` conserva el
orden del JSON.

```
df_cases :  f0.8  f0.9  f1  f2  f5  f10     (case_idx 0 1 2 3 4 5)
df_state :  f0.8  f0.9  f1  f10 f2  f5      (orden listdir)
```

**Evidencia.** `CoefLift` promediado sobre las últimas 1000 iteraciones del stage 1,
comparando `get_df_metrics` contra la lectura directa del monitor de cada carpeta:

```
folder  esperado      obtenido
f0.8    0.00349758    0.00349758     OK
f1      0.00398735    0.00398735     OK
f10     0.00886448    0.00280372     <<< MAL   (recibe las de f2)
f2      0.00280372    0.00727932     <<< MAL   (recibe las de f5)
f5      0.00727932    0.00886448     <<< MAL   (recibe las de f10)
```

La permutación es exactamente la que predice el desfase de órdenes. **No depende de qué
casos haya en disco**: es la composición de dos ordenaciones distintas, así que basta con
que un nombre de carpeta ordene distinto en lexicográfico que en el JSON (`f10` frente a
`f2`/`f5`) para arrastrar a los demás. Se verificó idéntica antes y después de añadir un
caso nuevo a la raíz.

**Por qué no se detectó.** `residuals/coda.py` no tiene ningún test de `get_df_metrics`
(ver ANALISIS §6), y en CODA la coincidencia de ordenaciones lo tapa.

**Gravedad.** Es el bug más serio del conjunto: corrompe justo la magnitud que se usa en un
estudio GCI (coeficientes integrales vs. refinamiento de malla), y lo hace en silencio. Si
se han sacado conclusiones de convergencia con esta función sobre un dataset CODA_SINGLE,
hay que rehacerlas.

**Nota de arreglo.** El `df_state` de CODA_SINGLE ya trae la columna `case_idx` correcta;
basta indexar por ella en lugar de por la posición (`for irow, case_idx in
df_state['case_idx'].items()`), o reordenar `df_state` para que siga a `df_cases`. Toca
código compartido con CODA: conviene cubrirlo antes con un test que use un `df_state`
deliberadamente desordenado.

---

## B2 · `[ESPECÍFICO]` · CRÍTICO — `mesh_map` nunca llega al lector

**Síntoma.** La identidad de malla declarada en `cases_metadata.json` se ignora por
completo. `sim_metadata[caso]['mesh_info']['abspath']` apunta a la copia dentro de
`outputs/<caso>/`, no a la ruta declarada.

**Causa raíz.** `CODAReader.__init__` construye `self.metadata` con **exactamente cuatro
claves** más `df_cases` (`readers/coda.py:133-138`):

```python
self.metadata = {
    'eq_type':     cm.get('eq_type',    None),   # :134
    'folder_fmt':  cm.get('folder_fmt', None),
    'design_vars': cm.get('design_vars', None),
    'num_stages':  cm.get('num_stages', None),   # :137
}
```

`mesh_map` no está, y **`CODASingleReader` no sobreescribe `__init__`**. Así que en
`readers/coda_single.py:85`:

```python
mesh_map = self.metadata.get('mesh_map', {}) or {}     # siempre {}
...
mesh_abspath = mesh_map.get(folder)                    # :123  siempre None
if mesh_abspath is None:
    mesh_abspath = self._find_case_mesh(full_path)     # :125  siempre este camino
```

`_find_case_mesh` (`:188-193`) devuelve el **primer `*.msh` alfabético** de la carpeta del
caso.

**Evidencia.**

```
=== metadata keys cargadas por el reader ===
['design_vars', 'df_cases', 'eq_type', 'folder_fmt', 'num_stages']
mesh_map en metadata? -> False

=== JSON en disco tiene ===
['design_vars', 'df_cases', 'eq_type', 'folder_fmt', 'mesh_map', 'num_stages', 'root_dir']
```

**Por qué no se detectó.** El fixture de `FotR/tests/test_coda_single.py` coloca el `.msh`
**dentro** de la carpeta del caso y hace que `mesh_map` apunte ahí mismo. Fallback y
`mesh_map` dan el mismo resultado, así que el test pasa igual con `mesh_map` vacío. El
`CODASingleRing` real, en cambio, escribe rutas a `<GCI>/meshes/` — fuera de `outputs/`.

**Consecuencias más allá de lo cosmético.**
- Si la malla no se copia dentro de la carpeta del caso (que es el escenario que el ring
  contempla, porque `mesh_map` guarda la ruta fuente), `mesh_info` queda a `None` sin aviso.
- Si hubiera más de un `.msh` en la carpeta, se elige uno por orden alfabético.
- Toda la trazabilidad malla↔caso que el ring se molesta en persistir es decorativa.

**Nota de arreglo.** Añadir `'mesh_map': cm.get('mesh_map', {})` en `CODAReader.__init__`
es inocuo para CODA (que simplemente no lo tiene), o bien dar a `CODASingleReader` un
`__init__` propio que llame a `super()` y complete la clave. Lo segundo es más limpio: es
metadato específico del formato. Y el test necesita un fixture con la malla **fuera** de la
carpeta para que el fallback y `mesh_map` sean distinguibles.

---

## B3 · `[HEREDADO]` · ALTO — `Conec` es inválida (dos defectos compuestos)

**Síntoma.** El array `Conec` de cada caso no describe la conectividad del grupo CAD
extraído. No es que esté ordenado de otra forma: son celdas distintas y los índices no
apuntan a `NodeCoord`.

**Causa raíz.** `readers/coda_single.py:283` (idéntico a `readers/coda.py:797`):

```python
connectivity = SAM.Backpack.get_unified_connectivity(mesh)[mask]
```

**(a) Índices de la malla completa contra nodos del subconjunto.**
`get_unified_connectivity(mesh)` opera sobre la malla **entera**, mientras `NodeCoord` sale
de `celdas.points` — el subconjunto extraído, que PyVista **reindexa desde 0**. Los índices
de `Conec` no son válidos contra `NodeCoord`.

**(b) Reordenación por tipo de celda vs. máscara en orden original.**
`get_unified_connectivity` (`sam.py:1824-1854`) recorre `mesh.cells_dict`, que agrupa las
celdas **por tipo**, y las concatena en ese orden:

```python
for _, cells in cell_dict.items():          # :1850  agrupa por tipo
    connectivity[start:start + n, :cells.shape[1]] = cells
```

Pero `mask` está en el **orden original de celdas** de la malla. Aplicar una máscara de un
orden sobre un array del otro selecciona filas que no corresponden.

Además, `NodeCoord` se ordena con `sort_fn` (`readers/coda_single.py:286`) y `Conec` nunca
se remapea con `idx_nodes`, así que aunque (a) y (b) se arreglasen seguiría desalineado.

**Evidencia** (caso `f10`, grupo CAD 3):

```
mesh completa: n_cells=16910  n_points=9008
tipos de celda presentes: {5: (15804, 3), 9: (1106, 4)}     <- tri y quad mezclados

grupo 3: n_cells=1082  NodeCoord=(2164, 3)  Conec=(1082, 4)
Conec  min=-1  max=9005
>>> Conec indexa la malla COMPLETA, no NodeCoord: True       (2164 nodos disponibles)

primera celda del grupo 3 (índice global 324)
  pyvista point_ids : [52, 75, 140, 90]     <- un quad
  conec fila 0      : [191, 196, 195, -1]   <- un triángulo
```

**Por qué no se detectó.** El fixture de `test_coda_single.py` construye mallas de **un
solo tipo de celda** (todo quads), así que (b) no se manifiesta; y ningún test valida
`Conec` contra `NodeCoord`.

**Alcance.** Afecta por igual a CODA. Hoy no molesta porque nada del pipeline consume
`Conec`… salvo `create_pylom_mesh` y `create_NN_pylom`, que son precisamente los que
exportan a pyLOM.

---

## B4 · `[ESPECÍFICO]` · ALTO — las listas de geometría se desalinean si un caso falla a mitad

**Síntoma.** Si un caso falla en un stage posterior al 0, `Coord[i]`, `NodeCoord[i]`,
`Conec[i]`… dejan de corresponder a `case_order[i]` y a `FlCc[i]`. Solo se emite un
`UserWarning`.

**Causa raíz.** `readers/coda_single.py:287-331`. Los `append` no son atómicos: la
geometría entra en `stage == 0` (`:287-297`), pero la contabilidad del caso se hace **al
final**, tras recorrer todos los stages:

```python
if stage == 0:
    coords.append(centroids_sorted)      # :288   ← entra pronto
    ...
FlCc[len(kept)] = [...]                  # :314   ← se registra al final
case_order.append(sim_key)               # :321
kept.append(case_i)                      # :324
except Exception as exc:                 # :325
```

Entre medias está el chequeo de coherencia inter-stage (`:307`,
`"Inconsistent {label} coordinates"`), que es exactamente lo que puede lanzar en `stage>0`.

**Evidencia** (forzando que `f5` falle en stage 1; se piden `f5` y `f10`):

```
case_order   : ['f10']  (len 1)
FlCc         : [[0.0222, 1.0715, 10.0]]
Coord        len=2  shapes=[2165, 1082]
NodeCoord    len=2  shapes=[4330, 2164]

>>> Coord tiene 2 entradas pero case_order 1 -> DESALINEADO
>>> Coord[0] tiene 2165 puntos y case_order[0]='f10'
    f5 tiene 2165 ptos, f10 tiene 1082 -> la geometría pertenece a f5
```

Es decir: `FlCc[0]` dice `mesh=10`, `case_order[0]` dice `f10`, y la geometría es la de
`f5`.

**Por qué no se detectó.** El fixture usa `num_stages=1`, así que la rama `stage != 0` —
justo donde vive el fallo — **nunca se ejecuta** en los tests. El dataset real usa
`num_stages=2`.

**Atenuante.** `extract_outputs` compara `case_order` con la selección pedida
(`:391-396`) y aborta con `ValueError("Selection mismatch")`, así que un fallo parcial
suele convertirse en bloqueo total del grupo en lugar de en datos silenciosamente
corruptos. Pero si se consume `data_dict` sin pasar por `extract_outputs`, la corrupción es
silenciosa.

**Nota de arreglo.** Acumular la geometría del caso en variables locales y volcarla a las
listas solo en el mismo punto donde se hace `kept.append` — o envolver todo el caso en una
transacción y hacer `pop()` de lo ya añadido en el `except`.

---

## B5 · `[ESPECÍFICO]` · ALTO — `extract_inputs` borra los outputs ya extraídos

**Síntoma.** Llamar a `extract_inputs` después de `extract_outputs` vacía `Vars`
silenciosamente.

**Causa raíz.** `readers/coda_single.py:345` incluye `'Vars': {}` en el `.update(...)`
final:

```python
self.data_dict.setdefault(key, {}).update({
    'Coord': coords, ..., 'case_order': case_order,
    'Vars':           {},        # :345  ← reinicia
})
```

`CODAReader.extract_inputs` **no** incluye `'Vars'` en su `update`, precisamente para no
pisarlo. Es una regresión respecto al formato del que hereda.

**Evidencia.**

```
tras extract_outputs   -> Vars keys: ['1']
tras 2º extract_inputs -> Vars keys: []          <<< se ha borrado

CODAReader.extract_inputs       incluye "'Vars'": False
CODASingleReader.extract_inputs incluye "'Vars'": True
```

**Impacto práctico.** El flujo natural para comparar stages es extraer inputs una vez y
llamar a `extract_outputs` por stage. Pero cualquier iteración que reextraiga inputs (p. ej.
al añadir otro grupo CAD, o al reintentar tras un fallo) tira los outputs sin avisar.

---

## B6 · `[ESPECÍFICO]` · MEDIO — `df_state` no sigue el orden de `df_cases`

**Síntoma.** La fila *i* de `df_state` no es el caso *i* de `df_cases`.

**Causa raíz.** `readers/coda_single.py:95` construye `df_state` recorriendo
`sorted(os.listdir(self.output_dir))`, que es orden **lexicográfico de nombres de carpeta**,
mientras `df_cases` conserva el orden del JSON (numérico por `mesh`). Con nombres como
`f0.8, f1, f2, f5, f10` los dos órdenes difieren en cuanto hay un `f10`.

**Evidencia.**

```
df_cases :  case_idx  0→f0.8  1→f0.9  2→f1  3→f2  4→f5  5→f10
df_state :  fila      0→f0.8  1→f0.9  2→f1  3→f10 4→f2  5→f5
            columna case_idx:  0, 1, 2, 5, 3, 4     ← correcta
```

**Matiz importante.** El `df_state` de CODA_SINGLE **sí** lleva una columna `case_idx`
correcta (`readers/coda_single.py:149`). El problema no es que le falte información, sino
que ningún consumidor la usa: todos indexan por posición de fila. Esto convierte a B6 en la
causa raíz de **B1** y **B7**.

**Nota de arreglo.** Dos opciones no equivalentes: (a) ordenar `df_state` por `case_idx` al
construirlo, que arregla a todos los consumidores de golpe pero cambia un orden observable;
(b) hacer que los consumidores usen la columna `case_idx`, que es más correcto pero toca
código compartido con CODA. La (a) es la de menor riesgo.

---

## B7 · `[HEREDADO]` · MEDIO — `plot_state` etiqueta con la posición de fila, no con `case_idx`

**Síntoma.** El número que `plot_state` dibuja junto a cada punto no es el que hay que
pasar a `plot_residuals_from_case(case_idx=...)` ni a `case_per_idx`.

**Causa raíz.** `readers/coda.py:1231` y `:1251` anotan con el índice del `enumerate` sobre
`df_state`:

```python
for i, x in enumerate(df_state[dvf[0]].values):
    ax.annotate(f"{i}", (x, 0), ...)        # :1230-1231
```

Combinado con B6, ese `i` deja de ser el `case_idx`.

**Evidencia.**

```
fila 0: folder=f0.8  case_idx real=0
fila 1: folder=f1    case_idx real=1
fila 2: folder=f10   case_idx real=4   <<< la etiqueta del plot no es el case_idx
fila 3: folder=f2    case_idx real=2   <<<
fila 4: folder=f5    case_idx real=3   <<<

case_per_idx(2) -> f2    (lo que realmente obtienes si pides el "2" del gráfico)
```

Es decir: miras el gráfico, ves que el punto «2» tiene mal aspecto, pides
`plot_residuals_from_case(case_idx=2)` y estás mirando otro caso.

---

## B8 · `[HEREDADO]` · MEDIO — `plot_integrals_from_case` no arranca

**Síntoma.** `TypeError` inmediato.

**Causa raíz.** `readers/coda.py:1309` pasa `file_end=` a `find_files`, cuyo parámetro se
llama `endswith=` (`sam.py:998`). El mismo defecto está en `readers/coda.py:183`, dentro de
`_infer_metadata_from_folders` — el camino de fallback cuando no hay
`cases_metadata.json`.

**Evidencia** (sobre el dataset real):

```
reader.plot_integrals_from_case(case_idx=0, stage=[1])
  TypeError: SAM.Backpack.pattern_pocket.find_files() got an unexpected keyword
             argument 'file_end'
```

**Nota.** Ya estaba documentado en [`ANALISIS.md`](ANALISIS.md) §4; se repite aquí porque
se dispara con este dataset y porque la alternativa que sí funciona
(`CODASets.plot_wall_integrals`) **no está disponible en CODA_SINGLE** (`db.sets is None`,
ver B9/B10). En este formato, por tanto, no hay ninguna vía operativa para graficar
integrales de pared desde la API.

---

## B9 · `[DISEÑO]` · MEDIO — `merge_datasets` es inusable en CODA_SINGLE

**Síntoma.** `AttributeError` al fusionar dos bases CODA_SINGLE.

**Causa raíz.** `frodo.py:686` llama `db.sets.interpolate_msh2msh(...)` para homogeneizar
mallas, pero `SETS_REGISTRY` no tiene entrada `CODA_SINGLE`, así que `db.sets is None`.

**Evidencia.**

```
FRODO.merge_datasets(sources=[(db_1071,'3'), (db_1386,'3')], new_group_id='3m')
  AttributeError: 'NoneType' object has no attribute 'interpolate_msh2msh'
```

**Segunda barrera, aguas abajo.** Aunque existiera una clase Sets, el bucle de `Vars` de
`frodo.py:858-882` asume `ndarray`:

```python
var_concat = np.concatenate(var_list, axis=-1)    # :882
```

y en CODA_SINGLE `Vars[stage][var]` es una **lista de arrays de longitudes distintas**. La
fusión de bases con mallas heterogéneas necesita un diseño propio, no un parche.

---

## B10 · `[DISEÑO]` · MEDIO — ningún método de `CODASets`/`CODAStats` acepta el layout

Este es el «planteamiento de sets» que motiva la revisión.

**Situación.** `SETS_REGISTRY` y `STATS_REGISTRY` no tienen `CODA_SINGLE`, así que
`db.sets` y `db.stats` son `None` y `db.create_jset(...)` no existe (el `__getattr__` de
FRODO salta los subobjetos nulos). Está documentado y es deliberado — pero deja el formato
sin ninguna capacidad de ensamblado ni de estadística.

**La pregunta real es si bastaría con registrar las clases de CODA. No basta.** Enchufando
`CODASets` y `CODAStats` a mano sobre un `data_dict` CODA_SINGLE real (2 casos, grupo 3),
**los siete puntos de entrada fallan**, y ninguno con un mensaje útil:

| método | error |
|---|---|
| `create_jset(stage='1', id_group='3')` | `AttributeError: 'list' object has no attribute 'ndim'` |
| `compute_stats(id_group='3', stage='1')` | `AttributeError: 'list' object has no attribute 'ndim'` |
| `save_to_npy(stage=1, id_group='3')` | `AttributeError: 'list' object has no attribute 'shape'` |
| `crop_bounding_box(id_group='3')` | `AttributeError: 'list' object has no attribute 'shape'` |
| `change_order_coord(id_group='3','lexsort')` | `AttributeError: 'list' object has no attribute 'shape'` |
| `interpolate_msh2msh('3'→'x')` | `ValueError: setting an array element with a sequence… inhomogeneous shape` |
| `create_pylom_mesh(id_groups=('3',))` | `TypeError: list indices must be integers or slices, not tuple` |

**El invariante que se rompe.** Todo `CODASets`/`CODAStats` está construido sobre dos
supuestos que CODA_SINGLE invalida por definición:

1. **Una geometría compartida**: un único `Coord` de forma `(n_points, n_dim)` válido para
   todos los casos. En CODA_SINGLE hay una geometría por caso, con `n_points` distinto
   (1082 … 2165 en este dataset).
2. **Un eje de casos denso**: `Vars[stage][var]` de forma `(n_points, n_cases)`, que permite
   `arr[:, idx_flcc]`, `np.percentile` sobre el conjunto, concatenaciones, etc. En
   CODA_SINGLE es una lista de arrays `(n_points_i,)`.

El primero es el que mata a `SAM.Gardener.create_final_tensor`, que recibe un único
`tensor_ptos` y lo repite por casos (`tensor_ptos.repeat(ncases, 1)`): sin malla común no
hay tensor conjunto que ensamblar.

**Consecuencia para el diseño de `sets/coda_single.py`.** Hay que decidir explícitamente la
semántica antes de escribir código, porque no hay una respuesta única:

- **Interpolar a una malla de referencia** → recupera el layout denso y todo `CODASets`
  vuelve a valer, pero introduce error de interpolación. Es lo correcto para ML; es
  justamente lo que **no** se quiere en un estudio GCI, donde la diferencia entre mallas
  *es* la señal.
- **Un `jset` por caso** → conserva los datos, pero rompe el contrato de `create_jset`
  (devolver un tensor único) y obliga a repensar la normalización (`mins`/`maxs`
  compartidos o por caso).
- **Estadística por caso y luego agregación** → es lo que pide el GCI (una magnitud
  integral o un valor puntual por malla, y después Richardson). Probablemente `compute_stats`
  sobre listas sea trivial y lo que de verdad hace falta sea un helper de orden de
  convergencia, no un `create_jset`.

---

## B11 · `[HEREDADO]` · BAJO (latente) — `sort_values` por una design var constante

**Causa raíz.** `readers/coda.py:142` ordena `df_cases` por `design_vars[0]` y **después**
inserta `case_idx` (`:146-147`). En CODA_SINGLE `design_vars[0]` es `aoa`, constante dentro
de cada raíz. `pandas.sort_values` usa quicksort por defecto, que no es estable por
contrato.

**Evidencia** (columna constante, comprobando si se preserva el orden original):

```
n=  5  orden preservado: True
n= 12  orden preservado: True
n= 30  orden preservado: True
n= 80  orden preservado: False      <<<
```

**Estado actual: no muerde.** Con 5 casos por raíz el sort es un no-op y el orden de
`df_cases` coincide con el JSON. El riesgo aparece si un estudio futuro tiene ≳80 casos con
`design_vars[0]` constante: `case_idx` dejaría de coincidir con el orden de
`metadata/df_cases.csv`, que es lo que el usuario mira.

**Nota adicional.** Si `design_vars` fuese `None` en el JSON, `self.metadata['design_vars'][0]`
lanza `TypeError`, que **no** está entre las excepciones capturadas en `readers/coda.py:150`
(`FileNotFoundError, json.JSONDecodeError, KeyError`).

---

## B12 · `[ENTORNO]` · BAJO — una dependencia ausente se reporta como formato no soportado

**Síntoma.** `UserWarning: No Sets class for format 'CODA'` en un entorno donde CODA sí
debería tener Sets.

**Causa raíz.** `sets/__init__.py` envuelve los imports en `except ModuleNotFoundError: pass`.
`sets/coda.py:31` hace `import pyLOM as SMEAGOL` a nivel de módulo, y `pyLOM` **no está
instalado en `envkan_nvidia` ni declarado en `requirements.txt`/`setup.py`**.

**Evidencia.**

```
SETS_REGISTRY     : {'Airfoil': None, 'NUMPYFILE': 'NUMPYFILESets', 'PYLOM': 'PYLOMSets'}
STATS_REGISTRY    : {'CODA': 'CODAStats', 'NUMPY': 'CODAStats'}
READER_REGISTRY   : ['CODA', 'CODA_SINGLE', 'HORSES3D', 'NUMPY', 'NUMPYFILE']

pyLOM NO disponible -> No module named 'pyLOM'
```

CODA y NUMPY **pierden Sets**; `PYLOM` pierde el *reader* (porque `readers/pylom.py` sí
importa pyLOM a nivel de módulo) mientras conserva el *sets*, dejando el formato
inconsistente.

**Por qué importa en esta auditoría.** Hace indistinguible «CODA_SINGLE no tiene Sets por
diseño» de «falta una dependencia». Al depurar B9/B10 conviene tenerlo presente: en este
entorno `db.sets is None` para **todos** los formatos que dependen de pyLOM, no solo para
CODA_SINGLE.

---

## B13 · `[ESPECÍFICO]` · BAJO — `extract_outputs` no valida el `stage`

`CODAReader.extract_outputs` comprueba `stage >= idx_sort.shape[0]` y lanza un `ValueError`
claro. `CODASingleReader.extract_outputs` (`readers/coda_single.py:398`) no comprueba nada y
falla más tarde, desde `load_vtu_from_stage`:

```
extract_outputs(stage=7): FileNotFoundError: No .vtu file with type 'surface'
                          found in stage 7 of simulation 'f10'.
```

El mensaje culpa al fichero cuando el problema es el argumento. Los sorters por caso están
en `group['idx_sort'][cont]`, un dict cuyas claves son los stages disponibles: validar
contra él es inmediato.

---

## Apéndice — lado generador (solo lectura de código, sin ejecutar)

Tres riesgos en `rings/coda_single.py` que condicionan lo que el lector recibe. **No los he
verificado ejecutando**, a diferencia de todo lo anterior.

1. **No se valida que los nombres de carpeta sean únicos.** `generate_folders` valida que
   los *basenames* de malla no se repitan, pero no los `folder_name`. `folder_fmt` no tiene
   por qué ser inyectivo sobre `design_vars` (`f{mesh:g}` colapsa dos `mesh` que redondeen
   igual; un `folder_fmt` que omita una design var colapsa siempre). Si colapsan,
   `mesh_map[folder_name]` se sobrescribe (`rings/coda_single.py:125`) y `df_cases` acaba
   con dos filas con el mismo `folder`; el lector se queda con la última
   (`readers/coda_single.py:88-90`).

2. **`mesh_map` puede mentir tras un re-run.** Con `overwrite=False` y la carpeta ya
   existente, se registra `mesh_map[folder] = malla_nueva` (`:118`) y se hace `continue`
   (`:120`) **sin copiar la malla**. El disco conserva la antigua; el JSON anuncia la nueva.

3. **`mesh_map` guarda la ruta fuente, fuera de `outputs/`** (`:118`, `:125`:
   `os.path.abspath(mesh_path)`), no la copia `outputs/<folder>/<basename>`. Si el dataset
   se mueve o se limpia `meshes/`, las rutas quedan rotas — mientras que el basename sí
   sobrevive. Relevante al arreglar **B2**: conviene resolver `mesh_map` con fallback al
   basename dentro de la carpeta, no confiar ciegamente en el `abspath`.

---

## Cómo reproducir

Todas las evidencias salen de scripts cortos contra
`GCI/aoa_0.022_m_1.071`, con el intérprete
`/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python`. Esqueleto:

```python
import warnings
from FotR import FRODO
ROOT = '/home/m.jaraiz/Documentos/DATASETS/data_TIFON/GCI/aoa_0.022_m_1.071'
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    db = FRODO(root_dir=ROOT, format='CODA_SINGLE')
    # B2: 'mesh_map' in db.reader.metadata            -> False
    # B6: db.reader.df_state[['case_idx','folder']]
    # B1: db.residuals.get_df_metrics(['CoefLift'], iter_var=1000, save=False)
    # No fijes cases_idx a mano: el dataset crece y los case_idx se desplazan.
    # Resuélvelos desde df_cases por nombre de carpeta:
    dfc = db.reader.metadata['df_cases']
    idx = dfc.loc[dfc['folder'].isin(['f5', 'f10']), 'case_idx'].tolist()
    db.extract_inputs(id_groups=(3,), cases_idx=idx, vtu_type='surface')
    db.extract_outputs(stage=1, id_groups=(3,), cases_idx=idx, vtu_type='surface')
```

Avisos: pasar siempre `load_in_metadata=False` / `save=False` para no escribir en el
dataset, y usar `MPLBACKEND=Agg` para las funciones de plot. Conviene trabajar con las dos
mallas más pequeñas, `f10` (1082 celdas en el grupo 3) y `f5` (2165): las más finas tienen
`.vtu` de superficie de decenas de MB (`f0.8` llega a 75 MB).
