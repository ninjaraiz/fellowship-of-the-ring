# ANÁLISIS — FRODO + SAM

> Documento de referencia para quien vaya a tocar `FotR/characters/frodo.py`,
> `sam.py` y los subpaquetes `readers/`, `sets/`, `stats/`, `residuals/`.
> El resumen operativo está en [`doc/HANDOFF.md`](../HANDOFF.md); aquí está el detalle.
>
> **Base**: lectura completa del código en el árbol de trabajo sobre `663e3d1`
> (develop) **más los cambios sin commitear** de `residuals/coda.py`, `sets/coda.py`
> y `test_coda_residuals_plot.py`. Los números de línea se moverán cuando eso se
> commitee; los nombres de función no.

---

## §1 Arquitectura efectiva

### FRODO: coordinador delgado

`FRODO` (884 líneas) hace tres cosas y delega todo lo demás:

1. **Montar el cuarteto** (`_set_subclasses`, `frodo.py:118`): busca `self.format` en
   los cuatro registries y construye `reader`, `sets`, `residuals`, `stats`. Si la clave
   existe con valor `None` o falta, deja el atributo a `None` y emite `UserWarning`
   (salvo el reader, que es obligatorio → `ValueError`).
2. **Parsear** (`_parse`, `frodo.py:172`) y **re-sincronizar** (`_sync_reader`,
   `frodo.py:178`): copia `sim_metadata`, `df_state` y `data_dict` del reader a sí mismo.
   Son *aliases*, no copias: el mismo objeto vive en `db.X` y en `db.reader.X`.
3. **Fusionar** (`merge_datasets`, `frodo.py:451`) — la única lógica pesada propia.

Todo lo demás es delegación.

### El `__getattr__` es el mecanismo central, y es frágil

```python
# frodo.py:68-85
for sub in ('sets', 'reader', 'residuals', 'stats'):
    ...
    if obj is not None and hasattr(obj, name):
        return getattr(obj, name)
```

La resolución es **por nombre, en ese orden, sin contrato**. Consecuencias prácticas:

- `db.plot_state()`, `db.metadata`, `db.npy_dict`, `db.define_subset(...)` no existen en
  FRODO: llegan del subobjeto que primero los tenga.
- **El mismo nombre significa cosas distintas según el formato.** `db.extract_inputs`
  acepta `(id_groups, vtu_type, method_to_sort, cases_idx, subset, verbose)` en CODA,
  `(id_groups, cases_idx, subset, verbose)` en NUMPY, `(keys_inputs, keys_aux, ...)` en
  PYLOM/NUMPYFILE y `(p, subset, cases_idx, verbose)` en HORSES3D.
- `sets` va **antes** que `reader`. Si un día una clase Sets define un nombre que ya
  existía en el reader, la delegación cambia de destino en silencio.
- Un typo en un atributo no falla donde se escribió: recorre los cuatro subobjetos y
  acaba en un `AttributeError` que no dice quién lo buscaba.

Este es el punto que más cuidado pide en cualquier refactor, y **no tiene ni un test**
(ver §6).

### SAM: namespace estático

`SAM` (3992 líneas) no se instancia: es un contenedor de clases anidadas.

| Sub-clase | Línea | Contenido |
|---|---|---|
| `Gardener` | `sam.py:56` | Ensamblado de tensores ML (`create_final_tensor`, versión *scored*, concatenación, reducción por frecuencia) |
| `HDF5reader` | `sam.py:625` | Único con `__init__`. Lector fino de `.h5` → numpy/torch |
| `Backpack` | `sam.py:715` | Utilidades de ficheros y malla. Contiene `pattern_pocket` (`:716`), `FilenamePattern` (`:718`), helpers HORSES3D y `HorsesLazyField` (`:2108`) |
| `Weapons` | `sam.py:2224` | Ordenación de nubes de puntos, derivadas de superficie, GMM, interpoladores malla→malla |
| `DifferentialOperators` | `sam.py:3552` | Gradiente / jacobiano / divergencia por MLS sobre nube de puntos |
| `DictVisualizer` | `sam.py:3885` | `rich_tree`, `pretty_print`, `plot_graph` |

**`frodo.py` solo llama a SAM una vez** (`frodo.py:199`, `DictVisualizer.rich_tree`).
Quien consume SAM de verdad son los subpaquetes — ver §5.

---

## §2 Matriz formato × registry

Leída de los cuatro `__init__.py`:

| formato | reader | sets | stats | residuals |
|---|---|---|---|---|
| `CODA` | `CODAReader` | `CODASets` | `CODAStats` | `CODAResiduals` |
| `CODA_SINGLE` | `CODASingleReader` | **—** (warning) | **—** (warning) | `CODAResiduals` (reusado) |
| `NUMPY` | `NUMPYReader` | `CODASets` (reusado) | `CODAStats` (reusado) | **—** (clave ausente → warning) |
| `NUMPYFILE` | `NUMPYFILEReader` | `NUMPYFILESets` | **—** | `None` explícito |
| `PYLOM` | `PYLOMReader` | `PYLOMSets` | **—** | `None` explícito |
| `HORSES3D` | `Horses3DReader` | **—** | **—** | `Horses3DResiduals` |

Dos observaciones que importan:

- **`NUMPY` reutiliza `CODASets` y `CODAStats`** (`sets/__init__.py:33`,
  `stats/__init__.py:24`, comentario «Use the same data_dict format»). Funciona mientras
  esas clases solo toquen `data_dict`. Deja de funcionar en cuanto llaman al reader:
  `CODASets.plot_wall_integrals` hace
  `self.db.reader._resolve_cases_idx(list(cases_idx), None)` (firma de CODA), pero
  `NUMPYReader._resolve_cases_idx(id_group, cases_idx, subset)` interpreta el primer
  argumento como el id de grupo. Sobre una base NUMPY, esa llamada no hace lo que dice.
- **`CODA_SINGLE` es el hueco abierto**: tiene reader y reutiliza residuals, pero
  `db.sets` y `db.stats` son `None`. Es exactamente el siguiente paso previsto.

---

## §3 Contratos de datos

Conviven **dos layouts de `data_dict` incompatibles**, más una variante.

### 3.1 Layout CADGroup — CODA, NUMPY, HORSES3D

```
data_dict['CADGroup_<id>'] = {
    'Coord':          (n_points, n_dim)      # centroides de celda
    'NodeCoord':      (n_nodes,  n_dim)
    'FlCc':           (n_cases,  n_dvars)    # solo design_vars del usuario
    'Conec':          (n_cells,  max_nodes)  # relleno con -1
    'idx_sort':       (n_stages, n_cases, n_points)
    'idx_sort_nodes': (n_stages, n_cases, n_nodes)
    'eltype':         (n_points,)            # celltypes VTK
    'cellOrder':      (n_points,)
    'pointOrder':     (n_nodes,)
    'Vars': { '<stage>': { '<var>': (n_points, n_cases)              # escalar
                                 o (n_dim, n_points, n_cases) } }    # vectorial
    'Aux':  { '<name>': ... }                # opcional
}
```

### 3.2 Layout tres cubos — NUMPYFILE, PYLOM

```
data_dict = {'inputs': {'ptos': ..., <params>...},
             'outputs': {...},
             'aux': {...}}
```

`'ptos'` (o `'xyz'`) es la clave de coordenadas y es obligatoria.

### 3.3 Variante por caso — CODA_SINGLE

Mismas claves que el layout CADGroup, pero **`Coord`, `NodeCoord`, `Conec`, `eltype`,
`cellOrder`, `pointOrder`, `idx_sort` y cada `Vars[stage][var]` son listas alineadas con
`case_order`**, una entrada por caso, porque `npts` difiere entre mallas
(`readers/coda_single.py:333-346`). `FlCc` sigue siendo un array `(n_cases, n_dvars)`.
Añade `mesh_files` y `case_order`.

Además: `mesh` puede ser una design var (factor numérico de malla), así que la identidad
de la malla vive en `sim_metadata[caso]['mesh_info']` y en `df_state['mesh_file']` —
nunca bajo la clave `mesh`, que colisionaría (`readers/coda_single.py:137-140`).

### 3.4 Los dos espacios de índices de caso

Esto es la fuente de confusión más probable al escribir código nuevo:

| | Espacio **global** | Espacio **local** |
|---|---|---|
| Qué indexa | filas de `metadata['df_cases']` | eje de casos ya extraído (`FlCc` del grupo) |
| Quién lo usa | `CODAReader._normalise_cases_idx` / `._resolve_cases_idx`, los *subsets* | `CODASets._normalise_cases_idx` (`sets/coda.py:1675`), `CODAStats._normalise_cases_idx` (`stats/coda.py:1157`) |
| Dónde aparece | `extract_inputs(cases_idx=)`, `extract_outputs(subset=)` | `save_to_npy(case_idx=)`, `compute_stats(cases_idx=)`, `create_jset(idx_flcc=)` |

Los subsets con nombre viven **solo** en el espacio global. Por eso `save_to_npy` no los
acepta: traducir requeriría pasar por `CODAReader.active_cases_idx`, que es el diccionario
que registra, por CADGroup, qué posiciones globales se extrajeron
(`readers/coda.py:867`). Ese puente existe pero nadie lo cruza todavía.

### 3.5 `df_cases` vs `df_state` (CODA)

- `metadata['df_cases']` = **identidad**. Se carga una vez de `cases_metadata.json`, se
  muta *in place* (columnas `folder`, `subsets`) y nunca se reemplaza. Es la verdad.
- `df_state` = **vista derivada y desechable** del estado de ejecución. Se reconstruye
  entera en cada `parse_simulation_dirs()` uniendo lo que hay en disco con `df_cases`.
  Si discrepan, gana `df_cases`.

### 3.6 `merge_datasets`

`frodo.py:451-885`. Principio rector: **`FlCc` es la verdad sobre qué casos se fusionan**,
no `df_cases` ni `df_state` (que pueden describir casos nunca extraídos).

- Sin deduplicación silenciosa: dos fuentes con el mismo caso físico →
  `ValueError` (`_check_no_duplicate_cases`, `frodo.py:270`).
- `df_state` y `df_post` se reconstruyen emparejando **cada fila de `FlCc`** contra la
  tabla origen por identidad de design vars redondeadas
  (`_match_rows_by_identity`, `frodo.py:336`). Si un caso no aparece, o aparece dos
  veces, es `RuntimeError` — nunca un relleno silencioso.
- Núcleo agnóstico al formato; solo el paso de `df_post` es específico de CODA
  (usa `CODAResiduals.get_df_metrics`) y se salta para el resto.

---

## §4 Hallazgos

Ordenados por criticidad. Todas las referencias verificadas contra el árbol de trabajo.
**Ninguno de estos está arreglado**: esta sesión solo documenta.

### [ALTO] `find_files(file_end=...)` — dos llamadas muertas

`readers/coda.py:183` y `readers/coda.py:1309` pasan `file_end=`. La firma real es
`endswith=` (`sam.py:998`). Son `TypeError` seguros en cuanto se ejecuta la línea.

Lo que deja muerto:
- `_infer_metadata_from_folders()` — todo el camino de fallback cuando **no** hay
  `cases_metadata.json`. Es decir: una base CODA sin metadata no se puede abrir.
- `plot_integrals_from_case()` entero (`readers/coda.py:1275`).

Nadie lo ha notado porque los flujos documentados siempre tienen `cases_metadata.json`
(lo escribe GANDALF) y porque los plots de integrales se hacen por la vía de
`CODASets.plot_wall_integrals`, que sí usa `endswith=` (`sets/coda.py:1263-1267`).

### [ALTO] Un caso que falla en `extract_inputs` se convierte en un caso con ceros

`readers/coda.py:847-853`: el `try/except` por caso avisa *"This case was skipped"*,
pero **no lo salta**. Los arrays ya están preasignados con `np.zeros`, el bucle sigue, y
al final se guardan tal cual. Resultado: `FlCc[cont]` queda a `(0, 0, ...)` y
`idx_sort[stage, cont]` a ceros, así que ese caso entra en el dataset como un caso
válido con condiciones de vuelo nulas y un orden de puntos degenerado.

Contraste: `readers/coda_single.py` sí lleva la contabilidad correcta con la lista
`kept` y recorta `FlCc[:len(kept)]` — pero tiene su propio problema, abajo.

### [MEDIO] `coord_idx` llega a `sns.histplot`

`stats/coda.py:1066-1077` llama a `_plot_diff_histogram_2D(..., coord_idx=coord_idx, ...)`,
pero la firma (`stats/coda.py:2020-2028`) no tiene ese parámetro: cae en `**kwargs_plot`,
que se reenvía entero a `sns.histplot(**kwargs_plot)` → `TypeError`.

Se dispara con `plots={'histogram2D': True}`. El hermano `_plot_diff_histogram` tiene el
mismo desajuste de firma pero solo lee `kwargs_plot` con `.get()`, así que sobrevive.

### [MEDIO] `create_pylom_mesh` y `create_NN_pylom` asumen layouts opuestos

`create_pylom_mesh` (`sets/coda.py:338-351`) indexa `eltype[0, :]`, `conec[0, :, :]` y
`cellOrder[0, :]` — es decir, espera un eje de casos al principio. `CODAReader` **no**
produce eso: ahí `eltype` y `cellOrder` son 1-D y `Conec` es 2-D. Ese layout con eje
extra es el que escribe `save_to_npy` (`sets/coda.py:1597-1616`).

`create_NN_pylom` (`sets/coda.py:495-498`) hace lo contrario: usa `eltype.copy()` y
`conec.shape[0]` directamente, coherente con el reader.

Dos métodos hermanos del mismo fichero esperan entradas distintas. Hay que decidir cuál
es el contrato.

### [MEDIO] `plot_residuals_from_case(mode='absolute')` no dibuja nada

`residuals/coda.py:898-903` filtra columnas con `mode in c`. Las columnas absolutas se
llaman `DensityResidual`, `MomentumResidual`… — ninguna contiene la cadena `'absolute'`.
Con `mode='absolute'` la lista sale vacía y la figura queda sin curvas. Solo funcionan
`'norm'` y `'scaled'`, que sí son sufijos reales.

### [MEDIO] CODA_SINGLE: los `append` no son atómicos

`readers/coda_single.py:287-331`. Dentro del `try`, la geometría se añade a `coords` /
`node_coords` / `conecs` / `eltypes` / … **en `stage == 0`**, mientras `kept.append(case_i)`
ocurre al final, tras recorrer todos los stages. Si un caso falla en `stage > 0`
(p. ej. el chequeo de coordenadas consistentes entre stages, `:303-310`), las listas de
geometría se quedan con una entrada de más respecto a `case_order`, `FlCc` y `kept`.

A partir de ahí, `Coord[i]` y `case_order[i]` dejan de referirse al mismo caso.
`extract_outputs` sí detecta el desajuste de selección (`:391-396`) y aborta, lo cual
convierte un fallo parcial en un bloqueo total del grupo.

### [BAJO] `create_jset` aplica `sol` dos veces

`sets/coda.py:227-253`: primero filtra variables por índice contra `sol_num`, y después
pasa `sol=sol` a `create_final_tensor`, que vuelve a aplicar una selección de canal.
Hoy es inocuo — cada tensor llega con un solo canal y la segunda selección es un no-op
guardado por `out.shape[2] > 1` — pero engaña a quien lea el código y se romperá si
algún día se admiten variables vectoriales aquí.

### [BAJO] `save_to_h5` comprueba la existencia antes de normalizar la extensión

`sets/coda.py:1793-1800`: el `os.path.exists(filepath)` se evalúa **antes** del
`if not filepath.endswith('.h5'): filepath += '.h5'`. Con `filepath='db'` y un `db.h5`
existente, `overwrite=False` no protege nada y el fichero se sobrescribe igual.

### [BAJO] `plot_all_final_residuals` con ≥3 design vars produce una figura vacía

`residuals/coda.py:1074-1124`: la figura se crea siempre, pero solo hay rama de dibujo
para `len(dvf) == 1` (que retorna antes) y `len(dvf) == 2`. Con tres o más variables que
varían, se guarda/enseña un lienzo de ejes vacíos sin ningún aviso.

### [DOC] Desajustes docstring ↔ código

- `PYLOMSets.create_jset` documenta que las coordenadas 2-D dan columnas `['x','z']`;
  el código genera `["x","y","z"][:n]` → `['x','y']` (`sets/pylom.py:266`).
- `PYLOMReader.extract_outputs` documenta un `RuntimeError` si no se llamó antes a
  `extract_inputs`; esa comprobación no existe.
- `SAM.Backpack.pattern_pocket.folder_fmt_to_pattern` se cita así en su propio ejemplo,
  pero vive en `pattern_pocket_ant` (ver §7).

---

## §5 Cómo se usa realmente

El contrato *efectivo* no es el de los docstrings, sino el que ejercen los llamadores.

### 5.1 Secuencia canónica por formato

**CODA** — la única completa (`examples/TIFON_database_CODA/`):

```python
db = FRODO(root_dir=..., format='CODA')
db.reader.print_available_cadgroup_ids(stage=1, vtu_type='surface')
db.extract_inputs(id_groups=(3,), cases_idx=[...], vtu_type='surface')
db.extract_outputs(id_groups=(3,), stage=1, cases_idx=[...],
                   var_name_excluded=[...])
db.summary_data()
db.sets.create_jset(stage='1', id_group='3')
db.residuals.get_df_metrics(var_metrics=['CoefLift','CoefDrag'], iter_var=1000)
```

`extract_inputs` **antes** que `extract_outputs` siempre: el segundo consume el
`idx_sort` que construye el primero.

**NUMPY** — round-trip CODA → `.npy` → NUMPY
(`examples/CylinderPINNS_database_CODA_NUMPY/CODA2numpy2.ipynb`). Particularidad:
`NUMPYReader.extract_outputs` **no acepta** `cases_idx` ni `subset` por diseño; replica
la selección que registró `extract_inputs`.

**PYLOM / NUMPYFILE** — API de diccionarios de claves, radicalmente distinta:
`extract_inputs(keys_inputs={'ptos':'xyz', ...}, keys_aux={})`.

**HORSES3D** — solo carga y residuals (`examples/esphere_3D_HORSES3D/basic_load.ipynb`):
no hay sets ni stats en los registries, así que `db.sets is None`.

**CODA_SINGLE** — **ningún ejemplo de lectura**. La única cobertura viva es
`FotR/tests/test_coda_single.py`. Ojo: `examples/TIFON_database_CODA/coda_single_GCI.ipynb`
está mal nombrado — usa `format="CODA"` sobre las carpetas GCI.

### 5.2 Quién consume SAM

| Símbolo SAM | Consumidores |
|---|---|
| `Backpack.get_df_from_csv` | `residuals/coda.py` ×5, `sets/coda.py` ×2, `readers/coda.py` ×1 |
| `Gardener.create_final_tensor` | `sets/numpy_file.py`, `sets/pylom.py`, `sets/coda.py` |
| `Backpack.pattern_pocket.find_files` | `residuals/coda.py`, `readers/horses3d.py`, `readers/coda.py`, `sets/coda.py` |
| `Weapons.sort_*` | `readers/coda.py`, `readers/coda_single.py`, `readers/numpy_file.py`, `sets/coda.py`; HORSES3D solo `sort_lexsort` |
| `Backpack.{get_unified_connectivity, ensure_cell_data, same_columns}` | `readers/coda.py`, `readers/coda_single.py` |
| `Backpack.{read_horses_mesh_h5, read_horses_hsol_header, HorsesLazyField}` | solo `readers/horses3d.py` |
| `Weapons._interpolate_*` | solo `sets/coda.py` |
| `DictVisualizer.rich_tree` | `frodo.py:199` — única llamada a SAM desde frodo |

Es decir: **tocar SAM afecta a los subpaquetes, no a FRODO**. Eso es lo que permite que
los tests stubeen SAM con una decena de métodos no-op.

### 5.3 LEGOLAS es el único consumidor externo de FRODO

`legolas.py:21` recibe la instancia y depende de:

- atributos: `db.format`, `db.data_dict`, `db.df_state`
- `db.residuals.plot_residuals_from_case`, `.plot_all_final_residuals`,
  `.get_all_final_residuals`, `.get_df_metrics`
- `db.plot_state` vía `hasattr` (`legolas.py:738-740`) — **existe solo gracias al
  `__getattr__`**, porque el método vive en `CODAReader`
- la estructura de `data_dict`: `['CADGroup_X']['Vars'][stage][var]` y `['Coord']` para
  CODA; `['inputs']['ptos']` o `['Coord']` para el resto

Ese es el contrato que un refactor de FRODO no puede romper. GANDALF, ARAGORN y GIMLI
**no** tocan FRODO.

### 5.4 Contrato mínimo de `db` según los dobles de test

Los `_FakeDB` de los tests son la documentación más precisa del acoplamiento real:

| Qué se prueba | `db` necesita |
|---|---|
| `CODAResiduals` (`test_coda_residuals_plot.py:53`) | `root_dir`, `format`, `sim_metadata[folder]={'path','stages'}`, `metadata{folder_fmt, design_vars, num_stages, df_cases}`, `df_state` |
| `CODASets.plot_wall_integrals` (`test_coda_wall_integrals.py:43`) | `root_dir`, `metadata`, `reader` con `_resolve_cases_idx` y `subsets` |
| `CODASets.save_to_npy` (`test_save_to_npy.py:128`) | `data_dict`, `name` |
| `merge_datasets` (`test_merge_datasets.py:206`) | `format`, `name`, `metadata['design_vars']`, `data_dict`, `df_state`, `sets.interpolate_msh2msh`, `residuals.get_df_metrics` |

### 5.5 Llamadas rotas en `examples/` (verificadas)

- `examples/TIFON_database_CODA/basic.ipynb`:
  `db.sets.add_to_data_dict(..., key_location='Airfoil')` — el parámetro se llama
  `array_name` (`sets/coda.py:605`) → `TypeError`. En `CODA2numpy2.ipynb` sí se usa bien.
- `examples/gandalf/{TIFON1,TIFON2,cylinder}.ipynb`: llaman `GANDALF.Backpack.isa_atmosphere`
  y `.Sutherland_law`. `GANDALF._NON_DELEGATED = frozenset({'Backpack'})`
  (`gandalf.py:49`) bloquea esa delegación a propósito; los métodos reales están en
  `BaseRing.Backpack` (`rings/base.py:1331` y `:1402`).

### 5.6 El `.npy` del formato NUMPY viaja sin metadata

El propio flujo documentado tiene que restituir las design vars a mano:

```python
# doc/frodo_sam/flows/flow_numpy.py:57-58
# El .npy viaja sin metadata: se restituyen las design_vars de origen.
db2.metadata["design_vars"] = list(db.metadata["design_vars"])
```

No es un bug, es una limitación del formato: `save_to_npy` escribe un único CADGroup sin
`cases_metadata.json` acompañante, y `NUMPYReader` cae en nombres genéricos `dv_0`,
`dv_1`… si no lo encuentra (`readers/numpy.py:329-331`). Conviene saberlo antes de
construir nada encima.

---

## §6 Cobertura de tests: dónde no hay red

`FotR/tests/` — 9 ficheros, **162 tests en verde**, sin `conftest.py`. Los ficheros usan
stubs mutuamente incompatibles (para no arrastrar pyvista/torch/pyLOM) y el aislamiento
depende del orden de importación: si se añade un test que importe el `FotR` real antes
que los demás, hay que revisar ese orden.

Lo que **sí** está cubierto: `merge_datasets` (20 tests), subsets de CODA (22),
`NUMPYReader` (26), `Horses3DReader` descubrimiento (32), CODA_SINGLE (7),
`save_to_npy` (26), `plot_wall_integrals` (4), rings/GANDALF (21), rama 1-D de
`plot_all_final_residuals` (4).

Lo que **no**:

- **SAM: cero tests reales.** En los 9 ficheros se stubea o se ignora. Sin red: todo
  `Gardener`, `pattern_pocket`/`find_files`/`FilenamePattern`, `Weapons.sort_*`,
  `surface_derivative`, `GMM`, los interpoladores, `DifferentialOperators`,
  `DictVisualizer`, `HDF5reader`.
- **De FRODO solo `merge_datasets`.** Sin cubrir: `__getattr__` (el mecanismo más
  frágil del diseño), `_set_subclasses` con registry `None`, `copy()`, `_sync_reader`,
  los passthrough de `extract_*`.
- **`stats/coda.py` (2062 líneas): cero tests.** `residuals/horses3d.py`: cero.
  `readers/pylom.py`, `readers/numpy_file.py`, `sets/pylom.py`, `sets/numpy_file.py`: cero.
- De `sets/coda.py` solo `save_to_npy` y `plot_wall_integrals`; el resto
  (`create_jset`, `create_NN_pylom`, `create_pylom_mesh`, `interpolate_*`,
  `crop_bounding_box`, `change_order_coord`, `save_to_h5`) sin cubrir.
- LEGOLAS, ARAGORN, GIMLI y EarendilsLight: sin tests.

Esto es **contexto de riesgo**, no una propuesta de trabajo. Pero explica por qué
conviene ser conservador: hoy un refactor de SAM o del `__getattr__` va a ciegas.

---

## §7 Deuda estructural

### `pattern_pocket_ant`: ~435 líneas de código muerto

`sam.py:1268-1704`. Duplicado completo de `pattern_pocket` (dataclass `FilenamePattern`
propia, `_tokenize_filename`, `find_files`, `infer_filename_pattern`) más
`folder_fmt_to_pattern`. **Cero llamadores en todo el repo.** Matices:

- `infer_filename_pattern` construye y devuelve la `FilenamePattern` de `pattern_pocket`
  (la nueva), no la suya; la dataclass `pattern_pocket_ant.FilenamePattern` no se usa
  jamás.
- `folder_fmt_to_pattern` solo existe aquí, y su propio docstring la cita como
  `SAM.Backpack.pattern_pocket.folder_fmt_to_pattern` — ruta que no existe. Su sustituta
  es `FilenamePattern.from_template(..., numeric=True).compiled`, que es lo que usan
  `readers/coda.py` y `residuals/coda.py`.

### Asimetrías de firma entre formatos

Importan **porque el `__getattr__` las expone todas bajo el mismo nombre**:

| método | CODA | NUMPY | HORSES3D |
|---|---|---|---|
| `_resolve_cases_idx` | `(cases_idx, subset)` | `(id_group, cases_idx, subset)` | `(subset, cases_idx)` |
| `define_subset` | `(name, cases_idx, overwrite)` | `(id_group, name, cases_idx, overwrite)` | `(name, case_idx)` |
| `extract_inputs` | `(id_groups, vtu_type, method_to_sort, cases_idx, subset, verbose)` | `(id_groups, cases_idx, subset, verbose)` | `(p, subset, cases_idx, verbose)` |

El orden `(subset, cases_idx)` de HORSES3D está invertido respecto a CODA. Nada lo
impide y nada avisa.

### HORSES3D contamina el nivel superior de `data_dict`

`readers/horses3d.py:1023-1030` escribe `data_dict['FlCc']` y `data_dict['case_order']`
como alias del primer grupo, «por compatibilidad». El problema es que `save_to_h5` y
`merge_datasets` recorren `data_dict.items()` asumiendo que **todo** valor es un dict de
CADGroup: se encontrarán un `ndarray` y una `list` mezclados con los grupos.

### Imports pesados a nivel de módulo

`sam.py` importa torch, pyvista, seaborn, sklearn, h5py y matplotlib en el import.
Cualquier `from FotR import FRODO` arrastra todo el stack. Es la razón directa de que
los tests tengan que stubear SAM en vez de importarlo.

### Duplicaciones funcionales vivas

- `BaseRing.Backpack` (`rings/base.py:1239`) reimplementa parte de `SAM.Backpack`
  (`isa_atmosphere`, `Sutherland_law`, utilidades SLURM) **sin importarla**. Ningún
  fichero de `rings/` referencia `SAM`.
- `GIMLI` declara en su docstring que sustituye a `SAM.Weapons.GMM()`, pero `GMM` sigue
  vivo en `sam.py:2985` y es lo que siguen llamando los examples. No hay ni un ejemplo
  que use `GIMLI`.

### `EarendilsLight`: el sistema de ayuda no funciona donde más haría falta

Patrón `light = EarendilsLight(__name__)` + `some_light()` en FRODO (`frodo.py:56`),
SAM (`sam.py:46`), GANDALF, `BaseRing` y `xgboost_preliminar`. No lo tienen LEGOLAS,
ARAGORN ni GIMLI.

Estado real: **sin uso en el repo** (ni una llamada a `.some_light(...)`), sin tests
(siempre stubeado), y el propio README lo describe como «de momento va a pilas y no
funciona muy bien». La razón técnica concreta: `_render_signature`
(`EarendilsLight.py:229-231`) llama `get_type_hints(fn)` dentro de un `try/except`
amplio que devuelve `"⚠️ Firma no disponible"` al fallar. Las anotaciones en string tipo
`db: 'FRODO'` bajo `if TYPE_CHECKING` — que son exactamente las de `sets/base.py`,
`stats/base.py`, `residuals/base.py` y `legolas.py` — **no resuelven**, así que la ayuda
se rinde justo en las clases que más interesan.

Además `_render_signature` mezcla render y valor de retorno: imprime en consola y
devuelve `""`.

---

## §8 Siguiente paso previsto

`sets` y `stats` propios de CODA_SINGLE, más un helper de orden GCI/Richardson.

Precondiciones ya verificadas en el reader — está todo listo para consumirlo:

- `data_dict[key]['case_order']` da el orden canónico de casos.
- `data_dict[key]['mesh_files']` da la identidad de malla por caso.
- `data_dict[key]['idx_sort'][i][stage]` da el sorter del caso `i` en ese stage.
- `FlCc` ya viene recortado a los casos realmente extraídos (`FlCc[:len(kept)]`).
- `active_cases_idx[key]` mapea el eje local de vuelta a posiciones globales de `df_cases`.

Lo que hay que diseñar: `create_jset` con `npts` variable por caso (el
`SAM.Gardener.create_final_tensor` actual asume un `tensor_ptos` único compartido, así
que **no sirve tal cual**) y `compute_stats` operando sobre listas en lugar de arrays
`(n_points, n_cases)`.

Antes de eso conviene decidir qué hacer con los dos hallazgos [ALTO] del §4, porque el
de `extract_inputs` afecta a la fiabilidad de cualquier dataset construido con CODA.
