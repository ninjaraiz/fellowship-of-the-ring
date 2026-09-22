# Diseño del `sets` para CODA_SINGLE

> Documento de decisión. Recoge el problema, las restricciones verificadas y la
> arquitectura.
>
> **Estado: implementado.** `FotR/characters/sets/coda_single.py` y
> `FotR/characters/stats/coda_single.py`, registrados en sus registries, con
> 32 tests en `FotR/tests/test_coda_single_sets.py`.
> Contexto: [`BUGS_CODA_SINGLE.md`](BUGS_CODA_SINGLE.md) (auditoría, B10) ·
> [`ANALISIS.md`](ANALISIS.md) (arquitectura general).

## 1. El problema

`CODASets` se escribió sobre un invariante que CODA_SINGLE rompe por definición: **una
sola malla compartida**. De ahí que `data_dict` guarde un único `Coord (npts, 3)` y cada
variable como matriz densa `(npts, ncases)` con el eje de casos alineado con `FlCc`.

En CODA_SINGLE cada caso tiene su malla, así que `Coord` y `Vars[stage][var]` son listas
por caso con `npts_i` distintos. **No existe eje de puntos común**, luego nada puede ser
una matriz indexada por (punto, caso). Los siete puntos de entrada de `CODASets` fallan.

La pregunta no es «¿adaptamos o reescribimos?» sino **qué significa "juntar" cuando las
mallas difieren**, porque significa cosas distintas según para qué.

## 2. Las cinco misiones y qué pide cada una

| misión | qué necesita | ¿malla común? |
|---|---|---|
| M1 · Convergencia de malla (GCI) | reducir cada malla a algo comparable | **no** — comparar mallas *es* el objetivo |
| M2 · Surrogate ML pointwise | tabla plana, una fila por (punto, caso) | **no** — mallas distintas = muestreo diverso |
| M3 · Exportar con malla (pyLOM/VTK) | geometría + campos | **no**, pero un artefacto por caso |
| M4 · Round-trip npy/h5 | contenedor que admita ragged | **no** |
| M5 · Álgebra de variables por caso | operar variables dentro de un caso | **no** |

**Ninguna de las cinco necesita malla común.** Solo la necesita reusar la maquinaria
densa de CODA. Por eso la interpolación tiene que ser **una operación explícita**, nunca
el camino por defecto: en un GCI, interpolar a una malla de referencia destruye justo la
señal que se quiere medir.

## 3. Restricciones verificadas

Medidas sobre el GCI real (`GCI/aoa_*`, 3 raíces × 6 casos):

| | valor |
|---|---|
| Geometría del grupo de pared (3), todo el estudio | **12 MB** |
| Variables del grupo 3, 2 stages × 3 raíces | **11 MB** |
| Geometría del grupo de volumen (2), por raíz | **363 MB** (1.1 GB las tres) |
| Releer + extraer un `.vtu` de 78 MB | **0.04 s + 0.01 s** |

Lectura: para el estudio actual la carga perezosa **no ahorra memoria relevante**. La
justifica el día que se extraiga volumen, y el hecho de que releer es casi gratis. Se
adopta por diseño, sabiendo que hoy el ahorro es cosmético.

Estructurales, comprobadas contra el fuente de pyLOM (`/home/m.jaraiz/repos/pyLowOrder/`):

- **`Mesh(mtype, xyz, connectivity, eltype, cellOrder, pointOrder, ptable)`** y su `xyz`
  son **nodos**, no centroides (`mesh.py:70`, y su `__str__` los cuenta como nodos).
  `CODASets.create_pylom_mesh` pasa `Coord` (centroides) con un `Conec` que indexa
  `NodeCoord`: está mal por partida doble, y además indexa `conec[0,:,:]` / `eltype[0,:]`,
  layout que `CODAReader` no produce. **No sirve como modelo.**
- **`Dataset` tiene un solo `xyz`** (`dataset.py:25-44`) y sus campos son rectangulares:
  el writer deduce `ndim = value.shape[0] // npoints` (`io_h5.py:366-372`), así que
  `value.shape[0]` debe ser exactamente `ndim*npoints`. Además escribe con
  `dset['value'][istart:iend,:]` (`io_h5.py:394`), luego **un campo 1-D `(npoints,)`
  revienta al guardar**. → n mallas = n `Dataset`.
- **`Dataset.save` no escribe la malla**; `Mesh.save` es independiente.
- `SAM.Gardener.create_final_tensor` (`sam.py:59`) exige malla común por construcción
  (`repeat` / `repeat_interleave`). **Pero su salida ya es una tabla plana**, y esa forma
  sí se construye con mallas distintas.
- `NUMPYReader` rechaza ragged (`numpy.py:803`, `:999`), **pero admite varias claves
  top-level**, cada una con su propio espacio de casos (`numpy.py:64-74`).
- **La geometría nunca sale del `.msh`**: ese fichero solo se usa como identidad
  (basename). Toda la geometría sale de los `.vtu`. «Las mallas ya están en la carpeta
  base» son, en realidad, los `.vtu`.

## 4. Decisión

**Clase nueva `CODASingleSets(BaseSets)` en `sets/coda_single.py`.** No se adapta
`CODASets` con ramas: son contratos de datos distintos, y meter una bifurcación en cada
uno de sus 17 métodos —de los que hoy solo 2 tienen test— es pedir una regresión en la
ruta que ya funciona.

El puente hacia la maquinaria densa **es un método, no una rama**: `sample_on()`.

### Capa 0 — `CaseRef`: la dirección y la lógica, no la malla

Un objeto por (grupo, caso) que sabe cómo volver a leer lo suyo: carpeta, `.vtu` por
stage, ids del grupo CAD, método de ordenación y el `idx_sort` ya calculado. Expone
`coord()`, `nodes()`, `conec()`, `var(name, stage)`, `varnames(stage)`, `npts`.
Materializa al pedirlo, con caché opt-in.

Tiene precedente en el repo: `SAM.Backpack.HorsesLazyField` (`sam.py:2180`) hace esto
para HORSES3D y `SAM.DictVisualizer` ya sabe representarlo. Reutiliza
`SAM.Backpack.cell_connectivity_in_order` y `SAM.Weapons.sort_*`.

**Cómo quedó.** El lector no cambió su contrato: sigue materializando en
`extract_inputs`. Lo que sí guarda ahora son `group_ids`, `vtu_type` y `method_to_sort`
en el grupo, que es lo que permite a `CaseRef` **releer un caso cuando el array no
está**. La resolución es: array denso si existe → si no, relectura vía
`CODASingleReader.read_case_geometry` / `read_case_vars`. Esos dos métodos se
factorizaron *fuera* de `extract_inputs`, que ahora los usa, así que la ruta ansiosa y
la perezosa comparten implementación y no pueden divergir.

### Capa 1 — `CODASingleSets`

| método | misión | qué hace |
|---|---|---|
| `create_jset(...)` | M2 | **No hay ensamblador nuevo**: `SAM.Gardener.create_final_tensor` una vez por caso (`n_cases=1`, su propio `Coord`) y unión con `SAM.Gardener.concatenate_sets` (`sam.py:469`). Añade `case_offsets`, `case_ids` y `case_order`, y una columna `case` en `df_data`. Mismo dict de salida que CODA → cumple el `@abstractmethod` de `BaseSets`. |
| `compute_var(name, formula, stage)` | M5 | Álgebra por caso reutilizando `BaseRing._eval_formula` (`rings/base.py:341`), que evalúa sin `eval` y —verificado— funciona sobre arrays numpy tal cual. Hay que subirlo a SAM para que `sets` no importe `rings`. |
| `reduce_cases(var, how, stage)` | M1, M5 | Una fila por caso: `mean/median/min/max/std/sum/rms/absmax` o un callable, + design vars, `n_points` y `stages_available`. Alimenta el GCI y deja ver tendencias entre simulaciones. |
| `sample_on(target, method='idw')` | puente | Remuestrea todos los casos sobre puntos comunes (sonda, curva, o la malla de un caso) y **recupera el layout denso `(npts, ncases)`**. A partir de ahí vale toda la maquinaria de `CODASets`. Reutiliza `SAM.Weapons._interpolate_idw_tree`. |
| `to_pylom(...)` | M3 | Un `Dataset` por caso, con `idim` densos y campos siempre ≥2-D (`(npoints,1)` para escalares). El `Mesh` queda fuera: su `xyz` son nodos y exige una conectividad de ancho fijo, y `Dataset.save` no lo escribe de todas formas. |
| `save_to_npy` / `save_to_h5` | M4 | **Una clave top-level por caso** (`CADGroup_3__f1`, …), cada una con su `FlCc` de una fila: el round-trip funciona **sin tocar `NUMPYReader`**. En HDF5, un subgrupo por caso. |
| `plot_wall_integrals` | M1 | **No necesita malla**: lee los `.dat` de integrales. Hoy vive en `CODASets` por accidente histórico y el notebook del GCI ya lo llama sobre CODA_SINGLE. |

### Capa 2 — GCI

`reduce_cases` produce la tabla (magnitud vs. refinamiento); el orden de convergencia
observado y el índice GCI van aparte, en `stats/coda_single.py`: es análisis, no
ensamblado. `CODASingleStats.richardson` resuelve el orden `p` por punto fijo sobre cada
tripleta consecutiva, con `h ∝ N**(-1/dim)` (solo entran las razones, así que la
constante se cancela). Validado contra `f(h) = f_exact + C·h^p`: recupera `p` y
`f_exact` exactamente para `p = 1, 2, 2.5`. Las tripletas mal planteadas —dos mallas con
el mismo valor, o que no refinan (`r = 1`)— devuelven `NaN` con `converged=False` en vez
de un número con pinta de autoridad.

### Helpers a subir a `BaseSets`

Ya están duplicados, así que no es refactor especulativo:
`_save_result` (idéntico en `sets/coda.py:1864`, `sets/numpy_file.py:337`,
`sets/pylom.py:706`) y `_normalise_cases_idx` (`sets/coda.py:1675`, `stats/coda.py:1157`).

## 5. Trampas a evitar al implementar

- **Importar pyLOM dentro del método**, como `sets/pylom.py:535`, nunca en la cabecera
  como `sets/coda.py:31`: el `except ModuleNotFoundError` de `sets/__init__.py` se lo
  traga y el formato desaparece del registro sin decir por qué (bug B12).
- **`concatenate_sets` no recalcula `scaled`**: toma `mins`/`maxs` del set de referencia
  pero deja cada bloque escalado con los suyos. Hay que renormalizar al final, o pasar un
  `ref=` global en dos pasadas.
- **`save_to_h5` recorre todo `data_dict` con acceso directo por clave**: reventaría con
  `mesh_files`, `case_order` y `stages_available`, que no son arrays.
- **`SAM.Backpack.create_tensors_from_h5` es código muerto** apuntando a una jerarquía
  (`Mesh/…`) que ningún escritor produce. No tomarlo como contrato.
- **`create_NN_pylom` pone `idim=0` en todas las variables paramétricas**, lo que
  contradice `Dataset.X()`, que espera `idim` densos 0,1,2… `sets/pylom.py:551` sí lo
  hace bien.
- **Las entradas de `Vars` pueden ser `None`** (caso sin ese stage o sin esa variable):
  todo consumidor tiene que contemplarlo.

## 6. Fuera del alcance de `sets`

**LEGOLAS no se arregla con esto.** Despacha por `format == "CODA"` exacto
(`legolas.py:98,192,214`), así que CODA_SINGLE cae en la rama NUMPYFILE y termina en
`ValueError: No coordinates found for plotting`. Aunque se amplíe ese despacho, seguiría
fallando porque `Coord` es una lista y `_get_coda_variable` hace `arr.ndim`. Necesita su
propio cambio: indexar por caso. Hay además una incoherencia previa de layout 3-D —
LEGOLAS interpreta `(n_points, n_dims, n_cases)` y CODA almacena
`(n_dim, n_points, n_cases)`.

## 7. Consecuencia que conviene aceptar de frente

Un artefacto exportado perezoso **no es portable**: si guarda rutas y se mueve el
dataset, se rompe. Recomendación: **perezoso en memoria, autocontenido al exportar**, con
una opción explícita para quien quiera solo el manifiesto de rutas. Es la misma lección
que dejó `mesh_map` apuntando fuera de `outputs/` (bug B2).

## 8. Orden de ataque (completado)

1. **`plot_wall_integrals` disponible en CODA_SINGLE.** Es lo único que el notebook
   necesita hoy y no depende de nada de lo demás. Desbloquea el flujo ya.
2. **`CaseRef` + `CODASingleSets` mínimo**: `reduce_cases` y `compute_var` (M1 y M5),
   que es donde está el trabajo real del GCI.
3. **`create_jset` largo** (M2) y round-trip npy/h5 (M4).
4. **`sample_on`** y, encima, `to_pylom` (M3).
5. **Helper GCI/Richardson** en `stats/coda_single.py`.
