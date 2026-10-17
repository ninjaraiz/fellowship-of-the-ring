# HANDOFF — Fellowship of the Ring

> Lee esto primero. Para el detalle (arquitectura, contratos de datos, bugs
> localizados, cobertura de tests) → [`doc/frodo_sam/ANALISIS.md`](frodo_sam/ANALISIS.md).
> Para usar una clase, los visores de `doc/` (§7). No releas el repo entero.

## 1. Estado — `develop` @ `fb6e486`, **con trabajo sin commitear**

Suite: **239 passed** (215 + 24 del lector `.hsol`).

```bash
/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q
```

Sin commitear, cuatro bloques independientes:

1. **Imports perezosos** (§8) — toca `FotR/__init__.py`, los cinco `__init__.py` de
   registries, `sam.py`, `legolas.py`, `gimli.py`, `readers/coda.py`, `residuals/*`,
   `sets/*`, `stats/coda.py`, `rings/base.py`, y añade `FotR/_lazy.py` (nuevo,
   sin trackear).
2. **Documentación** — narrativa de los dos visores reescrita con el tono del
   repositorio, `doc/gandalf/` entero, y este fichero.
3. **Examples** — `coda_single_GCI*.ipynb`, `build_gci_report.py`, `TIFON2.ipynb`,
   `df_post_analysis.ipynb`, más `examples/gandalf/coda_1sim.py` y `coda_TIFON.py`
   (nuevos, sin trackear). Son tuyos, no los he tocado.
4. **Lector `.hsol`/`.hmesh` de HORSES3D, hito M1** (§9) — `FotR/solver_tools/horses3d.py`
   y `FotR/tests/test_horses3d_hsol.py` (nuevos, sin trackear), y
   `FotR/solver_tools/__init__.py` perezoso. No toca ningún fichero de los otros bloques.

Commits recientes: `cc963aa` (docs + GCI TIFON cerrado + merge_datasets),
`da04abb` (arranque de la doc de GANDALF), `fb6e486` (rans5 en examples).

Aviso sobre los tests: los ficheros usan **stubs mutuamente incompatibles** y el
aislamiento depende del orden de importación. El motivo original (no arrastrar
pyvista/torch/pyLOM) ya no existe desde §8, pero los stubs siguen ahí y siguen
siendo frágiles.

## 2. Decisiones vigentes que no están en el código

* `FlCc` = solo `design_vars` del usuario, nunca params del solver
  (`readers/coda.py`, `readers/horses3d.py`).
* HORSES3D: `p` único exigido en `extract_outputs`, `.hsol` header+lazy
  (`HorsesLazyField`, cuyo `load()` lanza `NotImplementedError` hasta M2, §9).
  **Cambiado el 10/10/2026:** `MESH/*.h5` sigue siendo la geometría por defecto, pero se
  permite leer el `.hmesh` (coordenadas exactas de los nodos de solución, opción
  `geometry='hmesh'`) y los snapshots `*_<iter>.hsol` (series). bmesh/pmesh/cgns, nunca.
* `GANDALF.Backpack` NO se delega: solo existe `BaseRing.Backpack`
  (`gandalf.py`, `_NON_DELEGATED`). Y `BaseRing` no se re-exporta en
  `FotR/__init__.py`, así que la ruta buena es
  `from FotR.characters.rings import BaseRing`.
* En CODA_SINGLE, `mesh` puede ser design var (factor numérico); la identidad de malla
  vive en `sim_metadata[*]['mesh_info']` y `df_state['mesh_file']`, nunca en la clave
  `mesh`, que colisionaría.
* `data_dict['CADGroup_X']` en CODA_SINGLE guarda geometría y `Vars` como **listas por
  caso** (los `npts` difieren).
* Sin tocar a propósito: LEGOLAS (pendiente de reestructuración), columnas 2-D
  `['x','y']` vs `['x','z']` en `PYLOMSets`, orden de args `(subset,cases_idx)` vs
  `(cases_idx,subset)` entre readers.
* `xgboost_preliminar.py` es un módulo huérfano: nadie lo importa, hereda de
  `XGBClassifier` en la propia definición de clase y declara una clase llamada
  `GANDALF` que no tiene nada que ver con la de verdad.

Heredada del handoff anterior, **no verificable desde aquí**: `run_sst_v4.py` (fuera
del repo), solo se tocó `alt = ALT_PLACEHOLDER`. No tocar nada más de ese fichero.

## 3. Mapa

| pieza | dónde | qué es |
|---|---|---|
| `FRODO` | `FotR/characters/frodo.py` | coordinador: monta el cuarteto desde los registries, delega por `__getattr__`, y `merge_datasets` |
| `SAM` | `FotR/characters/sam.py` | namespace estático: `Gardener`, `HDF5reader`, `Backpack`, `Weapons`, `DifferentialOperators`, `DictVisualizer` |
| readers | `characters/readers/` | CODA · CODA_SINGLE · NUMPY · NUMPYFILE · PYLOM · HORSES3D |
| sets | `characters/sets/` | CODA (lo usa también NUMPY) · CODA_SINGLE · NUMPYFILE · PYLOM |
| stats | `characters/stats/` | CODA (lo usa también NUMPY) · CODA_SINGLE |
| residuals | `characters/residuals/` | CODA (lo reusa CODA_SINGLE) · HORSES3D |
| generadores | `characters/rings/` + `gandalf.py` | DoE, SLURM, `cases_metadata.json` |
| plots | `characters/legolas.py` | **único consumidor externo de FRODO** |
| imports perezosos | `FotR/_lazy.py` | `lazy_module`, `LazyRegistry` (§8) |
| documentación | `doc/frodo_sam/` · `doc/gandalf/` | dos visores HTML (§7) |

## 4. Trampas conocidas

* `db.<lo-que-sea>` puede venir de `sets`, `reader`, `residuals` o `stats`, en ese orden,
  y **la misma llamada tiene firma distinta según el formato** → ANÁLISIS §1 y §7.
  Ejemplo real: `extract_inputs` es `id_groups=(3,)` en CODA y `id_groups='1'` en NUMPY.
* `db.sets` / `db.stats` son `None` en HORSES3D; `db.residuals` es `None` en NUMPY,
  PYLOM y NUMPYFILE → matriz completa en ANÁLISIS §2.
* Hay **dos espacios de índices de caso** (global sobre `df_cases` vs local sobre el
  `FlCc` ya extraído) y los subsets solo viven en el global → ANÁLISIS §3.4.
* **SAM no tiene ni un test.** De FRODO solo se testea `merge_datasets`.
  `stats/coda.py` está a cero → ANÁLISIS §6.
* `EarendilsLight` no funciona sobre FRODO ni SAM: `some_light()` lanza
  `TypeError: sequence item 0: expected str instance, Syntax found`. **Es anterior a
  los imports perezosos** (verificado con `git stash` el 10/10/2026), no una regresión.
* `SAM.Backpack` y `BaseRing.Backpack` **no son la misma cosa** y no comparten un solo
  método: la primera lleva ficheros y mallas, la segunda atmósfera, viscosidad y Slurm.
* `BaseRing.Backpack.normalize_airfoil` **muta el array de entrada** si es `float64`
  (`np.asarray` solo copia al cambiar dtype); con enteros no. El docstring ya lo avisa;
  el arreglo sería `np.array` en vez de `np.asarray`, pero cambia comportamiento y está
  sin hacer.
* Dos examples están rotos: `basic.ipynb` llama `add_to_data_dict(key_location=...)`
  (es `array_name`) y hay notebooks de `examples/gandalf/` que llaman
  `GANDALF.Backpack.*` → ANÁLISIS §5.5.
* **CODA_SINGLE está completo**: los 14 bugs auditados están arreglados, B12 incluido
  (§8). Catálogo en [`BUGS_CODA_SINGLE.md`](frodo_sam/BUGS_CODA_SINGLE.md); diseño de
  sets en [`DISENO_SETS_CODA_SINGLE.md`](frodo_sam/DISENO_SETS_CODA_SINGLE.md).
  Lo que **no** se tocó: la misma conectividad errónea sigue en la ruta de CODA y
  LEGOLAS sigue sin soportar este formato.

## 5. Siguiente paso previsto

* **LEGOLAS** — lo último del repo según tu plan. Despacha por `format == "CODA"`
  exacto, así que CODA_SINGLE cae en la rama NUMPYFILE y muere con «No coordinates
  found for plotting». Ampliar el despacho no basta: `Coord` es una lista y
  `_get_coda_variable` hace `arr.ndim`.
* **La conectividad de CODA** sigue con el defecto que se arregló en la ruta
  CODA_SINGLE. El arreglo está disponible en `SAM.Backpack.cell_connectivity_in_order`.
* **Declarar pyLOM** en `requirements.txt`/`setup.py` si se quiere que el formato
  PYLOM funcione de serie. B12 ya no es un fallo silencioso, pero la dependencia sigue
  sin declararse.
* **Los stubs de los tests** ya no hacen falta por coste de import. Simplificarlos
  quitaría la fragilidad de orden, pero no es urgente.

## 6. Arranque mínimo

```bash
git log --oneline -8 && git status --short
/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q
python3 -m http.server -d doc 8000     # los visores
```

## 7. Documentación

Dos visores HTML, misma maquinaria, cada uno autocontenido. **Sírvelos desde `doc/`**,
no desde su subcarpeta: se enlazan entre sí con rutas `../` y si no dan 404.

| visor | qué cubre |
|---|---|
| `doc/frodo_sam/` | Inicio (mapa + quién es quién + matriz de formatos) · FRODO (una pestaña por formato, dentro una por pieza) · SAM (una por área) · `data_dict` · Flujos · buscador global |
| `doc/gandalf/` | GANDALF (la fachada) · Rings (una pestaña por ring, dentro una por fase de la campaña) · Backpack · Problemas |

Reglas de mantenimiento:

* `docs.json` es **100 % derivado**: nunca se edita a mano. Se regenera con
  `python3 doc/<carpeta>/generate_docs.py`. Lee los registries por AST y entiende la
  forma `LazyRegistry(__name__, {...})`; si cambias la forma de un registry, hay que
  tocar ese parser (ya pasó una vez y dejó las tablas vacías).
* `guides.json` (narrativa larga ES/EN) e `i18n.json` (cadenas de interfaz) van a mano,
  **las dos ramas de idioma**, con la misma estructura de bloques.
* `examples.json`: ejemplos tomados de `examples/`, no inventados.
* Los docstrings son la fuente de verdad del API y van **en inglés**; el tono del
  repositorio vive solo en la narrativa. El generador entiende numpydoc y estilo
  Google (`Args:`), que es el que usan los rings.
* En esta máquina no hay runtime JS: el JS de los visores solo se puede validar
  estáticamente. La comprobación visual la haces tú.

## 8. Imports perezosos (10/10/2026)

Medido antes: `import FotR` **3362 ms**. Después: **3 ms**, y `from FotR import FRODO`
292 ms. Un flujo CODA completo no llega a cargar torch, seaborn, scikit-learn ni plotly.

Tres mecanismos, todos en `FotR/_lazy.py`:

1. **`lazy_module('torch')`** — sustituto que importa en el primer acceso a un atributo.
   Se asigna a nivel de módulo y **los cuerpos de las funciones no cambian**.
2. **`LazyRegistry`** — los cuatro registries y el de rings mapean
   `formato → '.modulo:Clase'` e importan en el primer `.get()`. Abrir CODA ya no carga
   los demás readers.
3. **`__getattr__` de `FotR/__init__.py`** (PEP 562) — cada personaje se carga al
   nombrarlo.

**Reglas al añadir código pesado:**

* `from libreria import Nombre` **no se puede diferir** con el proxy (el nombre se liga
  al importar): va dentro de la función que lo usa.
* Un módulo con anotaciones que mencionen una librería diferida necesita
  `from __future__ import annotations`.
* PEP 562 **no vale** para nombres globales dentro de funciones, solo para atributos del
  objeto módulo. Da `NameError`. Por eso el mecanismo 1 es un objeto y no un hook.

**Imports pesados que quedan a propósito:** `rich` en `EarendilsLight` (4 ms), `pyLOM`
en `readers/pylom.py` (ahí pyLOM *es* el formato, y el registry perezoso ya evita
cargarlo salvo que pidas PYLOM), y `xgboost_preliminar.py` (huérfano, §2).

**B12 arreglado de paso:** `sets/coda.py` importa pyLOM de forma perezosa, así que sin
pyLOM instalado CODA y NUMPY **conservan** su clase Sets y solo fallan los métodos de
exportación. Antes perdían Sets entero en silencio.

**Cambio de comportamiento menor:** iterar un registry lista los formatos *declarados*,
no solo los importables; tras un `.get()` fallido el formato se descuelga de la
iteración. Hay `.available()` si hace falta la verdad completa a costa de importar.

## 9. Lector HORSES3D `.hsol` (10/10/2026)

Encargo: leer las soluciones de HORSES3D para **visualizarlas, validarlas y sacar
conclusiones en equipo**. La exportación a tensor (`create_jset`) queda para después.
Formato verificado contra `HexMesh_SaveSolution`/`SolutionFile.f90` del solver.

**M1 hecho — `FotR/solver_tools/horses3d.py`**, solo numpy (sirve en el `pvpython` de
MN5; un test comprueba que importarlo no carga torch, pandas, h5py…):

* `read_header` (cabecera de 204 B, `complete` = marcador 99 al final), `scan` (offsets
  de cada elemento; N uniforme por tamaño o recorrido si hay p-adaptación),
  `read_solution(path, elements=None)` (`Q` y, según el tipo, `U_x/U_y/U_z`, `sensor`,
  `mu_total/mu_sgs` de LES deducidos del tamaño), `read_mesh` (`.hmesh`), `to_nodes`,
  `primitive_variables` (NS e iNS, adimensional o SI) y `find_solutions`.
* Por elemento devuelve `[k, j, i, var]` sin copiar; `to_nodes` aplana a
  `(n_nodos, n_var)` con el nodo local `i + (Nx+1) j + (Nx+1)(Ny+1) k`. Índices de
  elemento desde 0 (`globID - 1`).
* **Validado:** 24 tests (ficheros sintéticos escritos con `struct` como el solver, más
  la esfera y el fan reales si están); idéntico a `horses2plt` nodo a nodo en la esfera
  (coordenadas y `Q`, error relativo 5·10⁻¹⁰ = redondeo del `.tec`); el fan
  (704 736 elementos, 775 MB) se lee entero en 1.4 s.
* No lee aún los ficheros de superficie (tipos 5–7) ni los de estadísticas (4).

**Siguiente, M2** (toca `sam.py` y `readers/horses3d.py`; mejor tras commitear el
bloque 1): `HorsesLazyField.load()` sobre este núcleo, `Coord_p[str(p)]` en el CADGroup
junto a `Vars[str(p)]` (los campos viven en Σ(N+1)³ nodos de solución, no en los de
`NodeCoords`), opción `geometry='h5'|'hmesh'`, `list_snapshots` y actualizar el test que
exige el `NotImplementedError`. Después: M3 escritores VTU/Tecplot, M4 planos
(`extract_plane.py` de `~/Documentos/utils_H3d/hsol_reference/`) y `Horses3DSets`, M5
derivadas, integrales y `Horses3DStats` (registrados con `LazyRegistry`). Datos de
prueba: la esfera (`~/Documentos/COMAC/my_ESPHERE_1`, pequeña, con gradientes) y
`~/Documentos/COMAC/fan_duct_AD_pressure` (grande).
