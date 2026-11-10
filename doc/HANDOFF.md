# HANDOFF — Fellowship of the Ring

> Lee esto primero. Para el detalle (arquitectura, contratos de datos, bugs
> localizados, cobertura de tests) → [`doc/frodo_sam/ANALISIS.md`](frodo_sam/ANALISIS.md).
> Para usar una clase, los visores de `doc/` (§7). No releas el repo entero.

## 1. Estado — `develop` @ `d6dd431` (17/10/2026), **con trabajo sin commitear**

Suite: **315 passed** (27/10/2026; 34 son del monitor, `test_coda_monitor.py`).

```bash
/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q
```

Los cuatro bloques que estaban sin commitear se commitearon el 17/10: lector `.hsol`
(`6316468`), imports perezosos (`18c7002`), documentación (`479efc2`), examples
(`477d054`), y después la reescritura de EarendilsLight (`e2c6dfe`, `f8a58b0`,
`4372d1a`, `d6dd431`).

Sin commitear hoy:

1. **`CODAResiduals.restart_stalled_cases`** (21/10/2026, §10) —
   `FotR/characters/residuals/coda.py` (método, `_prepare_restart`, `_apply_restart`
   y helpers de módulo `_current_slurm_cluster`, `_slurm_active_workdirs`,
   `_slurm_node_cpus`, `_sbatch`…), `FotR/tests/test_coda_restart.py` (19 tests, sin
   trackear) y `doc/frodo_sam/docs.json` regenerado.
2. **Monitor en directo de casos CODA** (27/10/2026, §11) — nuevos y sin trackear:
   `FotR/solver_tools/coda_log.py`, `coda_monitor.py`, `coda_monitor.html` y
   `FotR/tests/test_coda_monitor.py`, más `examples/gandalf/monitor_TIFON.py`. Toca
   `CODAResiduals.monitor_cases` (en `residuals/coda.py`, que ya tenía el bloque 1),
   `solver_tools/__init__.py` (docstring), `setup.py` (`package_data` del `.html`),
   `README.md`, `examples.json` y `docs.json` regenerado.
3. **Examples** — `examples/gandalf/restart_TIFON.py` (nuevo, sin trackear; usado el
   27/10 para relanzar casos de rans5), `df_post_analysis.ipynb`, `TIFON2.ipynb` y
   `codasingle.ipynb` (este último pierde ~23 000 líneas: salidas borradas). Son tuyos.

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

* `restart_stalled_cases` **reproduce el reinicio manual y no corrige** los dos bugs de
  reinicio de `run_sst_v5.py` (el stage reiniciado arranca desde corriente libre; §10).
  Decidido el 21/10/2026 para que los casos de rans5 sean coherentes entre sí.

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
| monitor en directo | `solver_tools/coda_log.py`, `coda_monitor.py` | logs de CODA y servidor web, solo biblioteca estándar (§11) |
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
* `EarendilsLight` **reescrito el 17/10/2026** y ya con tests (20). Antes lanzaba
  `TypeError: expected str instance, Syntax found` en toda consulta que encontrase un
  docstring. Ahora hace dos cosas y solo dos: resolver la cadena de delegación sobre un
  objeto vivo (`db.some_light('extract_inputs')` → qué reader te tocó y su firma real) y
  buscar un nombre en el paquete con `ast`, sin importar nada. Render en texto plano:
  `rich` fuera, y con él ~49 ms de cada arranque de FRODO.
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
* Los dos [ALTO] de ANÁLISIS §4: el de `find_files(file_end=...)` **ya no está** (no
  queda ningún `file_end=` en `readers/coda.py`); el de `CODAReader.extract_inputs`
  **sigue**: un caso que falla avisa «This case was skipped», pero entra en el dataset
  con `FlCc` e `idx_sort` a cero (`readers/coda.py`, `except` del bucle por caso).
* **Slurm y clústeres.** Los casos de TIFON corren en **flexo**, cuyo `ClusterName` es
  **`linux`** (así lo usa `restart_TIFON.py`). El defecto `cluster='flexo'` de
  `restart_stalled_cases` no casa con ese nombre: con `dry_run=False` lanza
  `RuntimeError` salvo que se pase `cluster='linux'`. Las sesiones de Claude suelen
  correr en **nibbler** (`ClusterName = nibbler`), donde `squeue` contesta con otra
  cola, y desde nibbler **no hay ssh a flexo** (comprobado el 27/10/2026). El NFS sí
  es compartido: los ficheros de los casos se ven desde los dos.

## 5. Siguiente paso previsto

* **Monitor en directo (§11): hecho, sin commitear y sin probar en tu navegador.**
  Probado con un navegador headless (QtWebEngine), no con VS Code + Brave reales.
  Falta que lo abras tú y decidas si `plot_state_calculation` sigue teniendo
  sentido.
* **HORSES3D M2** (§9) sigue pendiente: `HorsesLazyField.load()` aún lanza
  `NotImplementedError`.
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
* No hay node ni deno, pero **sí se puede ejecutar JS**: QtWebEngine de PySide6 en
  modo headless carga la página, devuelve los errores de consola y hace capturas (ver
  la nota de §11). La comprobación en tu navegador sigue siendo tuya.

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

**M1 hecho y commiteado (`6316468`) — `FotR/solver_tools/horses3d.py`**, solo numpy (sirve en el `pvpython` de
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

## 10. Reinicio de casos estancados (21/10/2026, sin commitear)

`db.residuals.restart_stalled_cases(cases, cpus, node, stage=None, cluster=…,
dry_run=True, submit=False)` automatiza el reinicio manual de un stage que llegó al
límite de iteraciones sin converger:

1. renombra `output_<stage>.h5` a `ci.h5` y borra el resto de `output_<stage>*`;
2. en el script pone `custom_ci = True` y `stages = ['stage<stage>']`;
3. en `run.sh` cambia el nodo y las CPUs;
4. con `submit=True`, hace el `sbatch`.

Valida cada caso antes de escribir nada y devuelve un DataFrame con lo que hizo (o
haría) por caso. Con `dry_run=False` exige estar en el clúster de los casos (ver §4:
en flexo es `cluster='linux'`). Ejemplo real: `examples/gandalf/restart_TIFON.py`.
El docstring explica por qué el stage reiniciado arranca desde corriente libre con
`run_sst_v5.py`.

## 11. Monitor en directo de casos CODA (27/10/2026, sin commitear)

```python
mon = db.residuals.monitor_cases([219, 228, 243])     # case_idx o carpetas
mon = db.residuals.monitor_cases(query='mach > 0.78') # o subset=, o un DataFrame, o 'all'
mon.url; mon.stop()
```

Página web pequeña que sigue una lista de casos mientras corren: un caso cada vez
(flechas para cambiar, `t` para la tabla de todos), refresco cada 5 s. Cabecera con
estado, stage, iteración frente al límite, s/iteración, tiempo hasta el límite, job,
nodo y si arrancó de `ci.h5`; un chip por criterio de parada frente a su umbral; y una
gráfica con los residuos relativos (con su umbral), un panel por coeficiente con el
tubo de Cauchy y la CFL.

**Piezas** (las dos primeras, solo biblioteca estándar; hay test de que no cargan
numpy ni pandas):

| pieza | qué hace |
|---|---|
| `solver_tools/coda_log.py` | `CodaLog`: lector incremental del `-out.txt` (solo lee lo nuevo, hasta la última línea completa). `CodaCase`: una carpeta de caso; sigue el log más nuevo (por job id), lee `run.sh` y `-err.txt`, calcula estado, ritmo y criterios |
| `solver_tools/coda_monitor.py` | `MonitorServer` (`http.server` en 127.0.0.1, hilo que relee cada `interval`, se para solo tras `idle_timeout` sin peticiones), `launch()` (proceso aparte, manifiesto y log en `~/.cache/FotR/coda_monitor/`) y CLI `python -m FotR.solver_tools.coda_monitor <carpetas>` |
| `solver_tools/coda_monitor.html` | la página; Plotly básico desde cdnjs, modo oscuro, pensada para ~500 px de ancho |
| `CODAResiduals.monitor_cases` | elige los casos desde FRODO (`case_idx`, carpeta, `subset`, `query`, DataFrame, `'all'`) y llama a `launch` |

**Hechos medidos** sobre rans5 corriendo en flexo y leído desde nibbler por el NFS:

* Durante la ejecución **solo crece el `-out.txt`**; los `output_<s>__monitors_*.dat`
  se exportan al acabar cada stage. Va línea a línea, sin buffer, a ~5 s por iteración
  con 40 CPUs. El log completo (3,5 MB) se lee en ~0,2 s; una actualización, en <1 ms.
* Trae por iteración la CFL, las tres reducciones de residuo y CD, CL y CMy (con 10
  decimales), y por stage los criterios de parada completos. CODA escribe además
  `Iteration N: Cauchy convergence achieved over 1000 samples.` en cada iteración
  en que se cumple Cauchy, y el `StopResidualStagnation` de `run_sst_v5.py` escribe
  `residual stagnation achieved ...`. El monitor muestra los dos.
* **Criterio de Cauchy, deducido de los logs** (el código de CODA no se puede leer, ver
  la nota de abajo): por variable, `max |x_i − x_actual|` sobre el valor actual y las
  N=1000 muestras anteriores del stage, menor que su umbral. Reproduce el mensaje de
  CODA en **las 2,1 millones de iteraciones de los 329 logs de rans5**, sin una sola
  discrepancia; `max − min` sobre la ventana falla en ~2 %. Contrastado también
  contra los `.dat`: iteraciones, reducciones (= absoluto / residuo inicial) y
  coeficientes coinciden (hay test con el caso real `aoa_0.0222_mach_1.0715`).
* **Estados:** `pending` (sin log), `queued` (el último log acabó y su
  `output_<s>.h5` ya es `ci.h5`: reinicio preparado, job aún sin log), `starting`,
  `running`, `quiet` (>15 min sin escribir), `finished`, `limit`, `diverged` (NaN),
  `cancelled` (`CANCELLED` en `-err.txt`) y `error` (acabó con `-err.txt` no vacío).
  Desde nibbler no hay Slurm de flexo, así que todo se deduce de los ficheros.
* El ritmo es «vivo» (iteraciones vistas llegar en los últimos 10 min) o, al abrir,
  la media desde la primera línea `date`, que incluye el arranque.

**Notas:**

* **No se lee nunca el código fuente de CODA** (`~/flowsimulator2024`): orden
  explícita del usuario. Lo de CODA se deduce de sus salidas y de los scripts de caso.
* El servidor escucha en 127.0.0.1 de nibbler: cualquier usuario de nibbler puede
  llegar a ese puerto. Solo sirve la página y los logs de los casos que se le dieron,
  que de todos modos son legibles por todos (`-rw-r--r--`).
* **Sí hay forma de ejecutar JS aquí**: QtWebEngine de PySide6 (en `envkan_nvidia`) con
  `QT_QPA_PLATFORM=offscreen` y `QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu
  --no-sandbox'`. Carga una URL `http://`, devuelve los mensajes de consola
  (`javaScriptConsoleMessage`), ejecuta JS (`runJavaScript`) y hace capturas
  (`view.grab()`). Con `file://` no carga los scripts del CDN. Vale también para los
  visores de `doc/`.
