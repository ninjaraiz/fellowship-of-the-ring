# HANDOFF — Fellowship of the Ring

> Lee esto primero. Para el detalle (arquitectura, contratos de datos, bugs
> localizados, cobertura de tests) → [`doc/frodo_sam/ANALISIS.md`](frodo_sam/ANALISIS.md).
> Para usar una clase, los visores de `doc/` (§7). No releas el repo entero.

## 1. Estado — `develop` @ `6126066` (10/11/2026), **con trabajo sin commitear**

Suite: **542 passed** (24/11/2026; 170 son de LEGOLAS, `test_legolas_*.py`; 11 de `horses2plt`).

```bash
/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q
```

El reinicio de casos (§10) y el monitor de CODA (§11) están en `6126066`. Sin commitear:

1. **graphify** (§12): `CLAUDE.md`, `.claude/`, `.gitattributes`, `.graphifyignore`,
   `graphify-out/` y `.gitignore`.
2. **LEGOLAS vaciado** (18/11/2026): `legolas.py` y una referencia en `stats/coda.py`.
3. **Pipeline HORSES3D** (18/11/2026, §13 y [`PLAN_HORSES3D.md`](PLAN_HORSES3D.md)): ring
   `rings/horses3d.py` (nuevo), registro y docstrings de GANDALF, reader
   `readers/horses3d.py`, tests `test_rings_horses3d.py` (nuevo) y `test_horses3d.py`,
   ejemplo `examples/gandalf/horses3d_wind_turbine.py`, visores regenerados.
4. Un notebook tuyo (`merge_format_CODA.ipynb`).
5. **Rama `legolas`** (salida de `develop`; el árbol de trabajo está en ella): LEGOLAS L0–L6
   en `2208226`, `0fe65c6`, `dd9e591`, `d9e61c4`, `ceaeaa4`, L8 (soluciones) en `3df459d` y el
   deslizador en `00a1748` ([`PLAN_LEGOLAS.md`](PLAN_LEGOLAS.md)),
   y **aparte** (24/11/2026, a petición del usuario) el lector HORSES3D con estados y
   `Horses3DSets`/`horses2plt` en `07f1760`. Del bloque 3 sigue sin commitear el ring de
   GANDALF (`rings/horses3d.py`, `gandalf.py`, `rings/__init__.py`, sus tests, ejemplos y
   visor) y `PLAN_HORSES3D.md`. Sin commitear a propósito por mezclarse con otros bloques:
   esta HANDOFF, `doc/gandalf/guides.json` (un párrafo de LEGOLAS), `.graphifyignore` y
   `.gitattributes`.

Aviso sobre los tests: los ficheros usan **stubs mutuamente incompatibles** y el
aislamiento depende del orden de importación. El motivo original (no arrastrar
pyvista/torch/pyLOM) ya no existe desde §8, pero los stubs siguen ahí y siguen
siendo frágiles.

## 2. Decisiones vigentes que no están en el código

* `FlCc` = solo `design_vars` del usuario, nunca params del solver
  (`readers/coda.py`, `readers/horses3d.py`).
* HORSES3D: `p` único exigido en `extract_outputs`, `.hsol` header+lazy
  (`HorsesLazyField`, cuyo `load()` lanza `NotImplementedError` hasta M2, §9).
  `MESH/*.h5` es la geometría por defecto; se permite leer el `.hmesh` y los autosaves
  `*_<iter>.hsol`. bmesh/pmesh/cgns, nunca.
* HORSES3D, desde el 18/11/2026 (§13): la malla lleva `g<n>` en el nombre (= `Ngeo`); la
  solución de orden p es `RESULTS/<tag>_g<g>p<p>.hsol`; un caso sin ejecutar es
  `pending`, no un error; los controles se escriben **por claves** y sin comentarios.
* `GANDALF.Backpack` NO se delega: solo existe `BaseRing.Backpack`
  (`gandalf.py`, `_NON_DELEGATED`). Y `BaseRing` no se re-exporta en
  `FotR/__init__.py`, así que la ruta buena es
  `from FotR.characters.rings import BaseRing`.
* En CODA_SINGLE, `mesh` puede ser design var (factor numérico); la identidad de malla
  vive en `sim_metadata[*]['mesh_info']` y `df_state['mesh_file']`, nunca en la clave
  `mesh`, que colisionaría.
* `data_dict['CADGroup_X']` en CODA_SINGLE guarda geometría y `Vars` como **listas por
  caso** (los `npts` difieren).
* **LEGOLAS vaciado el 18/11/2026** y, desde el 20/11/2026, **dedicado a mallas**:
  lectura (HOPR `.h5`, CGNS/HDF5, Gmsh `.msh`, `.hmesh`) a una malla neutra en numpy, visor
  web propio (servidor + three.js, como `coda_monitor`) y censo, calidad, resolución LES y
  conversión, cada uno en una **clase anidada** (`LEGOLAS.Viewer`, `LEGOLAS.Census`…).
  L6 (24/11): `lg.convert` (`fix_order` arregla el `.msh` g4 de ANSA para PyHOPE, `to_vtu`,
  `to_vtm`, `write_msh`) y `lg.locate` (`points`, `horses3d_ad`, réplica de la búsqueda del
  disco actuador). L8 (24/11): `LEGOLAS.from_control(control)` y `lg.solution` (un `.hsol`
  evaluado en los nodos de la malla, validado contra el `.hmesh` del solver a 1e-14); el visor
  pinta la solución por nodo y tiene deslizador de autoguardados.
  Plan e hitos L0–L7 en [`PLAN_LEGOLAS.md`](PLAN_LEGOLAS.md). L1 hecho: lectores y
  `LEGOLAS.Census`. L2 hecho: `LEGOLAS.Viewer` (`lg.viewer.show()`), three.js **dentro del
  paquete**, sin internet. L3 hecho: `census.check_bcs(control=, ini=)`, `watertight`,
  `duplicates`, `compare`, `check`. L4 hecho: `lg.quality` (jacobianos, oblicuidad,
  no ortogonalidad…) y coloreado por métrica en el visor. L5 hecho: `lg.resolution` (Δ de DG,
  primera celda y primer nodo en pared, y⁺ estimado) y `lg.viewer.show(fields=…)` para
  cualquier campo por celda. Orden canónico: hex tensorial (i más rápido) **también el lineal**; el
  `.msh` v2.2 con parser propio (ANSA lo escribe big-endian y meshio no lo lee).
* Sin tocar a propósito: columnas 2-D
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
| mallas | `characters/legolas/` ([`PLAN_LEGOLAS.md`](PLAN_LEGOLAS.md)) | `LEGOLAS`: lectores HOPR/CGNS/Gmsh/hmesh → malla neutra numpy, `Census`, `Viewer` (web, three.js empaquetado) |
| imports perezosos | `FotR/_lazy.py` | `lazy_module`, `LazyRegistry` (§8) |
| monitor en directo | `solver_tools/coda_log.py`, `coda_monitor.py` | logs de CODA y servidor web, solo biblioteca estándar (§11) |
| documentación | `doc/frodo_sam/` · `doc/gandalf/` · `doc/legolas/` | tres visores HTML (§7) |

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
  Lo que **no** se tocó: la misma conectividad errónea sigue en la ruta de CODA.
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
* **Turbina** (§13): falta la plantilla `wind_turbine.control` y las condiciones del
  caso (modelo de rotor, IBM, condiciones iniciales, M, P, t final), que decide el
  usuario. Después: FSP, prueba corta y producción.
* **HORSES3D M2** (§9) sigue pendiente: `HorsesLazyField.load()` aún lanza
  `NotImplementedError`. Después, monitor en directo para HORSES3D y M3–M5.
* **LEGOLAS** ([`PLAN_LEGOLAS.md`](PLAN_LEGOLAS.md)): L0–L6, L8 y el deslizador commiteados en
  `legolas`; falta que pruebes el visor en tu navegador. Pendiente: estadísticas
  (`.stats.hsol`), dibujar al orden de la solución cuando p > Ngeo, gradientes, superficies, y L7.
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

Tres visores HTML, misma maquinaria, cada uno autocontenido. **Sírvelos desde `doc/`**,
no desde su subcarpeta: se enlazan entre sí con rutas `../` y si no dan 404.

| visor | qué cubre |
|---|---|
| `doc/frodo_sam/` | Inicio (mapa + quién es quién + matriz de formatos) · FRODO (una pestaña por formato, dentro una por pieza) · SAM (una por área) · `data_dict` · Flujos · buscador global |
| `doc/gandalf/` | GANDALF (la fachada) · Rings (una pestaña por ring, dentro una por fase de la campaña) · Backpack · Problemas |
| `doc/legolas/` | Inicio (mapa, estado de hitos, cruces) · LEGOLAS (fachada + una pestaña por clase anidada; las previstas, deshabilitadas) · Formatos (una por lector) · Malla neutra (contrato y orden canónico) · Problemas |

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
* **WebGL** (el visor de LEGOLAS) no funciona con `offscreen`: hace falta una pantalla
  virtual. Receta en [`PLAN_LEGOLAS.md`](PLAN_LEGOLAS.md), L2 (Xvnc de TurboVNC en `:77`,
  `QT_QPA_PLATFORM=xcb`, ANGLE/SwiftShader).

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

## 12. Grafo de graphify (10/11/2026)

Mapa del código para no releer ficheros: `graphify-out/graph.json` (2035 nodos, 3944
aristas, 99 comunidades) y `graphify-out/GRAPH_REPORT.md`. Solo código: `FotR/`
(tests incluidos) y los `.py` de `examples/`. Fuera: notebooks (graphify no lee
`.ipynb`), `doc/` salvo los `generate_docs.py`, HTML y logs (`.graphifyignore`).

* **Herramienta:** graphify 0.9.105, clonado en `~/repos/graphify`, instalado en el venv
  `~/.local/share/graphify-venv` (Python de miniconda) y enlazado en `~/.local/bin`.
  Skill global en `~/.claude/skills/graphify/` (registrada en `~/.claude/CLAUDE.md`).
* **Construir sin gastar tokens:** `GRAPHIFY_BACKEND=none graphify extract . --code-only`
  y `GRAPHIFY_BACKEND=none graphify cluster-only .` (unos 5 s). Sin `GRAPHIFY_BACKEND=none`
  puede usar `claude -p` para poner nombre a las comunidades. Tras cambios: `graphify
  update .`.
* **Consultas útiles:** `graphify explain "<símbolo>"` (vecinos exactos con fichero y
  línea, el más preciso), `graphify path A B`, `graphify query "<pregunta>"` (BFS,
  más ruidoso: ~4000 tokens en la prueba).
* **Se commitea** `graphify-out/` para trabajar desde otra máquina, salvo `cache/`,
  `graph.html` y `.graphify_python` (en `.gitignore`: locales o regenerables).
* **Integración:** `graphify claude install` (sección en `CLAUDE.md` y hooks
  `PreToolUse` en `.claude/settings.json`, modo normal) y `graphify hook install`
  (reconstrucción en cada commit y checkout). Tras un `git pull`, `graphify update .`.

## 13. Pipeline HORSES3D: ring de GANDALF + reader (18/11/2026, sin commitear)

Plan y decisiones en [`PLAN_HORSES3D.md`](PLAN_HORSES3D.md).

* **Tecplot (24/11/2026):** `db.to_tecplot(binary, script, p, kinds, …)` (`Horses3DSets`,
  núcleo en `solver_tools/horses2plt.py`, ejecutor `utils_H3d/POST/horses2plt_batch.slurm`)
  convierte finales, estadísticas y autoguardados a `outputs/<caso>/POST/tec/p<p>/`. Validado
  con Mariño 2024 A p2. Detalles y trampas del binario en el diario del plan y en la guía de
  HORSES3D §8.1.

* **Ring `horses3d`** (`GANDALF(root, ring='horses3d')`, sin `eq_type` ni `num_stages`):
  `define_cases`/`compute_param` heredados; `generate_folders(control_template,
  mesh_path, tag, folder_fmt, p_levels, control_values, p_values, extra_dirs,
  data_to_update, setup_dir, run_script, fsp, fsp_steps)` escribe por caso `MESH/`
  (copia), `RESULTS/`, `LOG/`, `run.sh`, un `file_control_p<p>.control` por p (escalera
  de restart) y `fsp_p<p>.control`; `submit_cases(p, repo, arch, node, …, fsp, dry_run=True)` (`repo` obligatorio: `HORSES3D_REPO` de `run.sh`, 20/11; `p` puede ser una lista ascendente de órdenes encadenados con `afterok`, 22/11)
  llama a `./run.sh`; `fsp_report(p)` compara la solución de la prueba con la inicial; `load_campaign()`
  recupera la campaña de `cases_metadata.json` (lo hacen solos los dos anteriores).
* **FSP (Free-Stream Preservation):** `Inflow` en todas las fronteras de la malla (las
  `BCType` 100, interiores, fuera), sin actuador, IBM, monitores, superficies ni
  restart, `fsp_steps` pasos, salida en `RESULTS/FSP/`. Comprueba la malla, no la física.
  FRODO no lee esos controles.
* **Reader:** `g` del token `g<n>` del nombre (vale `_cgns_g2_` y `_v2g2_`) o del
  atributo `Ngeo`; nombre y `Ngeo` distintos → error. Cada solución tiene `status`
  (`pending`/`started`/`finished`, este por el terminador 99 del `.hsol`), `snapshots`
  y `actuator_files`; `df_state` añade `p_finished` y `state`. Ya no son error: `.hsol`
  ausente, restart desde otro p del mismo caso aún sin correr, `restart polorder >= p`
  (aviso). `extract_outputs` exige soluciones terminadas.
* **Comprobado con datos reales:** `my_ESPHERE_1` y `fan_duct_AD_pressure` se leen en modo
  estricto (re_400 pendiente; el fan pasa a `g = 2`). El ring, probado en el scratchpad
  con la malla g2 de la turbina y el control del tutorial como plantilla.
* **19/11/2026:** HORSES3D exige `restart file name` aunque `restart = .false.` (el primer
  intento de FSP murió por eso); el ring la escribe siempre y comprueba las claves
  obligatorias antes de escribir nada. `overwrite=True` ya no borra la carpeta: nunca toca
  `RESULTS/` ni `LOG/`. HORSES3D exige también un `type` para **cada zona de la malla**,
  interiores incluidas, aunque estas no tienen caras (son interiores) y su tipo no se
  aplica nunca: si la plantilla no las declara, el ring añade `type = FreeSlipWall` y lo
  imprime; la FSP las declara todas; falta una frontera real → error. `fsp_report` juzga
  por la desviación de la solución (los residuos a Mach bajo no bajan del redondeo). FSP
  de la turbina superada (job 4271). Diario del caso de la turbina en
  [`PLAN_HORSES3D.md`](PLAN_HORSES3D.md).
* **Qué se puede cambiar del `.control`** desde el ring: guía del visor de GANDALF (pestaña
  Rings) y ejemplo `examples/gandalf/horses3d_sphere_re_mach.py` (Mach y Re por caso, p2 → p3).
* **No hay módulo común ring/reader**: como en CODA, cada uno por su lado y
  `cases_metadata.json` de puente; la regla de `g` está duplicada y la cubre el test de
  ida y vuelta (`test_rings_horses3d.py`).
