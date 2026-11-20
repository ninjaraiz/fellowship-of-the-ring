# PLAN — LEGOLAS: lectura, análisis y visor ligero de mallas CFD

> Acordado con el usuario el 20/11/2026. Estado de cada hito al final. Resumen operativo en
> [`HANDOFF.md`](HANDOFF.md).

## Contexto

LEGOLAS se vació el 18/11/2026 (`FotR/characters/legolas.py`, sin métodos; HANDOFF §2) y se
rehace con un propósito nuevo: **leer y analizar mallas CFD**. En el trabajo de mallas/LES
(COMAC: esfera, turbina NTNU, fan con disco actuador) se pierde tiempo abriendo ParaView para
mirar una malla, y han salido problemas que una herramienta propia habría detectado antes:
BC del `parameter.ini` que no están en la malla, el `.msh` g4 de ANSA con un orden de nodos
propio, nodos duplicados en los CGNS elevados, la búsqueda de estaciones del disco actuador.

Resultado buscado: `LEGOLAS('malla.{h5,cgns,msh}')` da una malla neutra en numpy y una
fachada con funcionalidades en **clases anidadas** (estilo `SAM.Backpack`): un **visor web de
bajo coste** (`LEGOLAS.Viewer`) y, además, censo/validación, calidad, resolución LES y
conversión/alto orden. Decidido con el usuario (20/11/2026): visor web propio (servidor +
three.js, patrón de `coda_monitor`), clases anidadas, y las cuatro familias de funciones.

## Viabilidad (comprobada el 20/11/2026 con ficheros reales, sin tocar el repo)

| formato | cómo | prueba |
|---|---|---|
| HOPR/PyHOPE `*_mesh.h5` | h5py: `NodeCoords`, `GlobalNodeIDs`, `ElemInfo`, `SideInfo`, `BCNames` | turbina g4 (38 718 hex Ngeo=4, 4,8 M nodos): **0,36 s** |
| CGNS | **es HDF5** → h5py basta, sin librería CGNS; `Elements_t` (MIXED=20, HEXA_27…), `ZoneBC` | turbina g4 `.cgns`: 0,10 s; esfera: zona `fluid`, BC `esphere_surface/inlet/outlet/lateral` |
| Gmsh `.msh` (v2.2/v4) | meshio 5.3.5 (ya es dependencia) | esfera v1g2: 2,17 M nodos, hex27+quad9, `PhysicalNames`, 5,6 s |
| `.hmesh` HORSES3D | `FotR/solver_tools/horses3d.py::read_mesh` (ya existe) | — |

Entorno `envkan_nvidia`: numpy, h5py, meshio, scipy, PySide6 (pruebas JS headless) presentes;
pyvista/vtk/trame también, pero **el núcleo no los usará**. Sin pycgns ni gmsh-python.

Riesgos y cómo se acotan:
* **Orden de nodos de alto orden** (HOPR tensorial, CGNS, Gmsh y el propio de ANSA difieren):
  es el riesgo técnico principal. Tablas de permutación por formato + tests de equivalencia
  con la **misma malla en tres formatos** (esfera v1g2 y turbina g1/g2 tienen `.h5`, `.cgns`
  y `.msh`).
* CGNS en ADF (no HDF5): no se lee; error claro (`h5py.is_hdf5`). CGNS estructurado y NGON:
  fuera de la v1.
* Navegador sin internet: el CDN de three.js falla (igual que Plotly en `coda_monitor`);
  opción posterior de empaquetar `three.module.min.js` en el paquete.
* Mallas grandes (TIFON `.msh` 131 MB, mixtas lineales): el visor solo manda superficies y
  cortes, con tope de triángulos; meshio es lento pero aceptable (parser propio solo si
  hace falta).

## Arquitectura

`FotR/characters/legolas.py` pasa a **paquete** `FotR/characters/legolas/` (el mapa perezoso
de `FotR/__init__.py` → `'.characters.legolas'` sigue valiendo). Núcleo **solo numpy + h5py**
(meshio y scipy diferidos con `lazy_module` / import dentro de la función, reglas de HANDOFF §8):

```
legolas/
├── __init__.py     LEGOLAS (fachada) + clases anidadas enganchadas como atributos de clase
├── mesh.py         Mesh, CellBlock, Boundary (dataclasses numpy)
├── ordering.py     orden canónico, tablas CGNS/Gmsh→canónico, detección de orden
├── topology.py     caras, caras exteriores, adyacencia, BC↔cara
├── io/             read(path, fmt=None): hopr.py, cgns.py, gmsh.py, hmesh.py
├── census.py  quality.py  resolution.py  convert.py  locate.py
└── viewer.py + viewer.html
```

**Malla neutra (`Mesh`)**: `points (n,3)` únicos (HOPR deduplica con `GlobalNodeIDs`),
`blocks` por tipo (`hex|prism|pyr|tet|quad|tri`, `order`, `conn` en **orden canónico**),
`boundaries` {nombre → caras + elemento padre + cara local}, `meta` (formato, ruta, Ngeo,
BCType, avisos). Orden canónico: lineales en orden VTK; hex/quad de alto orden en **tensorial
ijk, i más rápido** (el de HOPR/HORSES3D: interpolación y jacobianos triviales). Alcance v1:
cualquier elemento lineal + hex/quad hasta orden 4 (lo que exporta ANSA).

**Fachada**:

```python
from FotR import LEGOLAS
lg = LEGOLAS('MESH/wt_g2_mesh.h5')          # .cgns, .msh, .hmesh; o LEGOLAS(db, case=3)
lg.mesh                                     # malla neutra
lg.census.summary()                         # LEGOLAS.Census
h  = lg.viewer.show()                       # LEGOLAS.Viewer → handle (url, stop)
lg.quality.report(); lg.quality.worst(20)   # LEGOLAS.Quality
lg.resolution.yplus(zone='rotor_cylinder', Re=..., L=...)   # LEGOLAS.Resolution
lg.convert.to_vtu('wt.vtu', mode='lagrange')                # LEGOLAS.Convert
lg.locate.points(xyz)                       # LEGOLAS.Locate
```

## Visor (`LEGOLAS.Viewer`) — copia del patrón de `coda_monitor`

Reutilizar de `FotR/solver_tools/coda_monitor.py`: `launch()` (proceso hijo con
`PYTHONPATH`, manifiesto y log en `~/.cache/FotR/legolas/`, línea de "listo"),
`MonitorServer._bind` (puertos sucesivos), `_Handler` (solo `GET`, 127.0.0.1, solo lectura),
`idle_timeout`, `MonitorHandle` (`url`, `stop`, `_repr_html_`). Puerto por defecto 8766. El
hijo vuelve a leer la malla por ruta (o un `.npz` en caché si viene de memoria).

API (`GET`; geometría en binario: cabecera JSON con longitud + `Float32Array`/`Uint32Array`):
* `/api/info` — censo.
* `/api/surface?zone=<bc>&level=<n>` — caras de la zona trianguladas; las de alto orden se
  subdividen `n×n` (defecto Ngeo) para verse curvas; aparte, las **aristas de elemento**
  (no las diagonales de la triangulación), que es lo que enseña la conectividad.
* `/api/clip?axis=x&value=…&side=+` — *crinkle clip*: elementos enteros a un lado del plano,
  pintando sus caras exteriores → se ve la malla de volumen por dentro.
* `/api/cell?id=` — tipo, nodos, coordenadas, vecinos, métricas (al hacer clic).
* `/api/field?name=` — escalar por cara/elemento para colorear (zona, calidad, Δ, y+).

Página `viewer.html`: three.js (`cdn.jsdelivr.net/npm/three@<versión exacta>`) +
OrbitControls; lista de zonas con casillas y colores, sólido/alambre/aristas, plano de corte
con deslizador, coloreado por métrica con leyenda, selección de elemento por raycasting,
captura. Tope de triángulos (≈3 M) con aviso.

## Otras funcionalidades

**`LEGOLAS.Census` (censo y validación)** — `summary()`: nodos, elementos por tipo y orden,
zonas/BC con nº de caras y área, caja envolvente, nodos huérfanos/duplicados (`cKDTree`).
`check_bcs(control=, ini=)`: BC de la malla frente a la sección de BC de un `.control` de
HORSES3D y los `BoundaryName` de un `parameter.ini` de PyHOPE. `watertight()`, caras
no-manifold, caras internas con BC. Zonas interiores (`BCType` 100 en HOPR, p. ej.
`rotor_cylinder`, `disk_side`) se informan aparte: HORSES3D exige un `type` para cada zona
aunque no tenga caras (HANDOFF §13). `compare(other)`: g1/g2/g4 de la misma malla (mismas
esquinas, mismas BC y caras, pares de nodos coincidentes).

**`LEGOLAS.Quality`** — lineales: jacobiano escalado en esquinas, aspect ratio, skewness
(equiángulo), no-ortogonalidad (normal de cara vs. centroide-centroide), salto de volumen
entre vecinos. Hex de alto orden: jacobiano en puntos de Gauss-Lobatto con derivadas de
Lagrange tensoriales (mirar si sirve `SAM.Backpack._fornberg_weights`, `sam.py` ~l.2570),
mín/máx escalado y negativos. `report()` con histogramas; `worst(n)` → ids que el visor
resalta.

**`LEGOLAS.Resolution` (LES)** — por elemento `h = V^(1/3)` y Δ efectivo de DG `h/(p+1)`;
altura de primera celda en una BC (y, en alto orden, al primer nodo GL interior); ratio de
crecimiento de capa límite siguiendo la adyacencia normal a la pared; Δx⁺/Δz⁺ tangenciales;
**estimación** de y⁺ con Re, L y una correlación de placa plana (Cf = 0,026 Re_x^−1/7). Es
diseño de malla, no posproceso: el y⁺ real con `.hsol` queda para el plan HSOL (M5).

**`LEGOLAS.Convert`** y **`LEGOLAS.Locate` (alto orden)** — `to_vtu(mode='lagrange'|'linear')`
con escritor XML propio en numpy (VTK 72/70); `to_pyvista()` opcional y perezoso;
`detect_order()`: compara nodos de alto orden con la interpolación de esquinas/aristas y
puntúa permutaciones (caso real: g4 `.msh` de ANSA); `write_msh()` con orden corregido como
salida de la detección. `locate.points(xyz)`: prefiltro por cajas + Newton sobre el mapeo
tensorial (como HORSES3D) → elemento y coordenadas de referencia, o −1 y distancia a la
frontera; caso de uso: estaciones de `ActuatorDef/*.dat`.

**Integración con FRODO** — `LEGOLAS(db, case=i)` toma `MESH/*.h5` de un caso HORSES3D o el
`.msh` de un caso CODA_SINGLE. Pintar campos de solución queda fuera (depende de HSOL M3/M4).

## Hitos (cada uno con tests; la suite entera verde antes de cerrar)

| hito | contenido |
|---|---|
| **L0** | este plan en `doc/PLAN_LEGOLAS.md` + HANDOFF (§1 bloque nuevo, §2 decisión «LEGOLAS = mallas», §3 mapa, §5) |
| **L1** | paquete `legolas/`, `Mesh`, `ordering`, lectores HOPR/CGNS/Gmsh/hmesh, `topology` (caras exteriores), `Census.summary` |
| **L2** | Visor v1: superficies por zona, aristas, alambre, clip, selección; handle y servidor |
| **L3** | Census completo: `check_bcs`, estanqueidad, no-manifold, duplicados, `compare` |
| **L4** | Quality + coloreado en el visor + `worst` resaltado |
| **L5** | Resolution LES (h, Δ_DG, primera celda, crecimiento, y⁺ estimado) |
| **L6** | Convert (VTU), `detect_order`/`write_msh`, Locate (estaciones del disco) |
| **L7** | FRODO (`LEGOLAS(db, case)`), README. La documentación va en su visor propio `doc/legolas/`, que se mantiene en cada hito |

## Ficheros que se tocan

* Nuevo: `FotR/characters/legolas/` (sustituye a `legolas.py`), `doc/PLAN_LEGOLAS.md`,
  `FotR/tests/test_legolas_io.py`, `test_legolas_topology.py`, `test_legolas_viewer.py`, …
* Modificados: `doc/HANDOFF.md`, `setup.py` (`package_data` → `viewer.html`),
  `doc/frodo_sam/generate_docs.py` (TARGETS, en L7), `README.md` (en L7).
* Reutilizados: `coda_monitor.launch/_bind/_Handler/MonitorHandle`,
  `solver_tools/horses3d.read_mesh`, `SAM.Backpack.read_horses_mesh_h5` (referencia del
  formato HOPR), `FotR/_lazy.lazy_module`.

Convenciones (memoria del repo): docstrings en inglés estilo numpydoc; narrativa ES/EN solo en
`guides.json`/`i18n.json`; `docs.json` solo regenerado; nada se commitea sin pedirlo; tras
tocar código, `graphify update .`.

## Verificación

* Tests: `/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q`.
  * Sintéticos en `tmp_path`: hex 2×2×2 curvado analíticamente escrito como HOPR (h5py), como
    CGNS/HDF5 (h5py) y como `.msh` (meshio) → misma `Mesh` canónica; tablas de permutación
    verificadas geométricamente.
  * Reales (se saltan si no están): esfera `~/Documentos/COMAC/my_ESPHERE_1/outputs/M_0.3_re_200/MESH/my_esphere_v1g2{.cgns,_mesh.h5,_mesh.msh}`
    y turbina `~/Documentos/COMAC/wind_turbine_ansa/sources/Wind_turbine_disk_fluid_g{1,2}.{cgns,msh}`
    + `*_g{1,2}_mesh.h5` → mismos puntos, mismas BC y nº de caras en los tres formatos; g4
    `.msh` → `detect_order` debe señalarlo.
  * Visor: endpoints por HTTP como `test_coda_monitor.py`; JS con QtWebEngine headless
    (`QT_QPA_PLATFORM=offscreen`, servido por `http://`), errores de consola y captura.
* A mano (usuario): `lg.viewer.show()` en nibbler, abrir el puerto reenviado por VS Code y
  mirar esfera y turbina.

## Estado

| hito | estado |
|---|---|
| L0 | **hecho** (20/11/2026): este documento y HANDOFF |
| — | commits en la rama `legolas` (salida de `develop`): L0–L2 en `2208226`, L3 en el siguiente |
| L1 | **hecho** (20/11/2026): ver abajo |
| L2 | **hecho** (20/11/2026): ver abajo |
| L3 | **hecho** (20/11/2026): ver abajo |
| L4–L7 | pendientes |

### L1 (20/11/2026)

* Paquete `FotR/characters/legolas/` (sustituye a `legolas.py`): `mesh.py`, `ordering.py`,
  `topology.py`, `census.py`, `io/{hopr,cgns,gmsh,hmesh}.py` y la fachada en `__init__.py`.
* **Tablas de permutación medidas**, no transcritas: gmsh 4.13.1 malló un cubo con un elemento
  de orden 1–4 y lo escribió en msh v2.2, msh v4.1 y CGNS (`FotR/tests/data/legolas/`). meshio
  devuelve `hexahedron27` en orden VTK pero `hexahedron64/125` en el de gmsh.
* **HOPR verificado** cruzando caras con vecinos (100 %): nodos tensoriales también en el
  hex lineal 108; caras locales en orden CGNS. Una BC interior (`BCType` 100) va escrita en
  las dos caras: se guarda una vez.
* **Cambio frente al plan:** el `.msh` v2.2 se lee con **parser propio**, porque ANSA lo
  escribe binario en big-endian y meshio no lo lee (falla con «Couldn't read file»). El v4
  sigue por meshio.
* Control de orden tensorial al leer (`ordering.tensor_monotonic`, 2000 elementos): la g4 en
  `.msh` de ANSA falla el 100 % y su área de `inlet` sale 8,86 en vez de 4,88. El `.cgns`
  de la misma malla pasa limpio.
* Comprobado con datos reales: esfera v1g2 y turbina g1/g2 dan las mismas caras por BC y
  áreas en `.h5`, `.cgns` y `.msh`; g4 igual entre `.h5` y `.cgns`. El `.hmesh` de la esfera
  v2g2 reproduce nodos y caras exteriores de su `.h5`.
* Tests: `FotR/tests/test_legolas_io.py` (35). Ejemplo: `examples/legolas/census_turbine.py`.
* **Visor de documentación propio**, `doc/legolas/`, separado de FRODO/SAM y GANDALF (decidido
  por el usuario el 20/11/2026): Inicio · LEGOLAS (fachada + una pestaña por clase anidada; las
  previstas salen deshabilitadas con su hito) · Formatos (una por lector, lista leída de
  `FORMATS`) · Malla neutra · Problemas. Regenerar con `python3 doc/legolas/generate_docs.py`.
* Visto de paso, para L3: los `.cgns`/`.msh` elevados tienen más nodos que el `.h5` (322 297
  frente a 319 317 en la turbina g2): nodos coincidentes que PyHOPE fusiona.

### L2 (20/11/2026)

* **three.js dentro del paquete** (decidido por el usuario): `FotR/characters/legolas/static/three/`,
  versión 0.160.0 sacada del tarball oficial de npm (integridad sha512 comprobada contra el
  registro), solo `three.module.min.js`, `OrbitControls.js` y la licencia MIT; `VERSION` lleva
  procedencia y sha256, que comprueba un test. Fuera de graphify (`.graphifyignore`) y marcado
  como `linguist-vendored -diff` en `.gitattributes`. `setup.py` lo incluye en `package_data`.
* `scene.py` (solo numpy): superficies por zona y pseudo-zonas (`__exterior__`,
  `__unassigned__`), caras de alto orden dibujadas por su rejilla de nodos (N×N cuadriláteros,
  sin interpolar), aristas de elemento, corte *capa* (crinkle slice) y *recorte* (celdas por
  centroide, solo las caras que abre el corte), `cell_info` (vecinas, BC por cara, contorno)
  y `pack` (cabecera JSON + arrays alineados a 4 bytes). Posiciones relativas al centro de la
  caja, en float32.
* `viewer.py`: `ViewerServer` (stdlib, 127.0.0.1, solo lectura, lista blanca de estáticos),
  `launch`/`ViewerHandle` (proceso aparte, como `coda_monitor`), CLI
  `python -m FotR.characters.legolas.viewer malla`, y la clase anidada `LEGOLAS.Viewer`
  (`show`, `server`, `stop`). El proceso aparte **vuelve a leer el fichero**; una malla en
  memoria viaja por `Mesh.save`/`Mesh.load` (npz, nuevo) y se borra al leerla.
* **Cambio frente al plan:** el `idle_timeout` de `coda_monitor` se basa en que su página pide
  datos sin parar; un visor estático no lo hace, así que la página manda `/api/ping` cada
  minuto.
* **Tope de 1 M triángulos por petición** (unos 55 MB): con 3 M, la zona `entity_1` de la
  esfera pesaba 131 MB. Las zonas que lo superan se diezman y arrancan ocultas.
* Página (`viewer.html`): zonas con color y ⌖ para encuadrar, superficie/aristas, opacidad,
  capa y recorte (en recorte las fronteras se cortan con el mismo plano en la GPU), selección
  de celda con vecinas navegables y nº HORSES3D, vistas ±X/±Y/±Z, captura PNG, triedro, ES/EN
  y modo oscuro. `window.LEGOLAS_API` para pruebas automáticas.
* **Probado en navegador:** QtWebEngine con `offscreen` no tiene WebGL; con un Xvnc privado de
  TurboVNC (`~/local/turbovnc/.../Xvnc :77 -nolisten tcp -rfbunixpath …`), `QT_QPA_PLATFORM=xcb`
  y `--use-gl=angle --use-angle=swiftshader --ignore-gpu-blocklist` sí (llvmpipe). Turbina g2 y
  esfera v1g2: sin errores de consola, capturas revisadas.
* Tests: `FotR/tests/test_legolas_viewer.py` (23). Ejemplo: `examples/legolas/viewer_turbine.py`.
  Visor de documentación: pestaña `Viewer` activa, con `ViewerHandle`, `ViewerServer` y
  `legolas.scene`.

### L3 (20/11/2026)

* `bcfiles.py`: lectura de los `#define boundary` del `.control` (varias zonas por bloque con
  `__`, comentarios `!`) y de los `BoundaryName`/`BoundaryType` del `parameter.ini`. No
  comparte código con el ring `horses3d` a propósito (misma política que `PLAN_HORSES3D.md`).
* `Census` gana `zone_status`, `check_bcs`, `watertight`, `duplicates`, `compare`, `check`,
  `check_report` y `bcs_report`. Todas devuelven sus números y `issues` con nivel
  `error`/`warning`/`info`.
* Reglas de `check_bcs` sacadas de lo vivido: HORSES3D pide `type` para toda zona **con
  caras**, también las interiores (las vacías, como `fluid` de la esfera, no: la esfera corrió
  sin declararla); PyHOPE pide tipo 100 para las interiores («internal face with multiple
  BCs»).
* `duplicates` separa nodos de alto orden repetidos (inofensivos), grietas (caras exteriores
  superpuestas) y celdas colapsadas; `topology.coincident_labels` (k-d tree + componentes
  conexas) sale del lector `.hmesh`, que ahora lo usa.
* Resultados reales: el `.control` y el `parameter.ini` de la turbina son de la malla
  **empty** (limpios ahí); contra la **fluid** sale el error de `disk_side` (tipo 5, no 100).
  El `.cgns` g2 tiene 2980 nodos de alto orden repetidos y ninguna grieta. g1/g2/g4 comparten
  esquinas (1,6·10⁻¹⁴); el área de `rotor_cylinder` cambia un 0,03 % de g2 a g4. Esfera:
  `watertight` 1,1 s y `duplicates` 2,0 s con 2,2 M nodos.
* Tests: `FotR/tests/test_legolas_census.py` (19); `FotR/tests/_legolas_meshes.py` con la caja
  de prueba común a censo y visor. Ejemplo: `examples/legolas/check_turbine.py`.

### Decisiones abiertas (se resuelven al llegar al hito)

* L6: ¿`write_msh` corrige el orden en el `.msh` o se deja en diagnóstico + VTU?
