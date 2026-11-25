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
| — | commits en la rama `legolas` (salida de `develop`): L0–L2 `2208226`, L3 `0fe65c6`, L4 `dd9e591`, L5 `d9e61c4`, L6 `ceaeaa4`, L8 el siguiente; aparte, el lector HORSES3D y `Horses3DSets` en `07f1760` |
| L1 | **hecho** (20/11/2026): ver abajo |
| L2 | **hecho** (20/11/2026): ver abajo |
| L3 | **hecho** (20/11/2026): ver abajo |
| L4 | **hecho** (20/11/2026): ver abajo |
| L5 | **hecho** (24/11/2026): ver abajo |
| L6 | **hecho** (24/11/2026): ver abajo |
| L7 | pendiente |
| L8 | **hecho** (24/11/2026): soluciones de HORSES3D en el visor, ver abajo |

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

### L4 (20/11/2026)

* `quality.py` (solo numpy) y la clase anidada `LEGOLAS.Quality` (`lg.quality`): `metrics`,
  `field`, `summary`, `worst`, `issues`, `report`. Métricas por celda: `scaled_jacobian`,
  `jacobian_ratio` (mín/máx det J, la «scaled Jacobian» de PyHOPE), `aspect_ratio`,
  `skewness`, `non_orthogonality`, `volume_ratio`, `volume`.
* Hexaedros de orden N: jacobiano en los (N+1)³ puntos de Gauss-Lobatto con derivadas de
  Lagrange sobre nodos geométricos **equiespaciados**, por trozos para acotar memoria. No se usa
  `SAM.Backpack._fornberg_weights`: arrastraría SAM (y torch) a un módulo que debe ser ligero.
  Lineales: aristas de cada esquina, normalizadas a 1 en la celda regular.
* **Orientación por bloque:** VTK y gmsh numeran el prisma al revés; cada bloque toma la
  orientación de su mayoría (`meta['flipped']`). Error encontrado al probar con TIFON: tomar el
  mínimo por esquinas *antes* de girar el signo guardaba la mejor esquina (salían prismas con
  1,155); ahora se gira antes.
* Comprobado: celdas ideales dan 1; turbina g1/g2/g4 `jacobian_ratio` mínimo 0,46 sin negativos
  (lo que dio PyHOPE). La esfera `my_ESPHERE_1` da ≈ 0,074 en todas: son tetraedros partidos en
  hexaedros (0,0741 exacto en el regular), así que el aviso de `jacobian_ratio` va en 0,05.
  TIFON `mesh_f22_v5.msh` (724 k prismas): 7 s; esfera (260 k hex p2): 5 s.
* Visor: `/api/field` (float32 por celda, cabecera con p01/p99) y `/api/worst`; selector
  **Color** con leyenda azul→rojo (mejor→peor), rango p1–p99 por defecto, las diez peores
  clicables (seleccionan, llevan el plano de corte a la celda y acercan la cámara mirando a la
  abertura) y métricas en la ficha de la celda.
* Tests: `FotR/tests/test_legolas_quality.py` (24). Ejemplo: `examples/legolas/quality_turbine.py`.

### L5 (24/11/2026)

* `resolution.py` y la clase anidada `LEGOLAS.Resolution` (`lg.resolution`): `sizes(p)`
  (`h` = V^(1/3), `h_min`/`h_max` direccionales siguiendo los nodos curvos, anisotropía,
  Δ = h/(p+1)), `wall(zonas, p, layers)` (espesor de la celda de pared sobre la normal, primer
  nodo de solución = primer punto GL interior de orden p, tamaños tangenciales, crecimiento
  por caras opuestas), `yplus(zonas, Re, L, p, correlation, flow_dir)` (Cf de placa plana
  turbulenta 0,026·Re_x^(−1/7) o Blasius; Re_x local con `flow_dir`), `fields()` y `report()`
  con los objetivos WRLES (y⁺ ≤ 1, Δx⁺ 50–150, Δz⁺ 15–40).
* **El visor acepta campos por celda de cualquier origen:** `lg.viewer.show(fields={...})`
  (también `server(fields=)`); viajan al proceso del servidor en un npz que este borra al leerlo.
  Es el canal que usarán los campos de un `.hsol` (plan HSOL, M3/M4).
* `quality.worst_cells` sale de `Quality.worst` para servir también a esos campos.
* Error encontrado por un test: el borde de ataque se tomaba del centroide de caras y no de la
  esquina más aguas arriba de la zona.
* Esfera v2g2 (D = 1, Re = 400, laminar, p = 3): primer nodo de 0,016 a 0,052 D, y⁺ estimado
  0,86–5,3 (mediana 1,9). TIFON: primera capa 5·10⁻⁵ c exacta; el crecimiento sale NaN porque
  es 2D extruida y el prisma apoya en la pared por un cuadrilátero (limitación documentada).
* **WebGL en n008** (la sesión de hoy corría allí, no en nibbler): SwiftShader revienta
  (`EGL_BAD_ATTRIBUTE`); funciona `--use-gl=desktop` sobre el Xvnc `:77` (GLX por llvmpipe).
* Tests: `FotR/tests/test_legolas_resolution.py` (19). Ejemplo:
  `examples/legolas/resolution_sphere.py`.

### L6 (24/11/2026)

* `convert.py` y `LEGOLAS.Convert` (`lg.convert`): `detect_order`/`fix_order`, `to_vtu`
  (Lagrange 72 o lineal, campos por celda y por nodo), `to_vtm` (volumen + una pieza por
  frontera) y `write_msh` (gmsh 2.2 binario en orden nativo, zonas y fronteras como grupos
  físicos, también vacías e interiores).
* **Decisión abierta cerrada:** `write_msh` **corrige** el orden (no solo diagnostica). Caso
  real: el `Wind_turbine_disk_empty_g4.msh` de ANSA que PyHOPE rechazaba. `fix_order` da la
  permutación con 100 % de acuerdo y 0 celdas mal; gmsh lee el fichero reescrito; **PyHOPE
  genera un `.h5` idéntico conjunto a conjunto al del `.cgns`** (diferencia 0 en
  `NodeCoords`, `ElemInfo`, `SideInfo`…).
* Tablas nuevas en `ordering.py`, medidas: `VTK_HEX`/`VTK_QUAD` con `GetParametricCoords` de
  VTK 9.5 (la fórmula de `PointIndexFromIJK` que escribí de memoria fallaba desde orden 2),
  `GMSH_QUAD` sobre los cubos de referencia. `detect_hex_order`: inversión trilineal de cada
  nodo y voto.
* **Trampa de VTK:** con `version="1.0"` en el `.vtu`, VTK ≥ 9.1 cambia dos aristas del
  hexaedro Lagrange al leer (compatibilidad con el orden antiguo). Se escribe `version="2.2"`.
* `locate.py` y `LEGOLAS.Locate` (`lg.locate`): `points` (Newton sobre el mapeo curvo de
  cualquier orden, tetraedros por baricéntricas, prefiltro k-d + cajas) y `horses3d_ad`, réplica
  de `FindActuatorPointElementAD` (primer elemento trilineal por rango, luego la curva);
  `read_actuator_points` lee el `.dat` (modelos A y B). Generaliza `check_ad_points.py` (que
  solo sabía Ngeo = 2) y coincide con él número a número: 0 perdidos con el `.dat` actual,
  48 con la última estación en r = 0,446, |ξ| máximo 1,048558.
* Tests: `test_legolas_convert.py` y `test_legolas_locate.py` (33). Ejemplo:
  `examples/legolas/convert_locate_turbine.py`.

### Pendiente

* L7: `LEGOLAS(db, case)` desde FRODO y README.
* L8, siguientes pasos: estadísticas (`.stats.hsol`, que el núcleo aún no descifra), caras
  dibujadas al orden de la solución cuando p > Ngeo, gradientes (vorticidad, criterio Q),
  superficies.

## L8 (24/11/2026): soluciones de HORSES3D en el visor

Decisiones del usuario: **valores por nodo directamente** (sin pasar por la media por
elemento), **entrada por LEGOLAS**, deslizador después.

* `solution.py` y `LEGOLAS.Solution` (`lg.solution`): `load(hsol, physical)` evalúa el
  polinomio de cada elemento (nodos de Gauss o GL de orden p) en los nodos geométricos de la
  malla; `node_field`, `cell_field` (cuadratura), `point_field` (promedio por nodo, para VTU),
  `viewer_fields`, `summary`. Variables primitivas con
  `solver_tools.horses3d.primitive_variables`, calculadas en los nodos de solución.
* `LEGOLAS.from_control(control, solution=None, physical=False)`: malla y solución del propio
  `.control` (claves `mesh file name`, `solution file name`), como HORSES3D. Así no hay que
  adivinar cuál de las mallas de `MESH/` es la buena.
* **Validación clave:** la geometría de LEGOLAS en los nodos de solución reproduce el `.hmesh`
  del solver a 1e-14 (turbina Gauss p2/p3/p4, esfera GL p2): orden de elementos, de nodos y
  posiciones correctos. Está en los tests (se salta si no están los casos).
* Visor: las superficies llevan `vert_index` (posición de cada vértice en un array por
  (celda, nodo), `Mesh.cell_node_offsets`); `/api/nodefield`; grupo «Solución» en el selector;
  la página arranca coloreada por Mach si hay solución y su cabecera dice fichero, p, nodos,
  t e iteración. `show()` pasa al proceso del servidor la ruta del `.hsol` (lo relee, 2 s) y
  acepta `node_fields=` de cualquier origen.
* Mariño A, p3 (t = 40): u de 0,474 a 1,075; en una capa z = 0,81 se ve la estela del disco.
* Tests: `test_legolas_solution.py` (15). Ejemplo: `examples/legolas/solution_marino.py`.

### Deslizador (24/11/2026, después de L8, como pidió el usuario)

* `Solution.series()` con `solver_tools.horses3d.find_solutions`: autoguardados completos del
  mismo nombre y p, por iteración.
* Servidor: `info['series']`, `info['frame']` y `/api/nodefield?…&frame=k`; cada instante se lee
  al pedirlo (3 soluciones en memoria y las últimas 24 respuestas). `show(series=False)` y
  `server(series=False)` lo quitan.
* Página: panel **Tiempo** (deslizador, ◀ ▶, reproducir) y «escala de color fija», que conserva
  la escala del instante donde se eligió la variable.
* Mariño A p2: 11 instantes (t = 0 a 30); la estela nace a t = 3 y llega a la salida a t = 15.
* Tests: 2 más en `test_legolas_solution.py` (17 en total).

### Estadísticas y valores en puntos (24/11/2026, para la validación de Mariño)

- `solver_tools.horses3d.read_stats`: lee los `.stats.hsol` (tipo 4). Cada registro tiene una
  cabecera con dimensiones (9, n, n, n), los 9 momentos U V W UU VV WW UV UW VW y, sin cabecera,
  las 5 conservativas medias (y 3 gradientes si se guardan; el número de gradientes se deduce
  del tamaño). Orden uniforme; la p-adaptación lanza `NotImplementedError`.
- `lg.solution.load()` acepta estadísticas: u v w Vmag (medias), uu vv ww uv uw vw (tensiones de
  Reynolds = ⟨ab⟩ − ⟨a⟩⟨b⟩), tke y rho; con `physical=True`, ×V_ref, ×V_ref² o ×ρ_ref.
- `lg.solution.sample(xyz, variables)`: `locate.points` + el polinomio del elemento en ξ (los
  valores en los nodos de solución se guardan al cargar). Fuera de la malla, NaN.
- Comprobado con el caso real: muestreado en la posición de las sondas de A con p2, coincide con
  ellas hasta la cuarta cifra (test `test_statistics_of_the_turbine_match_its_probes`). 551 tests.

### Propuesta de la que salió L8 (24/11/2026)

**Qué hay.** En disco hay `.hsol` (final, `.stats.hsol`, autoguardados), `.hmesh` por p, `.tec`
de `horses2plt` y planos `.pvd`/`.vtu` de `extract_plane.py`. No hay soluciones en HDF5:
HORSES3D solo escribe la malla en `.h5`. El `.hsol` sigue el orden de elementos del `.h5` de la
malla (38 718 en la turbina), así que cada elemento cae en su celda de LEGOLAS sin búsqueda, y
se lee en 0,4 s con `solver_tools/horses3d.read_solution`. Los casos de Mariño usan nodos de
**Gauss** (no están en las caras) y la esfera Gauss-Lobatto: el `.hmesh` no sirve como malla del
visor; hay que interpolar por elemento. El `MESH/` de un caso puede tener varias mallas: cuál
va con cada solución lo sabe el reader (`sim_metadata`), no un `glob`.

**Reparto.** LEGOLAS sigue sin saber de HORSES3D: dibuja campos que le pasan. HORSES3D aporta
los campos: núcleo numpy en `solver_tools/horses3d.py` (interpolación de Gauss/GL de orden p a
puntos de dibujo, variables primitivas, medias por elemento) y orquestación en `Horses3DSets`
(`db.view(case, p, kind)`: malla del caso según el reader, campos de la solución elegida,
`lg.viewer.show(fields=...)`). Es lo mismo que necesita el M2 del plan del `.hsol`
(`HorsesLazyField.load`).

**Pasos.**
1. *Por celda* (ya funciona el canal): medias por elemento (cuadratura de Gauss) de ρ, u, p,
   T, Mach… Un color por elemento; sin tocar el visor.
2. *Por nodo*: la solución interpolada a los nodos de dibujo de cada cara, con su valor por
   elemento (sin promediar: el visor ya duplica los vértices por cara, así que las
   discontinuidades de DG se ven tal cual). El visor gana un campo por (celda, nodo local).
3. *Orden de dibujo*: caras subdivididas a orden max(Ngeo, p), con geometría y solución
   evaluadas ahí, para no perder detalle cuando p > Ngeo.
4. *Series*: deslizador de autoguardados y medias del `.stats.hsol`; exportar a `.vtu` con
   `to_vtu(point_fields=…)` para ParaView.
5. Después: superficies (`.surf.hsol`), y el y⁺ real enlazado con `lg.resolution`.

**No se propone** leer `.tec` (ASCII grande, lento; lo mismo sale del `.hsol` con
`primitive_variables`) ni los planos VTK externos (son superficies sueltas, otro tipo de capa).

## Auditoría (25/11/2026)

> Revisión del plan contra código, tests, documentación generada e historial Git. Se han
> documentado las propuestas para que Claude las valore; **no se ha corregido ningún fallo**
> ni se han tocado otros ficheros. Método: `graphify query/explain/path` antes de leer código;
> lectura puntual de las líneas localizadas; recogida de tests con
> `/home/migueljaraiz/anaconda3/envs/envkan_nvidia/bin/python -m pytest FotR/tests/
> --collect-only -q -p no:cacheprovider` (551 tests recogidos, 176 en `test_legolas_*.py`).
> Línea base verificada: `HEAD=81da02a`, rama `develop`, árbol limpio, sin rama local
> `legolas`.

### Resumen

| ID | Prioridad | Tipo | Hallazgo |
|---|---|---|---|
| LG-01 | Media | Documentación | La “Propuesta” final contradice L8: LEGOLAS ya lee soluciones HORSES3D. |
| LG-02 | Media | Planificación | L7 sigue pendiente aunque L8 está cerrado y no hay commit L7. |
| LG-03 | Media | Documentación | Estadísticas implementadas, pero HANDOFF y guías generadas las dan por pendientes. |
| LG-04 | Media | Documentación | Regla de BC interiores mal explicada en `HANDOFF.md`. |
| LG-05 | Baja | Robustez | Validación y manejo de errores incompletos en el visor. |
| LG-06 | Baja | Robustez | Caché FIFO y posible desajuste del frame inicial. |
| LG-07 | Baja | Documentación | Cifras por hito desactualizadas; el total 551 sí coincide. |
| LG-08 | Media | Documentación | Estado Git y métricas de graphify en `HANDOFF.md` son viejos. |
| LG-09 | Baja | Documentación | README habla de dos visores y no menciona el generador de `doc/legolas/`. |
| LG-10 | — | No incidencia | `p > Ngeo` sigue pendiente y está correctamente documentado. |

### LG-01 — Propuesta final obsoleta tras L8

- **Evidencia:** `doc/PLAN_LEGOLAS.md:424-429` afirma que “LEGOLAS sigue sin saber de
  HORSES3D” y propone que `Horses3DSets` orqueste `db.view(case, p, kind)`. La sección L8
  del propio plan (`:369-386`) ya describe lo contrario: `LEGOLAS.from_control()` en
  `FotR/characters/legolas/__init__.py:107-129` y `LEGOLAS.Solution` en
  `FotR/characters/legolas/solution.py:103-378` leen malla y solución. `Horses3DSets` no tiene
  `view`; solo `to_tecplot` y auxiliares.
- **Propuesta:** convertir esa “Propuesta” en registro histórico o reescribirla como “estado
  final tras L8”, indicando que la responsabilidad quedó en LEGOLAS y que el canal de campos
  sigue sirviendo para `node_fields=` externos.

### LG-02 — L7 sin cerrar

- **Evidencia:** `doc/PLAN_LEGOLAS.md:153,189,196,359` mantiene L7 (`LEGOLAS(db, case)` y
  README) como pendiente, pero L8 figura como hecho. `FotR/characters/legolas/__init__.py:90-105`
  solo acepta una ruta, un `Mesh` o `from_control`; no acepta `db`/`case`. No hay commit
  etiquetado L7 entre `ceaeaa4` (L6) y `3df459d` (L8).
- **Propuesta:** decidir una de tres salidas y documentarla: (a) implementar el puente FRODO,
  (b) darlo por sustituido por `from_control()` y cerrar L7 como “no necesario”, o (c)
  renombrar el hito. Mientras tanto, la tabla de estado resulta confusa.

### LG-03 — Estadísticas: documento y código divergentes

- **Evidencia:** `doc/PLAN_LEGOLAS.md:400-411` documenta `read_stats`, tensiones de Reynolds,
  `tke`, `rho` y `sample`. El código está en `FotR/solver_tools/horses3d.py:492-568` y
  `FotR/characters/legolas/solution.py:82-95,185-208`. Sin embargo
  `doc/HANDOFF.md:160-162` las incluye en “Pendiente”, y
  `doc/legolas/guides.json:572` y `:1557` todavía muestran el mensaje
  `file type 4 (STATS) is not supported yet`, en contradicción con `:967` del mismo archivo.
- **Propuesta:** quitar las estadísticas de la lista pendiente del HANDOFF y regenerar
  `doc/legolas/guides.json` desde `doc/legolas/generate_docs.py`.

### LG-04 — Regla de zonas interiores

- **Evidencia:** el plan acierta en `:269-272`: HORSES3D exige `type` para toda zona **con
  caras**, también las interiores; las zonas vacías no. PyHOPE exige tipo 100 para
  interiores. `FotR/characters/legolas/io/hopr.py:18-21` marca `BCType 100` precisamente en
  caras con elemento a ambos lados. `doc/HANDOFF.md:416-418` dice que esas zonas “no tienen
  caras”, lo cual es incorrecto.
- **Propuesta:** corregir el HANDOFF a “interiores incluidas cuando tienen caras; las vacías
  no requieren declaración”. El plan no necesita cambio, pero conviene citar aquí la errata.

### LG-05 — Visor: validación y errores HTTP

- **Evidencia:** `FotR/characters/legolas/viewer.py:70-91` valida `better` en campos por celda,
  pero `_normalise_node_fields` en `:94-110` no lo hace. El frente interpreta cualquier valor
  distinto de `'high'` como “low es bueno” (`viewer.html:566-569`). El flujo normal
  `FotR/characters/legolas/solution.py:355-360` siempre emite `better=None`, así que afecta
  solo a `node_fields=` manuales mal formados.
- **Evidencia 2:** `do_GET` en `viewer.py:480-528` captura solo `KeyError` y `ValueError`; un
  `OSError` al leer un frame borrado u otro error interno puede cerrar la conexión sin JSON.
- **Propuesta:** aplicar la misma validación en ambos normalizadores y añadir un `except
  Exception` final que devuelva 500 y deje constancia en el log.

### LG-06 — Caché y frame inicial

- **Evidencia:** `viewer.py:364-377` y `:419-424` eliminan la entrada más antigua insertada,
  no la menos usada; el plan `:392-394` dice “últimas 24 respuestas” y no promete LRU, así
  que es una precisión de documento, no un bug. En `:219-222`, `base_frame` solo se fija si
  el nombre del `.hsol` cargado aparece en la serie; `viewer.html:483-486,710-717` cae al
  último autosave cuando `frame` es `null`.
- **Propuesta:** decidir si se documenta el comportamiento FIFO o se implementa LRU; cargar
  un `.hsol` final con nombre distinto a los autosaves y comparar cabecera y frame inicial.

### LG-07 — Recuento de tests

- **Evidencia:** el total del plan (`:411`, 551) coincide con la recogida actual. Los
  desgloses por hito no: `test_legolas_solution.py` recoge 23 tests frente a los 15/17
  citados en `:386,398`. El desglose actual es: `io` 35, `viewer` 23, `census` 19, `quality`
  24, `resolution` 19, `convert` 26, `locate` 7, `solution` 23.
- **Propuesta:** actualizar los números por hito; dejar la cifra global atada a una fecha y a
  un entorno concreto.

### LG-08 — Estado Git y graphify en el HANDOFF

- **Evidencia:** `doc/HANDOFF.md:7,25-28` describe `develop @ 6126066`, trabajo sin
  commitear y una rama `legolas` que no existe; el estado real es `81da02a`, limpio, en
  `develop`. `doc/PLAN_LEGOLAS.md:189` dice “L8 el siguiente” aunque L8 está en `3df459d`.
  `doc/HANDOFF.md:361-362` cita 2035 nodos / 3944 aristas / 99 comunidades; el
  `graphify-out/GRAPH_REPORT.md:10` actual dice 2840 / 6133 / 146.
- **Propuesta:** sincronizar el HANDOFF con `git log`, `git branch` y el informe actual de
  graphify; actualizar la tabla del plan.

### LG-09 — Documentación de navegación

- **Evidencia:** `README.md:128` habla de “los dos visores” y `:140-141` solo lista los
  generadores de FRODO/SAM y GANDALF, aunque `doc/legolas/` existe y tiene su propio
  `generate_docs.py`. `doc/frodo_sam/guides.json:444,949` dice que visor, calidad y resolución
  están “en camino”, todos ellos implementados.
- **Propuesta:** añadir el tercer visor y su comando de regeneración; actualizar las dos
  frases de `guides.json` regenerando la guía.

### LG-10 — Dibujo a orden `max(Ngeo, p)`

- **Evidencia:** `doc/PLAN_LEGOLAS.md:357-362,431-438` lo deja como trabajo futuro;
  `FotR/characters/legolas/solution.py:221-229` interpola a nodos geométricos `Ngeo` y
  `scene.py` dibuja la rejilla disponible. Las guías lo declaran como no implementado.
- **Conclusión:** limitación documentada, **no un bug**. Solo debe mantenerse visible en la
  lista de pendientes.

### Revisado sin incidencias confirmadas

- Orden canónico tensorial y tablas de permutación (`ordering.py`, `convert.py`).
- Jacobiano escalado y giro por bloque antes de tomar el mínimo (`quality.py`).
- `h`, Δ_DG, primera celda, crecimiento y y⁺ estimado (`resolution.py`).
- `detect_order`/`fix_order`/`write_msh` y búsqueda de puntos (`convert.py`, `locate.py`).
- Carga de soluciones, series, estadísticas y escalado de tensiones de Reynolds
  (`solution.py`, `solver_tools/horses3d.py`). La p-adaptación sigue siendo `NotImplementedError`
  y está documentada (`solution.py:193-195`).
