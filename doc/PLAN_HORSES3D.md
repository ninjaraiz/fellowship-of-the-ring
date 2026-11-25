# PLAN — Pipeline HORSES3D: GANDALF → casos → lanzamiento → FRODO

> Acordado con el usuario el 18/11/2026. Estado de cada paso al final. Resumen operativo en
> [`HANDOFF.md`](HANDOFF.md).

## Objetivo

Que una campaña de HORSES3D se cree, se lance y se lea con FotR sin montar nada a mano, con
la misma estructura que CODA:

```
<raíz>/
├── metadata/   cases_metadata.json, df_cases.csv, (LEEME.md del usuario)
├── sources/    (opcional, del usuario: mallas de partida, plantillas)
└── outputs/<caso>/
    ├── file_control_p2.control      ← uno por p (escalera de restart)
    ├── file_control_p3.control
    ├── fsp_p2.control               ← prueba FSP (no la lee FRODO)
    ├── run.sh -> utils_H3d/run.sh
    ├── MESH/<malla>_g<g>_mesh.h5    ← copia
    ├── ActuatorDef/ … SETUP/ …      ← opcionales
    ├── RESULTS/<tag>_g<g>p<p>.hsol  (+ FSP/<tag>_g<g>p<p>_fsp.*)
    └── LOG/
```

Primer uso: la turbina del Blind Test de la NTNU (`~/Documentos/COMAC/wind_turbine_ansa`,
malla `Wind_turbine_disk_fluid_cgns_g2_mesh.h5`). Sus condiciones iniciales y su física
quedan para después: el pipeline no depende de ellas.

## Decisiones

| tema | decisión |
|---|---|
| nombres | la malla lleva `g<g>` en el nombre; cada solución es `RESULTS/<tag>_g<g>p<p>.hsol` (etiqueta corta de campaña + g de la malla + p del control) |
| plantilla | un `.control` real; el ring reescribe **claves** (`key = value`), no marcadores; los `#define` no se tocan y los comentarios se quitan (ficheros de caso sin comentarios) |
| malla | se **copia** en `MESH/` de cada caso |
| caso sin `.hsol` | estado **pendiente**, no error |
| módulo común ring/reader | **no hay**: como en CODA, el ring escribe y el reader lee por su cuenta; el puente es `cases_metadata.json` (columna `folder`). La regla de `g` se duplica (una expresión regular) y la garantiza un test de ida y vuelta |

## 1. Ring `horses3d` — `FotR/characters/rings/horses3d.py`

`Horses3DRing(BaseRing)`, registrado como `RING_REGISTRY['horses3d']`.

- `__init__` propio: sin `num_stages` ni `eq_type` de CODA (acepta y guarda lo que pase la
  fachada). Crea `metadata/`.
- Hereda `define_cases`, `add_param`, `compute_param`.
- `generate_folders(control_template, mesh_path, tag, folder_fmt, p_levels=(2,),
  control_values=None, p_values=None, extra_dirs=None, data_to_update=None,
  setup_dir=None, run_script=None, fsp=True, fsp_steps=..., overwrite=False)`:
  - malla: existe, `Ngeo` del `.h5` y `g<Ngeo>` en el nombre; todo `p ≥ g`;
  - por caso: `MESH/` (copia), `RESULTS/`, `LOG/`, enlace a `run.sh`;
  - por p: `file_control_p<p>.control` con `mesh file name`, `Polynomial order`,
    `solution file name` y la escalera de restart (el primer p desde cero; cada siguiente
    reinicia desde el anterior con `restart polorder`); `control_values` =
    `{clave: columna de df_cases | valor}`, `p_values` = `{p: {clave: valor}}`;
  - `extra_dirs` (p. ej. `ActuatorDef/`) copiados, con marcadores vía `data_to_update`
    (reutiliza `_resolve_update_value` de `rings/coda.py`);
  - `setup_dir` → `SETUP/` (se compila aparte con `build_setup.sh`);
  - `cases_metadata.json` con `solver`, `tag`, `mesh` (fichero, g), `p_levels`, `fsp`.
- **Modo FSP (Free-Stream Preservation)** — prueba de la malla, no física: `fsp_p<p>.control`
  con `Inflow` en todas las fronteras de la malla (las de `BCType` 100, interiores, fuera),
  sin línea/disco actuador, IBM, monitores, superficies ni restart, pocos pasos y salida en
  `RESULTS/FSP/`. Una malla válida conserva la corriente libre: la solución final no se mueve de la inicial
  más que el redondeo (`fsp_report` compara las dos; los residuos no sirven de criterio).
  FRODO no lo lee (el nombre no es `file_control_p<p>.control`).
- `submit_cases(p, repo, arch='GPU', node='n001', ngpu=1, cpus_per_gpu=16, branch=None,
  fsp=False, dry_run=True)`: `HORSES3D_REPO=<repo> ./run.sh <control> …` en cada caso; salta
  los que ya tienen `.hsol` o `.residuals`; `run.sh` detecta el clúster. `repo` es
  obligatorio (ver la entrada del 20/11 del diario).
- `load_campaign()`: recupera la campaña de `cases_metadata.json` en una sesión nueva. `assign_jobs` (plantillas `%NODO` de CODA)
  no aplica: error explicativo.
- `fsp_report(p)`: lee `RESULTS/FSP/*.residuals` y dice por caso si conserva la corriente libre.

## 2. Reader — `FotR/characters/readers/horses3d.py`

- `g`: token `g<n>` del nombre de la malla (vale `..._cgns_g2_mesh.h5` y `..._v2g2_mesh.h5`),
  con el atributo `Ngeo` del `.h5` como respaldo.
- Estado por solución (`status`: `pending` / `started` / `finished`, por `.hsol` final con
  terminador 99, autosaves y `.residuals`; `started` cubre «corriendo» y «cortado antes de
  acabar»); en `df_state`, `p_finished` y `state`.
- Ya no son error: `.hsol` ausente (pendiente), restart cuyo fichero producirá un p anterior
  del mismo caso, `restart polorder ≥ p` (aviso: continuar un p desde otro caso es legítimo).
  Siguen siendo error: p del nombre ≠ control, p < g, mallas distintas en un caso, malla
  inexistente.
- Ficheros relacionados: autosaves `<stem>_<iter10>.hsol`, salidas del actuador
  `<stem>_Actuator_*`, sondas `<stem>.<n>.probe`.
- `extract_outputs`: error claro si alguna solución pedida no está terminada.

## 3. Tests

- `test_rings_horses3d.py`: carpetas, controles por clave, escalera de restart, copia de
  malla, marcadores, JSON, FSP, errores; `submit_cases` en seco.
- Ida y vuelta: ring → `Horses3DReader(strict=True)` (todo pendiente) → `.hsol` sintético
  completo (terminado).
- `test_horses3d.py`: `g` nuevo, estados; los que esperaban error por `.hsol` ausente.

## 4. Documentación

`doc/gandalf/` (TARGETS, tabla de rings ES/EN, i18n, `examples.json`), `doc/frodo_sam/`
regenerado, ejemplo `examples/gandalf/horses3d_wind_turbine.py`, HANDOFF, `graphify update .`.

## 5. Después (fuera de este plan)

Turbina: modelo de rotor, IBM, condiciones iniciales y física (las decide el usuario); FSP y
prueba corta antes de producción. Luego: monitor en directo para HORSES3D, hitos M2–M5 del
`.hsol`, LEGOLAS desde cero.

## Estado

| paso | estado |
|---|---|
| 1. Ring + FSP | hecho (18/11/2026) |
| 2. Reader | hecho (18/11/2026) |
| 3. Tests | hecho: 39 nuevos, 354 en total (19/11/2026) |
| 4. Documentación | hecho: visores regenerados, ejemplo, HANDOFF §13 |
| 5. Turbina | FSP p2 superada (job 4271, 19/11/2026); falta decidir la física del caso real |
| 6. Tecplot con `horses2plt` | hecho (24/11/2026): `db.to_tecplot`, validado con Mariño 2024 A p2 |

## Diario

- **18/11/2026, 21:04.** El usuario escribió `sources/wind_turbine.control` (plantilla con
  BC FSP propias), generó `tsr_6.0` (sin `ActuatorDef`) y lanzó la FSP p2 (job 4265). El
  solver paró al leer el control: «Input file is missing entry for keyword: restart file
  name». El ring quitaba esa clave con `restart = .false.`, y HORSES3D la exige siempre
  (`Setup.f90`, guía §4.3).
- **19/11/2026.** Arreglado en el ring:
  - `restart file name` siempre presente (sin restart apunta a la propia solución, como en
    `my_ESPHERE_1`);
  - antes de escribir nada se comprueban las claves obligatorias de HORSES3D (`mesh file
    name`, `restart`, `restart file name`, `solution file name`, `number of time steps`,
    `output interval`, `convergence tolerance`; en NS además `mach number` y `reynolds
    number`);
  - `overwrite=True` reescribe controles, malla, `extra_dirs`, `SETUP` y `run.sh` pero
    **nunca** borra `RESULTS/` ni `LOG/` (regla de no borrar resultados);
  - `submit_cases` en una sesión nueva recupera la campaña antes de nombrar soluciones.

  Controles de `tsr_6.0` regenerados con `overwrite=True` (los logs del job 4265 se
  conservan) y FSP p2 relanzada: job 4270.
- **19/11/2026, job 4270.** Pasó el control y paró al leer la malla: «Missing boundary
  condition type for boundary rotor_cylinder» (`ZoneClass.f90:344`). HORSES3D crea una zona
  por cada nombre de `BCNames` y exige un `type` para todas, también las interiores (que se
  quedan sin caras: cualquier tipo vale). Los nombres se pasan a minúsculas, así que
  `sideA`/`sidea` da igual. Arreglado: la FSP declara todas las zonas; antes de escribir, el
  ring exige un `#define boundary` para cada zona de la malla y avisa de monitores o
  `boundaries to save` sobre zonas inexistentes (la plantilla de la turbina tiene monitores
  sobre `nacelle__edge`, que esta malla no tiene).
- **19/11/2026, job 4271.** FSP p2 completa: 200 pasos, 12 s de solver (0,057 s por millón
  de GdL e iteración), salida 0. **Malla válida:** la solución se mueve como mucho 4,9e-9
  (ρu) y 7,5e-8 (ρE; 3,6e-11 relativo) respecto al estado inicial; mediana por nodo 3e-12.
  Peor elemento 36418, en (6,24; 1,11; 0,61), dentro del cilindro de refinado aguas abajo.
  Los residuos, en cambio, valen hasta 1,6e-4 desde la iteración 0 con la solución aún
  uniforme: redondeo de flujos de energía grandes (M = 0,029) sumado sobre 10⁶ GdL. Por eso
  `fsp_report` ya no juzga por residuos sino por la desviación de la solución final frente a
  la de la iteración 0 (tolerancia por defecto 1e-6 relativo) y da el peor elemento.
- **19/11/2026, zonas interiores.** Duda del usuario: en un caso real, las superficies
  interiores (`rotor_cylinder`, `disk_side`) deberían ser «interior», y HORSES3D no tiene esa
  opción. No hace falta: el log del job 4271 da `Number of faces: 0` para las dos zonas, y las
  4682 caras de frontera son exactamente las exteriores (2·478 + 2·729 + 2·1134).
  `Zone_AssignFaces` (`ZoneClass.f90`) salta las caras con elemento a los dos lados, así que
  el `type` de esas zonas es un requisito del lector y no se aplica nunca. Decidido: la
  plantilla solo declara fronteras reales; si faltan zonas interiores (`BCType` 100), el ring
  añade a cada control `#define boundary … type = FreeSlipWall` y lo imprime en la salida
  (`print`, para que se vea en el notebook aunque se repita la celda). Si falta una frontera
  real, error.
- **19/11/2026, documentación.** Segundo ejemplo del ring en el visor de GANDALF:
  `examples/gandalf/horses3d_sphere_re_mach.py` (esfera de `my_ESPHERE_1`, 2×2 casos de Mach y
  Reynolds escritos por caso en el control, p = 2 → 3 con `p_values[3]`; probado en el
  scratchpad). Nueva sección de la guía «Qué se puede cambiar del .control desde GANDALF»:
  claves del ring (malla, p, solución: no), escalera de restart (sí, con `p_values`), resto
  fuera de `#define` (sí), bloques `#define` (no, se copian), orden de aplicación y cómo se
  escribe cada tipo de valor Python. Las claves del ring en `p_values` pasan a ser error,
  igual que en `control_values` (antes se ignoraban sin avisar).
- **20/11/2026, repositorio del solver explícito.** Para el disco actuador hace falta la rama
  `arr/actuator_disk` (PR #121 de horses3d-gpu), clonada y compilada aparte en
  `~/repos/horses3d-gpu-ad` (los repos de siempre son de solo lectura). `run.sh` y
  `build_setup.sh` aceptan `HORSES3D_REPO=<clon>`, y `submit_cases(p, repo, …)` lo exige: lo
  pasa siempre a `run.sh`, ignora el que haya en el entorno y, antes de planificar, comprueba
  que exista `Solver/bin/horses3d.<solver>` y que el repo esté en una rama. `branch` pasa a
  ser opcional (por defecto, la rama del repo; si se da, tiene que coincidir). El `SETUP` se
  compila con el mismo `HORSES3D_REPO` y `run.sh` lo comprueba. Para compilar el solver en
  nibbler: `utils_H3d/build_solver.sh <clon>` (lleva `-gpu=cuda12.6`, sin la cual nvfortran
  falla donde no ve una GPU).
- **20/11/2026, Flexo.** El `git` de Flexo no acepta `git -C`: los scripts usan `repo_git` y el
  ring llama a `git symbolic-ref HEAD` con `cwd`. Un `Marker` de monitor de superficie nombra una
  sola zona: HORSES3D no lo divide por `__` y, si no existe, se cae (job 52370, SIGSEGV en
  `VectorSurfaceIntegral`). Por eso el ring lo trata como error, y un nombre desconocido en
  `boundaries to save` como aviso.
- **22/11/2026, órdenes encadenados.** `submit_cases` acepta `p` como lista ascendente
  (`[2, 3, 4, 5]`). En cada caso, cada orden se envía con `DEPENDENCY=afterok:<job anterior>`.
  `run.sh` lo pasa a `sbatch` con `--kill-on-invalid-dep=yes` y no exige el fichero de reinicio
  que escribirá ese trabajo. Un orden ya empezado no deja dependencia; uno fallido corta la
  cadena del caso. El resultado gana las columnas `p` y `depends_on`. Primer uso: la turbina con
  disco actuador (`~/Documentos/COMAC/wind_turbine_ansa_AD`), p2 → p5 con 4 GPU en nibbler.
- **22/11/2026, recursos por orden.** `ngpu` y `ncpus` aceptan `{p: n}`. Se midió en n001 (H200
  NVL): ninguna GPU pasa del 30 % de uso, porque el caso es pequeño (38718 elementos) y el disco
  actuador copia a la CPU, en cada paso, el elemento de cada uno de sus 2304 puntos (unos 33–45 ms
  por paso). Con 1 GPU, p2 tarda 0.112 s/paso y p5 0.340; con 4 GPU, 0.065 y 0.134. Para la turbina
  se eligió `{2: 1, 3: 1, 4: 1, 5: 2}` (unas 37 h y 49 GPU·h). El notebook
  `examples/gandalf/horses3d.ipynb` prepara y lanza ese caso.
- **24/11/2026, `horses2plt` desde FRODO** (decisiones del usuario: FotR + script de
  `utils_H3d/POST`; salida en `POST/tec/`, `RESULTS/` intacto; por defecto finales,
  estadísticas y autoguardados; unidades SI).
  - `FotR/solver_tools/horses2plt.py` (solo biblioteca estándar): elige los ficheros completos
    del stem (`select_files`), la malla del mismo orden (`mesh_for`: `MESH/<stem>_<iter>.hmesh`),
    enlaza en `POST/tec/p<p>/` (`stage`, rutas relativas: el modo control lee 132 caracteres) y
    escribe el control (`control_text`): FE, Gauss → `Homogeneous` con orden p (sin huecos entre
    elementos), `has gradients` en estadísticas si el caso guarda gradientes.
  - `Horses3DSets` (registrado: HORSES3D ya tiene sets) con `to_tecplot(binary, script, p, kinds,
    cases, variables, stats_variables, dimensionless, iterations, run='sbatch'|'local', node,
    cpus, mem_gb, overwrite, dry_run=True)`. El simulacro no escribe nada; un `sbatch` por caso
    (log en `LOG/horses2plt_<job>.log`); lo convertido se salta; estados `planned`, `skipped`,
    `incomplete`, `no-mesh`, `submitted`, `converted`, `failed`.
  - `utils_H3d/POST/horses2plt_batch.slurm`: módulos por clúster (`H2P_CLUSTER`,
    `SLURM_CLUSTER_NAME` o `slurm.conf`), `xargs -P`, comprueba que el `.tec` existe.
  - Reader: `stats` (`<stem>.stats[.<iter>].hsol`) y `tec` (`POST/tec/*/`) por solución.
  - **Validado en n008 (`run='local'`) con A p2**: final + 2 estadísticas en 22 s; coordenadas
    iguales a las de FotR interpoladas de Gauss a equiespaciado (4e-11, redondeo del `.tec`),
    ρ = ρ_ref·ρ*, ρu = ρ_ref·V_ref·(ρu)* (SI); `umean` = media de las sondas en el eje (x/D = 1,
    3, 5: diferencias ≤ 0,007 m/s). `Vfrms` daba `NaN` (estela estacionaria: √ de redondeo
    negativo) → fuera de las variables por defecto de estadísticas.
  - En n008 no hay `sbatch` (solo `squeue`): el envío se hace desde el maestro de flexo o
    desde nibbler. Superficies (`.surf.hsol`) fuera: el binario no las lee.
  - Guía de HORSES3D §8.1–8.2 (binario = `ab3b02326`, no `7624e34ed`), clave `output variables`
    corregida en `POST/horses2plt-surface.control`. 492 tests (11 nuevos).

## Auditoría (25/11/2026)

> Revisión del plan contra código, tests, documentación generada e historial Git. Se han
> documentado las propuestas para que Claude las valore; **no se ha corregido ningún fallo**
> ni se han tocado otros ficheros. Método: `graphify query/explain/path` antes de leer código;
> lectura puntual de las líneas localizadas; recogida de tests con
> `/home/migueljaraiz/anaconda3/envs/envkan_nvidia/bin/python -m pytest FotR/tests/
> --collect-only -q -p no:cacheprovider` (551 tests recogidos).
> Línea base verificada: `HEAD=81da02a`, rama `develop`, árbol limpio.

### Resumen

| ID | Prioridad | Tipo | Hallazgo |
|---|---|---|---|
| HS-01 | Media | Documentación | La firma de `submit_cases` y el criterio de `fsp_report` del plan están desactualizados. |
| HS-02 | Media | Documentación | El plan dice que los `#define` no se tocan y que la FSP no tiene restart; el código hace ambas cosas. |
| HS-03 | Media | Cobertura | La regla `g` duplicada no se comprueba de ida y vuelta en los casos límite. |
| HS-04 | Media | Decisión/code | La cadena de `submit_cases` pierde la dependencia si el p anterior solo tiene `.hsol`/`.residuals`. |
| HS-05 | Media | Documentación | `Ngeo` no es solo respaldo: desajuste con el nombre es error en modo estricto. |
| HS-06 | Media | Bug latente | `to_tecplot` decide “ya convertido” solo por `mtime`. |
| HS-07 | Alta | Trazabilidad | `horses2plt_batch.slurm` se declara hecho pero no está versionado. |
| HS-08 | Media | Reproducibilidad | Ejemplos/tests usan rutas externas absolutas. |
| HS-09 | Media | Documentación | `HORSES3D` tiene `sets`, pero README/HANDOFF/examples dicen lo contrario. |
| HS-10 | Baja | Documentación | Cifras de tests y el estado de la física del caso real quedaron obsoletos. |

### HS-01 — Firma de `submit_cases` y criterio de FSP

- **Evidencia:** el plan describe `submit_cases(p, repo, arch='GPU', node='n001', ngpu=1,
  cpus_per_gpu=16, branch=None, fsp=False, dry_run=True)` en `doc/PLAN_HORSES3D.md:66-69`,
  pero la firma real en `FotR/characters/rings/horses3d.py:817-831` añade `ncpus`, `solver`,
  `cases`, admite `p` como lista y recursos `{p: n}`. El propio diario ya documenta esos
  cambios en `doc/PLAN_HORSES3D.md:188-199`.
- **FSP:** el plan dice en `:72` que `fsp_report` lee `RESULTS/FSP/*.residuals` y decide con
  ellos; `FotR/characters/rings/horses3d.py:1030-1099` decide por la desviación entre la
  solución inicial y la final (`:1102-1119`) y usa los residuos solo como información. El
  diario en `:153-154` ya recoge la decisión correcta.
- **Propuesta:** actualizar la sección 1 del plan con la firma real y el contrato real de
  `fsp_report`, o marcar explícitamente esa sección como especificación histórica.

### HS-02 — `#define` y restart en la FSP

- **Evidencia:** `doc/PLAN_HORSES3D.md:35` afirma “los `#define` no se tocan”, pero
  `FotR/characters/rings/horses3d.py:112-124` permite modificarlos con
  `'<block header>: <key>'`, `:651-653` añade bloques `#define boundary` para zonas
  interiores y `:700` elimina todos los bloques en `fsp_control`. La guía en
  `doc/PLAN_HORSES3D.md:168-172` ya describe el comportamiento real.
- **Restart:** `doc/PLAN_HORSES3D.md:60-64` resume la FSP como “sin restart”, pero
  `FotR/characters/rings/horses3d.py:691-699` escribe `restart=False` **y**
  `restart file name`; el diario en `:123-128` explica que HORSES3D exige esa clave.
- **Propuesta:** corregir esas dos frases del plan; distinguir “sin reinicio activo” de
  “sin clave de reinicio”.

### HS-03 — Regla `g` duplicada sin prueba de ida y vuelta

- **Evidencia:** la regex está duplicada en `FotR/characters/rings/horses3d.py:40` y
  `FotR/characters/readers/horses3d.py:188`. El plan afirma en `:38` que un test de ida y
  vuelta la garantiza, pero `FotR/tests/test_rings_horses3d.py:341-356` solo usa el nombre
  canónico `wt_cgns_g2_mesh.h5`; los casos `_v2g2`/tokens múltiples se prueban solo en el
  lector (`FotR/tests/test_horses3d.py:336-345`).
- **Propuesta:** añadir un test parametrizado que pase por ring y reader los mismos nombres
  límite, o extraer la regex a un único helper compartido.

### HS-04 — Cadena de órdenes sin dependencia tras un orden “ya empezado”

- **Evidencia:** `FotR/characters/rings/horses3d.py:946-971,975-988` pone `prev_job=None`
  cuando existe `.hsol` o `.residuals`; el docstring `:846-849` y el test
  `FotR/tests/test_rings_horses3d.py:476-491` codifican ese comportamiento. El plan también
  lo describe en `:189-191`, así que **no es una contradicción documental**.
- **Riesgo:** un `.hsol` presente no garantiza que el reinicio del orden anterior esté
  completo y válido; el siguiente `p` se envía sin `DEPENDENCY` y puede fallar o leer un
  fichero a medio escribir.
- **Propuesta:** decidir si “ya empezado” debe significar solo “finished” para encadenar, o
  si se mantiene el salto pero se valida el reinicio antes de lanzar el siguiente orden.
  Documentar la decisión en el plan.

### HS-05 — `Ngeo` como respaldo, no solo como dato informativo

- **Evidencia:** el plan en `:76-84` presenta el atributo `Ngeo` como respaldo y no incluye
  el desajuste nombre/`Ngeo` entre los errores. El reader
  `FotR/characters/readers/horses3d.py:647-662` lanza `ValueError` en modo estricto (que es
  el default, `:196`); `doc/HANDOFF.md:404-405` sí lo refleja.
- **Propuesta:** añadir ese caso a la lista de errores del plan y citar el test
  `FotR/tests/test_horses3d.py:358-364`.

### HS-06 — Reutilización silenciosa de `.tec` por `mtime`

- **Evidencia:** `FotR/solver_tools/horses2plt.py:246-251` devuelve `True` si el `.tec` existe
  y es más reciente que el `.hsol`; `FotR/characters/sets/horses3d.py:234-236` marca entonces
  `skipped/already converted`. La decisión de salto no consulta `variables`,
  `stats_variables`, `dimensionless`, gradientes ni la versión del control generado en
  `:248-251`.
- **Impacto:** cambiar las opciones sin `overwrite=True` puede dejar un `.tec` con el
  contenido anterior, sin aviso.
- **Propuesta:** guardar una huella de las opciones en el control/nombre del `.tec`, o
  incluir esas opciones en la decisión de `is_converted`.

### HS-07 — Script de conversión no versionado

- **Evidencia:** `doc/PLAN_HORSES3D.md:201-216` da por hecho
  `utils_H3d/POST/horses2plt_batch.slurm`. `git ls-files` no contiene ese fichero, y
  `FotR/characters/sets/horses3d.py:183-186` lanza `FileNotFoundError` si `script` no existe.
  El test real `FotR/tests/test_horses2plt.py:30` depende de una ruta externa
  `/home/m.jaraiz/Documentos/utils_H3d/POST/...`.
- **Impacto:** el flujo documentado no es reproducible desde un checkout limpio y el
  contrato `OK/ERROR <path>` usado en `:300-315` no se puede verificar en el repo.
- **Propuesta:** versionar el script (o un equivalente mínimo) o cambiar el plan para dejar
  explícito que `script` es una dependencia externa con contrato documentado.

### HS-08 — Rutas externas y entorno

- **Evidencia:** `examples/horses3d/marino2024_tecplot.py:5-7`, `FotR/tests/test_horses2plt.py:30`
  y `doc/HANDOFF.md:12` usan `/home/m.jaraiz/...`, que no existe en este workspace. El entorno
  real con `h5py`, `meshio`, `pyvista`, `vtk` y `pytest` es
  `/home/migueljaraiz/anaconda3/envs/envkan_nvidia/bin/python`.
- **Propuesta:** sustituir rutas absolutas por variables de entorno o rutas relativas, y
  actualizar el comando de tests del handoff.

### HS-09 — `HORSES3D` sí tiene `sets`

- **Evidencia:** `FotR/characters/sets/__init__.py:33` registra
  `'HORSES3D': '.horses3d:Horses3DSets'`. Sin embargo `README.md:80`,
  `examples/README.md:15`, `doc/HANDOFF.md:98` y `:112` dicen lo contrario. Este plan
  `:208-212` y `doc/frodo_sam/docs.json` ya reflejan la realidad.
- **Propuesta:** alinear README, examples y HANDOFF con el registry; no requiere cambiar código.

### HS-10 — Cifras y estado narrativo

- **Evidencia:** `doc/PLAN_HORSES3D.md:224` dice 492 tests; la recogida actual da 551
  (de ellos, 35 en `test_rings_horses3d.py`, 43 en `test_horses3d.py`, 27 en
  `test_horses3d_hsol.py` y 11 en `test_horses2plt.py`). `doc/HANDOFF.md:9` dice 542 passed.
  Además `:116` dice que falta decidir la física del caso real, mientras el diario `:188-199`
  ya registra disco actuador, recursos y lanzamiento p2→p5.
- **Propuesta:** unificar la cifra tras ejecutar la suite con `-rs` en el entorno correcto y
  precisar que lo pendiente es el modelo de rotor/IBM/condiciones iniciales, no la decisión
  de usar actuador.

### Revisado sin incidencias confirmadas

- Escalera `p` ascendente y `p >= g` en generación (`FotR/characters/rings/horses3d.py:425-436`).
- Criterio FSP por desviación de la solución, con residuos informativos
  (`FotR/characters/rings/horses3d.py:1030-1119`).
- Detección `pending`/`started`/`finished` por solución
  (`FotR/characters/readers/horses3d.py:1664-1697`). Un caso sin soluciones queda
  `pending`, no `finished` (`:1442-1448`): se descarta el falso positivo contrario.
- Nombres `<tag>_g<g>p<p>` y puente `cases_metadata.json` coherentes con el plan.
