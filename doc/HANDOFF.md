# HANDOFF — Fellowship of the Ring

> Lee esto primero. Para el detalle (arquitectura, contratos de datos, bugs
> localizados, cobertura de tests) → [`doc/frodo_sam/ANALISIS.md`](frodo_sam/ANALISIS.md).
> No relean el repo entero ni el historial anterior.

## 1. Estado — `develop` @ `50c0813`, **con trabajo sin commitear**

Suite: **207 passed** (el `156` de un handoff anterior estaba obsoleto).

```bash
/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q
```

Sin commitear: los arreglos de CODA_SINGLE (14 bugs) y el `sets`/`stats` propios del
formato. `git status --short` lo lista; el detalle está en los dos documentos citados
en §4.

Último commit (`50c0813`): refactor de `plot_all_final_residuals` (rama 1-D extraída a
`_plot_final_residuals_1d`, una figura por stage), `CODASets.plot_wall_integrals` nuevo,
sus dos ficheros de tests, `docs.json` regenerado y esta documentación.

Trabajo previo ya commiteado: limpieza conservadora de `frodo.py`/`sam.py`/los cuatro
subpaquetes (`print`→`logging`, type hints, guards, bugs de plot 1-D, `crop`, orientación
`(ncases,npts)` en `NUMPYReader`, `eval`→parser AST en `compute_param`); esquema `rings/`
(`BaseRing`, `CODARing`, `CODASingleRing`) con `gandalf.py` como fachada; reader
`CODA_SINGLE`; GCI TIFON real en `GCI/aoa_*/` (generado con `examples/gci_codasingle.py`
— **no ejecutar**: lanza `sbatch`); docs en `doc/frodo_sam/` (regenerar con
`python3 doc/frodo_sam/generate_docs.py`).

Aviso sobre los tests: los ficheros usan **stubs mutuamente incompatibles** (para no
arrastrar pyvista/torch/pyLOM) y el aislamiento depende del orden de importación. Si se
añade un test que importe el `FotR` real antes que los demás, revisar ese orden.

## 2. Decisiones vigentes que no están en el código

Verificadas leyendo el código en esta sesión:

* `FlCc` = solo `design_vars` del usuario, nunca params del solver
  (`readers/coda.py:842`, `readers/horses3d.py:983`).
* HORSES3D: `p` único exigido en `extract_outputs`, `.hsol` header+lazy
  (`HorsesLazyField`), `MESH/*.h5` prioritario — nunca bmesh/hmesh/pmesh/cgns.
* `GANDALF.Backpack` NO se delega: solo existe `BaseRing.Backpack`
  (`gandalf.py:49`, `_NON_DELEGATED`).
* En CODA_SINGLE, `mesh` puede ser design var (factor numérico); la identidad de malla
  vive en `sim_metadata[*]['mesh_info']` y `df_state['mesh_file']`, nunca en la clave
  `mesh`, que colisionaría (`readers/coda_single.py:137-140`).
* `data_dict['CADGroup_X']` en CODA_SINGLE guarda geometría y `Vars` como **listas por
  caso** (los `npts` difieren).
* Sin tocar a propósito: LEGOLAS (pendiente de reestructuración), columnas 2-D
  `['x','y']` vs `['x','z']` en `PYLOMSets`, orden de args `(subset,cases_idx)` vs
  `(cases_idx,subset)` entre readers.

Heredadas del handoff anterior, **no verificables desde aquí**:

* NUMPYFILE obsoleto (sin flujo de docs, sin trabajo nuevo). El código sigue registrado
  y funcional.
* `run_sst_v4.py` (fuera del repo): solo se tocó `alt = ALT_PLACEHOLDER`. No tocar nada
  más de ese fichero.

> El §5 del handoff anterior listaba notebooks sin seguimiento «que no son nuestros».
> Ya no aplica: `examples/gandalf/codasingle.ipynb` está trackeado y
> `examples/CylinderPINNS_database_CODA_NUMPY/gandalf_coda.ipynb` no existe.

## 3. Mapa

| pieza | dónde | qué es |
|---|---|---|
| `FRODO` | `FotR/characters/frodo.py` | coordinador: monta el cuarteto desde los registries, delega por `__getattr__`, y `merge_datasets` |
| `SAM` | `FotR/characters/sam.py` | namespace estático: `Gardener`, `HDF5reader`, `Backpack`, `Weapons`, `DifferentialOperators`, `DictVisualizer` |
| readers | `characters/readers/` | CODA · CODA_SINGLE · NUMPY · NUMPYFILE · PYLOM · HORSES3D |
| sets | `characters/sets/` | CODA (lo usa también NUMPY) · **CODA_SINGLE** · NUMPYFILE · PYLOM |
| stats | `characters/stats/` | CODA (lo usa también NUMPY) · **CODA_SINGLE** |
| residuals | `characters/residuals/` | CODA (lo reusa CODA_SINGLE) · HORSES3D |
| generadores | `characters/rings/` + `gandalf.py` | DoE, SLURM, `cases_metadata.json` |
| plots | `characters/legolas.py` | **único consumidor externo de FRODO** |

## 4. Trampas conocidas

Lo que hace perder tiempo nada más empezar (detalle en el ANÁLISIS):

* `db.<lo-que-sea>` puede venir de `sets`, `reader`, `residuals` o `stats`, en ese orden,
  y **la misma llamada tiene firma distinta según el formato** → ANÁLISIS §1 y §7.
* `db.sets` / `db.stats` son `None` en HORSES3D; `db.residuals` es `None` en NUMPY,
  PYLOM y NUMPYFILE → matriz completa en ANÁLISIS §2. Ojo además a B12: **sin pyLOM
  instalado, CODA y NUMPY también pierden su Sets en silencio**, porque
  `sets/coda.py` importa pyLOM en la cabecera y el registry se traga el
  `ModuleNotFoundError`.
* Hay **dos espacios de índices de caso** (global sobre `df_cases` vs local sobre el
  `FlCc` ya extraído) y los subsets solo viven en el global → ANÁLISIS §3.4.
* **SAM no tiene ni un test.** De FRODO solo se testea `merge_datasets`.
  `stats/coda.py` (2062 líneas) está a cero → ANÁLISIS §6.
* Dos examples están rotos: `basic.ipynb` llama `add_to_data_dict(key_location=...)`
  (es `array_name`) y tres notebooks de `examples/gandalf/` llaman `GANDALF.Backpack.*`
  → ANÁLISIS §5.5.
* Los dos [ALTO] de `readers/coda.py` que listaba el ANÁLISIS §4
  (`find_files(file_end=...)` y el caso fallido que quedaba con `FlCc` a ceros) **ya
  están arreglados**; el §4 del ANÁLISIS describe el defecto original, no el estado.
* **CODA_SINGLE está completo**: 14 bugs auditados y arreglados (salvo B12, de
  entorno) más `CODASingleSets`/`CODASingleStats` propios. Catálogo en
  [`doc/frodo_sam/BUGS_CODA_SINGLE.md`](frodo_sam/BUGS_CODA_SINGLE.md); diseño de
  sets en [`doc/frodo_sam/DISENO_SETS_CODA_SINGLE.md`](frodo_sam/DISENO_SETS_CODA_SINGLE.md).
  Lo que **no** se tocó: la misma conectividad errónea sigue en la ruta de CODA
  (`readers/coda.py:797`) y LEGOLAS sigue sin soportar este formato (despacha por
  `format == "CODA"` exacto); ninguna de las dos entraba en el encargo.

## 5. Siguiente paso previsto

CODA_SINGLE está cerrado (lector, sets, stats). Lo que queda pendiente y ya está
identificado:

* **LEGOLAS** — es lo último del repo según el plan del usuario. Hoy despacha por
  `format == "CODA"` exacto (`legolas.py:98,192,214`), así que CODA_SINGLE cae en la
  rama NUMPYFILE y muere con «No coordinates found for plotting». Ampliar el despacho
  no basta: `Coord` es una lista y `_get_coda_variable` hace `arr.ndim`.
* **La conectividad de CODA** (`readers/coda.py:797`) sigue con el defecto que se
  arregló en la ruta CODA_SINGLE. El arreglo está disponible en
  `SAM.Backpack.cell_connectivity_in_order`.
* **B12**: declarar pyLOM como dependencia, o dejar de importarlo en la cabecera de
  `sets/coda.py`.

## 6. Arranque mínimo

```bash
git log --oneline -8 && git status --short
/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q
```
