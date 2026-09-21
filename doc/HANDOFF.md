# HANDOFF — Fellowship of the Ring

> Lee esto primero. Para el detalle (arquitectura, contratos de datos, bugs
> localizados, cobertura de tests) → [`doc/frodo_sam/ANALISIS.md`](frodo_sam/ANALISIS.md).
> No relean el repo entero ni el historial anterior.

## 1. Estado — `develop` @ `663e3d1`, **con trabajo sin commitear**

Suite: **162 passed** (el `156` del handoff anterior estaba obsoleto).

```bash
/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q
```

Pendiente de commit en el árbol de trabajo:

| fichero | qué es |
|---|---|
| `FotR/characters/residuals/coda.py` | refactor de `plot_all_final_residuals`: rama 1-D extraída a `_plot_final_residuals_1d`, una figura por stage |
| `FotR/characters/sets/coda.py` | `plot_wall_integrals` nuevo (+216 líneas) |
| `FotR/tests/test_coda_residuals_plot.py` | tests de la rama 1-D |
| `FotR/tests/test_coda_wall_integrals.py` | **sin trackear** — 4 tests de `plot_wall_integrals` |
| `doc/frodo_sam/docs.json` | regenerado |
| `examples/TIFON_database_CODA/coda_single_GCI.ipynb` | — |
| `doc/HANDOFF.md`, `doc/frodo_sam/ANALISIS.md` | **sin trackear** |

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
| sets | `characters/sets/` | CODA (lo usa también NUMPY) · NUMPYFILE · PYLOM |
| stats | `characters/stats/` | solo CODA (lo usa también NUMPY) |
| residuals | `characters/residuals/` | CODA (lo reusa CODA_SINGLE) · HORSES3D |
| generadores | `characters/rings/` + `gandalf.py` | DoE, SLURM, `cases_metadata.json` |
| plots | `characters/legolas.py` | **único consumidor externo de FRODO** |

## 4. Trampas conocidas

Lo que hace perder tiempo nada más empezar (detalle en el ANÁLISIS):

* `db.<lo-que-sea>` puede venir de `sets`, `reader`, `residuals` o `stats`, en ese orden,
  y **la misma llamada tiene firma distinta según el formato** → ANÁLISIS §1 y §7.
* `db.sets` / `db.stats` son `None` en CODA_SINGLE y HORSES3D; `db.residuals` es `None`
  en NUMPY, PYLOM y NUMPYFILE → matriz completa en ANÁLISIS §2.
* Hay **dos espacios de índices de caso** (global sobre `df_cases` vs local sobre el
  `FlCc` ya extraído) y los subsets solo viven en el global → ANÁLISIS §3.4.
* **SAM no tiene ni un test.** De FRODO solo se testea `merge_datasets`.
  `stats/coda.py` (2062 líneas) está a cero → ANÁLISIS §6.
* Dos examples están rotos: `basic.ipynb` llama `add_to_data_dict(key_location=...)`
  (es `array_name`) y tres notebooks de `examples/gandalf/` llaman `GANDALF.Backpack.*`
  → ANÁLISIS §5.5.
* **Sin arreglar, documentados**: dos bugs [ALTO] en `readers/coda.py` —
  `find_files(file_end=...)` (deja muerto el fallback sin `cases_metadata.json` y
  `plot_integrals_from_case`) y un caso que falla en `extract_inputs` y queda como caso
  válido con `FlCc` a ceros → ANÁLISIS §4.

## 5. Siguiente paso previsto

`sets`/`stats` propios de CODA_SINGLE (`create_jset` con `npts` por caso, `compute_stats`
sobre listas) y helper de orden GCI/Richardson. El reader ya deja listos `case_order`,
`mesh_files`, `idx_sort` por caso y `active_cases_idx`; lo que hay que diseñar es el
ensamblado con `npts` variable, porque `SAM.Gardener.create_final_tensor` asume un
`tensor_ptos` único compartido y **no sirve tal cual** → ANÁLISIS §8.

Antes conviene decidir qué hacer con los dos [ALTO] del §4: el de `extract_inputs`
afecta a la fiabilidad de cualquier dataset construido con CODA.

## 6. Arranque mínimo

```bash
git log --oneline -8 && git status --short
/home/m.jaraiz/miniconda/envs/envkan_nvidia/bin/python -m pytest FotR/tests/ -q
```
