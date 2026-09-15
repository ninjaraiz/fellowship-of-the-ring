"""Flujo HORSES3D, parcial (re-ejecutable). Fuente: esfera 3D LES.

Redactado desde cero para esta documentacion. Cubre parseo,
``extract_inputs`` (malla MESH/*.h5 prioritaria, FlCc solo design_vars),
``extract_outputs`` (cabecera .hsol + lazy, p unico) y residuals.
Sets/stats de este formato estan pendientes: no aparecen aqui.
Datos locales: my_ESPHERE_1 (caso valido M_0.3_re_200, p=2).
Solo se documenta lo que este script obtiene de verdad al correr.
"""

import os
import sys

sys.path.append("/home/m.jaraiz/repos/pyLowOrder/")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/m.jaraiz/Documentos/COMAC/my_ESPHERE_1"

lines = []


def log(text=""):
    lines.append(text)
    print(text)


def run():
    from FotR import FRODO

    db = FRODO(root_dir=ROOT, format="HORSES3D", strict=False)
    log("## Flujo HORSES3D (parcial)")
    log("")
    log("ES: LES con malla por caso (grado g en el nombre) y soluciones por p.")
    log("EN: LES with per-case mesh (grade g in the name) and per-p solutions.")
    log("")
    log("```")
    log(f"casos: {len(db.df_state)}  "
        f"design_vars: {db.reader.metadata['design_vars']}")
    log(f"{db.df_state[['case', 'p_available', 'g']].to_string(index=False)}")
    log("```")

    db.extract_inputs(p=2, cases_idx=[0])
    key = next(k for k in db.data_dict if k.startswith("CADGroup_"))
    grp = db.data_dict[key]
    log("")
    log("ES: `extract_inputs` agrupa por malla/g; FlCc solo design_vars:")
    log("EN: `extract_inputs` groups by mesh/g; FlCc holds design_vars only:")
    log("```")
    log(f"grupo: {key}")
    log(f"Coord: {grp['Coord'].shape}  FlCc: {grp['FlCc'].tolist()}")
    log("```")

    db.extract_outputs(p=2, cases_idx=[0])
    log("")
    log("ES: `extract_outputs` guarda descriptores lazy (sin leer 56MB):")
    log("EN: `extract_outputs` stores lazy descriptors (no 56MB read):")
    log("```")
    log(f"Vars['2']: {sorted(grp['Vars']['2'])}")
    log(f"{grp['Vars']['2']['rho'][0]}")
    log("```")

    df = db.residuals.get_df_residuals_from_case(case_idx=0, p=2)
    log("")
    log("ES: historial de residuos del caso:")
    log("EN: case residual history:")
    log("```")
    log(f"filas: {len(df)}  columnas: {list(df.columns)[:5]}")
    log("```")

    log("")
    log("ES: pendiente — sets/stats HORSES3D (sin `create_jset` aun).")
    log("EN: pending — HORSES3D sets/stats (no `create_jset` yet).")

    with open(os.path.join(HERE, "flow_horses3d.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("written flow_horses3d.md")


if __name__ == "__main__":
    run()
