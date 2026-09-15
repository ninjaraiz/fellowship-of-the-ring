"""Flujo PYLOM (re-ejecutable). Fuente estudiada: TIFON_database_PYLOM.

Redactado desde cero para esta documentacion. El formato PYLOM lee un
Dataset pyLOM (.h5) con variables escalares por caso y fields por punto.
Datos locales: TIFON/CADGroup_3_stage_0.h5.
Solo se documenta lo que este script obtiene de verdad al correr.
"""

import os
import sys

sys.path.append("/home/m.jaraiz/repos/pyLowOrder/")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/airbus/CETACEO_cp_interp/DATA/TIFON"

lines = []


def log(text=""):
    lines.append(text)
    print(text)


def run():
    from FotR import FRODO

    db = FRODO(root_dir=ROOT, format="PYLOM", initial_parse=True,
               file="CADGroup_3_stage_0.h5")
    log("## Flujo PYLOM")
    log("")
    log("ES: Dataset pyLOM con variables por caso y fields por punto.")
    log("EN: pyLOM Dataset with per-case variables and per-point fields.")

    db.extract_inputs(
        keys_inputs={"ptos": "xyz", "aoa": "aoa", "mach": "mach"},
        keys_aux={},
    )
    db.extract_outputs(
        keys_outputs={"cp": "BoundaryValues_CoefPressure"}
    )
    log("")
    log("```")
    log(f"fields: {db.sets.field_names()}")
    log(f"variables: {db.sets.variable_names()}")
    log("```")
    xyz = db.sets.get_xyz()
    cp = db.sets.get_field("cp")
    log("")
    log("ES: entradas y campo extraidos:")
    log("EN: extracted inputs and field:")
    log("```")
    log(f"xyz: {xyz.shape}  cp: {cp.shape}")
    log(f"df_state: {db.df_state.shape}")
    log("```")

    res = db.sets.create_jset()
    log("")
    log("ES: tensor conjunto del formato PYLOM:")
    log("EN: PYLOM joint tensor:")
    log("```")
    log(f"tensor: {tuple(res['tensor'].shape)}  info: {res['info']}")
    log("```")

    with open(os.path.join(HERE, "flow_pylom.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("written flow_pylom.md")


if __name__ == "__main__":
    run()
