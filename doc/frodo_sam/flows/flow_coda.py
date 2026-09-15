"""Flujo CODA (re-ejecutable). Fuente estudiada: TIFON_database_CODA.

Redactado desde cero para esta documentacion. Datos locales:
Cylinder_PINNS (design_vars=[M], 2 stages, surface+volume VTU).
Solo se documenta lo que este script obtiene de verdad al correr.
"""

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.append("/home/m.jaraiz/repos/pyLowOrder/")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/airbus/CETACEO_cp_interp/DATA/Cylinder_PINNS"

lines = []


def log(text=""):
    lines.append(text)
    print(text)


def run():
    from FotR import FRODO

    buf = io.StringIO()
    with redirect_stdout(buf):
        db = FRODO(root_dir=ROOT, format="CODA", initial_parse=True)
    log("## Flujo CODA")
    log("")
    log("ES: parseo de la base Cylinder_PINNS (formato CODA, dos stages).")
    log("EN: parsing the Cylinder_PINNS base (CODA format, two stages).")
    log("")
    log("```")
    log(buf.getvalue().strip())
    log(f"casos: {len(db.df_state)}  design_vars: {db.metadata['design_vars']}")
    log("```")

    buf = io.StringIO()
    with redirect_stdout(buf):
        db.reader.print_available_cadgroup_ids(stage=1, vtu_type="surface")
    log("")
    log("ES: grupos CAD disponibles en superficie, stage 1:")
    log("EN: available surface CAD groups, stage 1:")
    log("```")
    log(buf.getvalue().strip())
    log("```")

    # Dos casos para que el flujo sea rapido.
    db.extract_inputs(id_groups=(3,), cases_idx=[0, 1])
    grp = db.data_dict["CADGroup_3"]
    log("")
    log("ES: `extract_inputs` con dos casos del grupo 3:")
    log("EN: `extract_inputs` with two cases of group 3:")
    log("```")
    log(f"Coord: {grp['Coord'].shape}  FlCc: {grp['FlCc'].tolist()}")
    log("```")

    db.extract_outputs(
        stage=1, id_groups=(3,), cases_idx=[0, 1],
        var_name_excluded=["GlobalNumber", "CADGroupID"],
    )
    var = next(iter(grp["Vars"]["1"]))
    log("")
    log("ES: `extract_outputs` del stage 1 (primera variable):")
    log("EN: `extract_outputs` of stage 1 (first variable):")
    log("```")
    log(f"{var}: {grp['Vars']['1'][var].shape}")
    log("```")

    df = db.residuals.get_all_final_residuals(
        stage=[1], only_finished=False, load_in_metadata=False
    )
    log("")
    log("ES: ultimos residuos agregados:")
    log("EN: aggregated final residuals:")
    log("```")
    log(f"filas: {len(df)}  columnas: {list(df.columns)[:6]}")
    log("```")

    res = db.sets.create_jset(stage="1", id_group="3")
    log("")
    log("ES: tensor conjunto via `sets.create_jset`:")
    log("EN: joint tensor via `sets.create_jset`:")
    log("```")
    log(f"tensor: {tuple(res['tensor'].shape)}  info: {res['info']}")
    log("```")

    with open(os.path.join(HERE, "flow_coda.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("written flow_coda.md")


if __name__ == "__main__":
    run()
