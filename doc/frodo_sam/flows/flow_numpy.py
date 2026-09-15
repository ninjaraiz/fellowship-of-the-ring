"""Flujo NUMPY (re-ejecutable). Fuente estudiada: CODA2numpy2.ipynb.

Redactado desde cero para esta documentacion. El formato NUMPY
reutiliza el layout CODA: se exporta con ``sets.save_to_npy`` y se
recarga con ``format='NUMPY'``. Datos locales: Cylinder_PINNS.
Solo se documenta lo que este script obtiene de verdad al correr.
"""

import os
import sys

sys.path.append("/home/m.jaraiz/repos/pyLowOrder/")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/airbus/CETACEO_cp_interp/DATA/Cylinder_PINNS"
SCRATCH = "/home/m.jaraiz/.cache/frodo_doc"
NPY_PATH = os.path.join(SCRATCH, "frodo_doc_codalike.npy")

lines = []


def log(text=""):
    lines.append(text)
    print(text)


def run():
    from FotR import FRODO

    db = FRODO(root_dir=ROOT, format="CODA", initial_parse=True)
    db.extract_inputs(id_groups=(3,), cases_idx=[0, 1])
    db.extract_outputs(
        stage=1, id_groups=(3,), cases_idx=[0, 1],
        var_name_excluded=["GlobalNumber", "CADGroupID"],
    )
    db.sets.save_to_npy(stage=1, id_group="3", filepath=NPY_PATH)
    log("## Flujo NUMPY")
    log("")
    log("ES: exportado CODA→npy y recarga como formato NUMPY.")
    log("EN: CODA→npy export reloaded as NUMPY format.")
    log("")
    log("```")
    log(f"npy: {NPY_PATH} ({os.path.getsize(NPY_PATH)} bytes)")
    log("```")

    db2 = FRODO(root_dir=SCRATCH, format="NUMPY",
                initial_parse=True, file="frodo_doc_codalike.npy")
    log("")
    log("ES: grupos detectados en el .npy:")
    log("EN: groups found in the .npy:")
    log("```")
    log(f"{sorted(db2.reader.group_index)}")
    log("```")

    key = next(iter(db2.reader.group_index))
    gid = key.replace("CADGroup_", "")
    # El .npy viaja sin metadata: se restituyen las design_vars de origen.
    db2.metadata["design_vars"] = list(db.metadata["design_vars"])
    db2.extract_inputs(id_groups=gid, cases_idx=[0, 1])
    db2.extract_outputs(stage=1, id_groups=gid)
    grp = db2.data_dict[key]
    log("")
    log("ES: malla y campos recargados:")
    log("EN: reloaded mesh and fields:")
    log("```")
    log(f"Coord: {grp['Coord'].shape}  FlCc: {grp['FlCc'].shape}  "
        f"Vars['1']: {sorted(grp['Vars']['1'])}")
    log("```")

    res = db2.sets.create_jset(stage="1", id_group=gid)
    log("")
    log("ES: tensor conjunto del formato NUMPY:")
    log("EN: NUMPY joint tensor:")
    log("```")
    log(f"tensor: {tuple(res['tensor'].shape)}  info: {res['info']}")
    log("```")

    with open(os.path.join(HERE, "flow_numpy.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("written flow_numpy.md")


if __name__ == "__main__":
    run()
