"""Generador de referencia API para la documentacion FRODO + SAM.

Lee el codigo fuente con ``ast`` (sin importar el paquete, para no
arrastrar dependencias pesadas) y vuelca ``docs.json`` con clases,
metodos, firmas y presencia de secciones numpydoc en cada docstring.
La narrativa ES/EN vive en ``i18n.json`` y en el HTML; este script no
traduce nada.
"""

import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# (fichero relativo al repo, clases de interes; None = todas las de nivel superior)
TARGETS = [
    ("FotR/characters/frodo.py", ["FRODO"]),
    ("FotR/characters/gandalf.py", ["GANDALF"]),
    ("FotR/characters/rings/base.py", ["BaseRing"]),
    ("FotR/characters/rings/coda.py", ["CODARing"]),
    ("FotR/characters/rings/coda_single.py", ["CODASingleRing"]),
    ("FotR/characters/sam.py", ["SAM"]),
    ("FotR/characters/readers/base.py", None),
    ("FotR/characters/readers/coda.py", None),
    ("FotR/characters/readers/horses3d.py", None),
    ("FotR/characters/readers/numpy.py", None),
    ("FotR/characters/readers/numpy_file.py", None),
    ("FotR/characters/readers/pylom.py", None),
    ("FotR/characters/sets/base.py", None),
    ("FotR/characters/sets/coda.py", None),
    ("FotR/characters/sets/numpy_file.py", None),
    ("FotR/characters/sets/pylom.py", None),
    ("FotR/characters/stats/base.py", None),
    ("FotR/characters/stats/coda.py", None),
    ("FotR/characters/residuals/base.py", None),
    ("FotR/characters/residuals/coda.py", None),
    ("FotR/characters/residuals/horses3d.py", None),
]

SECTIONS = ("Parameters", "Returns", "Raises", "Examples", "Attributes")


def _signature(fn: ast.FunctionDef) -> str:
    parts = []
    for arg in fn.args.args:
        if arg.arg in ("self", "cls"):
            continue
        parts.append(arg.arg)
    if fn.args.vararg:
        parts.append("*" + fn.args.vararg.arg)
    for arg in fn.args.kwonlyargs:
        parts.append(arg.arg)
    if fn.args.kwarg:
        parts.append("**" + fn.args.kwarg.arg)
    return f"{fn.name}({', '.join(parts)})"


def _doc_info(doc: str) -> dict:
    doc = doc or ""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    return {
        "summary": first,
        "sections": {s: (s in doc) for s in SECTIONS},
        "has_doc": bool(doc.strip()),
    }


def _decorators(fn: ast.FunctionDef) -> list:
    names = []
    for d in fn.decorator_list:
        if isinstance(d, ast.Name):
            names.append(d.id)
        elif isinstance(d, ast.Attribute):
            names.append(d.attr)
    return names


def _methods(classdef: ast.ClassDef, prefix: str = "") -> list:
    out = []
    for node in classdef.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name not in (
                "__init__",
            ):
                continue
            out.append(
                {
                    "id": f"{prefix}{classdef.name}.{node.name}",
                    "class": f"{prefix}{classdef.name}",
                    "name": node.name,
                    "signature": _signature(node),
                    "kind": next(
                        (
                            k
                            for k in ("staticmethod", "classmethod", "property")
                            if k in _decorators(node)
                        ),
                        "method",
                    ),
                    "doc": _doc_info(ast.get_docstring(node)),
                }
            )
        elif isinstance(node, ast.ClassDef):
            # Subclases anidadas (p. ej. SAM.Gardener, LEGOLAS no incluido).
            out.extend(_methods(node, prefix=f"{prefix}{classdef.name}."))
    return out


def parse_file(relpath: str, wanted) -> dict:
    path = os.path.join(REPO, relpath)
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if wanted is not None and node.name not in wanted:
                continue
            classes.append(
                {
                    "id": node.name,
                    "file": relpath,
                    "doc": _doc_info(ast.get_docstring(node)),
                    "methods": _methods(node),
                }
            )
    return {"file": relpath, "classes": classes}


def main() -> None:
    modules = [parse_file(rel, wanted) for rel, wanted in TARGETS]
    n_methods = sum(len(c["methods"]) for m in modules for c in m["classes"])
    n_nodoc = sum(
        1
        for m in modules
        for c in m["classes"]
        for md in c["methods"]
        if not md["doc"]["has_doc"]
    )
    payload = {"modules": modules}
    with open(os.path.join(HERE, "docs.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"modules: {len(modules)}  methods: {n_methods}  nodoc: {n_nodoc}")
    for m in modules:
        for c in m["classes"]:
            missing = [md["name"] for md in c["methods"] if not md["doc"]["has_doc"]]
            print(f"- {m['file']}::{c['id']}: {len(c['methods'])} metodos, "
                  f"sin doc: {missing if missing else 'ninguno'}")


if __name__ == "__main__":
    main()
