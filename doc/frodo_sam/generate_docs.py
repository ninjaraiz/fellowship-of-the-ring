"""Generador de referencia API para la documentacion FRODO + SAM.

Lee el codigo fuente con ``ast`` (sin importar el paquete, para no
arrastrar dependencias pesadas) y vuelca ``docs.json`` con clases,
metodos, firmas completas y el docstring numpydoc ya troceado en
secciones. La narrativa ES/EN vive en ``i18n.json`` y ``guides.json``;
este script no traduce nada.

Uso::

    python3 doc/frodo_sam/generate_docs.py

Esquema de ``docs.json``::

    {"modules": [{"file": str, "classes": [Class]}]}

    Class  = {id, file, parent, doc: Doc, methods: [Method]}
    Method = {id, class, name, signature, returns, kind, is_private, doc: Doc}
    Doc    = {summary, description, params, returns, raises, attributes,
              methods, text_sections, has_doc}

Las clases anidadas (``SAM.Gardener``, ``SAM.Backpack.pattern_pocket``…)
salen como entradas propias, con ``parent`` apuntando a la clase que las
contiene.
"""

import ast
import json
import os
import re

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
    ("FotR/characters/readers/coda_single.py", None),
    ("FotR/characters/readers/horses3d.py", None),
    ("FotR/characters/readers/numpy.py", None),
    ("FotR/characters/readers/numpy_file.py", None),
    ("FotR/characters/readers/pylom.py", None),
    ("FotR/characters/sets/base.py", None),
    ("FotR/characters/sets/coda.py", None),
    ("FotR/characters/sets/coda_single.py", None),
    ("FotR/characters/sets/numpy_file.py", None),
    ("FotR/characters/sets/pylom.py", None),
    ("FotR/characters/stats/base.py", None),
    ("FotR/characters/stats/coda.py", None),
    ("FotR/characters/stats/coda_single.py", None),
    ("FotR/characters/residuals/base.py", None),
    ("FotR/characters/residuals/coda.py", None),
    ("FotR/characters/residuals/horses3d.py", None),
]

# Secciones numpydoc que se parsean como lista de entradas "nombre : tipo".
DEF_SECTIONS = {
    "Parameters": "params",
    "Other Parameters": "params",
    "Returns": "returns",
    "Yields": "returns",
    "Raises": "raises",
    "Warns": "raises",
    "Attributes": "attributes",
    "Methods": "methods",
}

# Secciones que se conservan como texto crudo, en el orden en que aparecen.
TEXT_SECTIONS = (
    "Examples", "Notes", "See Also", "References", "Warnings",
    "Design principles",
)

# Secciones cuyas entradas no llevan nombre, solo tipo (Returns: "FRODO").
TYPE_ONLY = {"Returns", "Yields", "Raises", "Warns"}

# Dunders que SI forman parte del contrato publico y deben documentarse.
# ``FRODO.__getattr__`` es el caso que motiva la lista: es el mecanismo
# central de la clase y quedaba fuera de la referencia. Los puramente
# cosmeticos (``__str__``, ``__repr__``, ``__post_init__``) se omiten.
DUNDER_ALLOW = {
    "__init__", "__getattr__", "__getitem__", "__setitem__", "__call__",
    "__len__", "__iter__", "__contains__", "__enter__", "__exit__",
}

_DASHES = re.compile(r"^-{3,}$")
# Reglas decorativas: varios docstrings del repo subrayan su titulo con
# caracteres de dibujo (U+2500 y companyia), que no son guiones ASCII y por
# tanto no abren seccion numpydoc. Cuentan como fin de parrafo, no como texto.
_RULE = re.compile(r"^[-─━═=~_—]{3,}$")


# Los cuatro registries, para construir el mapa formato -> extension sin
# duplicarlo a mano: la fuente de verdad es el codigo.
REGISTRY_FILES = {
    "reader":    ("FotR/characters/readers/__init__.py",   "READER_REGISTRY"),
    "sets":      ("FotR/characters/sets/__init__.py",      "SETS_REGISTRY"),
    "stats":     ("FotR/characters/stats/__init__.py",     "STATS_REGISTRY"),
    "residuals": ("FotR/characters/residuals/__init__.py", "RESIDUALS_REGISTRY"),
}


def _registry_entries(relpath: str, regname: str) -> dict:
    """Read one registry __init__.py: {format: class name, or None}."""
    path = os.path.join(REPO, relpath)
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is None:
            continue
        for t in targets:
            # REG = {'Airfoil': None, ...}
            if (isinstance(t, ast.Name) and t.id == regname
                    and isinstance(value, ast.Dict)):
                for k, v in zip(value.keys, value.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        out[k.value] = v.id if isinstance(v, ast.Name) else None
            # REG['CODA'] = CODAReader
            elif (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                  and t.value.id == regname):
                key = t.slice
                if key.__class__.__name__ == "Index":   # Python < 3.9
                    key = key.value
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    out[key.value] = value.id if isinstance(value, ast.Name) else None
    return out


def _formats() -> dict:
    """Build {format: {reader, sets, stats, residuals}} from the registries.

    A format only counts as real when it has a reader: that is what FRODO
    requires, and it is what keeps the commented-out 'Airfoil' entries of
    the sets/residuals registries out of the documentation.
    """
    reg = {kind: _registry_entries(rel, name)
           for kind, (rel, name) in REGISTRY_FILES.items()}
    names = set()
    for d in reg.values():
        names |= set(d)
    out = {}
    for fmt in sorted(names):
        row = {kind: reg[kind].get(fmt) for kind in REGISTRY_FILES}
        if row["reader"]:
            out[fmt] = row
    return out


# ── Firmas ────────────────────────────────────────────────────────────────


def _unparse(node) -> str:
    """Render an AST expression back to source, or '' if not possible."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive, older grammars
        return "..."


def _arg(a: ast.arg, default=None) -> str:
    """Format one argument with its annotation and default value."""
    out = a.arg
    if a.annotation is not None:
        out += f": {_unparse(a.annotation)}"
    if default is not None:
        sep = " = " if a.annotation is not None else "="
        out += f"{sep}{_unparse(default)}"
    return out


def _signature(fn) -> str:
    """Build the full call signature, keeping annotations and defaults."""
    a = fn.args
    positional = list(getattr(a, "posonlyargs", [])) + list(a.args)
    # Los defaults cubren los ultimos N argumentos posicionales.
    pad = [None] * (len(positional) - len(a.defaults))
    defaults = pad + list(a.defaults)

    parts = []
    for arg, dflt in zip(positional, defaults):
        if arg.arg in ("self", "cls"):
            continue
        parts.append(_arg(arg, dflt))

    if getattr(a, "posonlyargs", None):
        # El marcador '/' va tras el ultimo posicional-only que hayamos escrito.
        kept = [x for x in a.posonlyargs if x.arg not in ("self", "cls")]
        if kept:
            parts.insert(len(kept), "/")

    if a.vararg is not None:
        parts.append("*" + _arg(a.vararg))
    elif a.kwonlyargs:
        parts.append("*")

    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(_arg(arg, dflt))

    if a.kwarg is not None:
        parts.append("**" + _arg(a.kwarg))

    return f"{fn.name}({', '.join(parts)})"


# ── Docstrings ────────────────────────────────────────────────────────────


def _split_sections(lines):
    """Split docstring lines into (lead, [(section_name, body_lines)])."""
    heads = []
    for i in range(len(lines) - 1):
        title = lines[i].strip()
        if not title or lines[i].startswith(" "):
            continue
        if _DASHES.match(lines[i + 1].strip()):
            heads.append((i, title))

    if not heads:
        return lines, []

    lead = lines[: heads[0][0]]
    sections = []
    for n, (i, title) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        sections.append((title, lines[i + 2 : end]))
    return lead, sections


def _dedent(lines):
    """Strip the common leading indentation from a block of lines."""
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return []
    pad = min(len(ln) - len(ln.lstrip()) for ln in body)
    return [ln[pad:] if ln.strip() else "" for ln in lines]


def _parse_entries(body, type_only: bool):
    """Parse a numpydoc definition list into {name, type, desc} records."""
    body = _dedent(body)
    entries = []
    current = None
    for ln in body:
        if ln.strip() and not ln.startswith((" ", "\t")):
            if current is not None:
                entries.append(current)
            head = ln.strip()
            name, _, typ = head.partition(" : ")
            if not _:
                name, typ = ("", head) if type_only else (head, "")
            current = {"name": name.strip(), "type": typ.strip(), "desc": []}
        elif current is not None:
            current["desc"].append(ln.strip())
    if current is not None:
        entries.append(current)

    for e in entries:
        e["desc"] = " ".join(x for x in e["desc"] if x).strip()
    return entries


def _doc_info(doc: str) -> dict:
    """Parse a numpydoc docstring into its summary, prose and sections."""
    info = {
        "summary": "",
        "description": "",
        "params": [],
        "returns": [],
        "raises": [],
        "attributes": [],
        "methods": [],
        "text_sections": [],
        "has_doc": bool((doc or "").strip()),
    }
    if not info["has_doc"]:
        return info

    lines = doc.strip("\n").rstrip().splitlines()
    lead, sections = _split_sections(lines)

    # El resumen es el PRIMER PARRAFO entero, unido en una sola linea: tomar
    # solo la primera linea fisica cortaba el 43 % de los resumenes del repo.
    lead = _dedent(lead)
    para, rest = [], []
    for n, ln in enumerate(lead):
        stripped = ln.strip()
        if (not stripped or _RULE.match(stripped)) and para:
            rest = lead[n + 1:]
            break
        if stripped and not _RULE.match(stripped):
            para.append(stripped)
    info["summary"] = " ".join(para).strip()
    info["description"] = "\n".join(rest).strip()

    for title, body in sections:
        if title in DEF_SECTIONS:
            key = DEF_SECTIONS[title]
            info[key].extend(_parse_entries(body, title in TYPE_ONLY))
        else:
            text = "\n".join(_dedent(body)).strip()
            if text:
                info["text_sections"].append({"title": title, "text": text})
    return info


# ── Recorrido del AST ─────────────────────────────────────────────────────


def _decorators(fn) -> list:
    names = []
    for d in fn.decorator_list:
        if isinstance(d, ast.Name):
            names.append(d.id)
        elif isinstance(d, ast.Attribute):
            names.append(d.attr)
    return names


def _methods(classdef: ast.ClassDef, class_id: str) -> list:
    """Collect the methods defined directly on *classdef*."""
    out = []
    for node in classdef.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") and node.name not in DUNDER_ALLOW:
            continue
        out.append(
            {
                "id": f"{class_id}.{node.name}",
                "class": class_id,
                "name": node.name,
                "signature": _signature(node),
                "returns": _unparse(node.returns),
                "kind": next(
                    (
                        k
                        for k in ("staticmethod", "classmethod", "property")
                        if k in _decorators(node)
                    ),
                    "method",
                ),
                "is_private": node.name.startswith("_")
                and node.name not in DUNDER_ALLOW,
                "doc": _doc_info(ast.get_docstring(node)),
            }
        )
    return out


def _collect_classes(classdef: ast.ClassDef, relpath: str, parent=None) -> list:
    """Emit a card for *classdef* and, recursively, for its nested classes."""
    class_id = f"{parent}.{classdef.name}" if parent else classdef.name
    cards = [
        {
            "id": class_id,
            "file": relpath,
            "parent": parent,
            "bases": [_unparse(b) for b in classdef.bases],
            "doc": _doc_info(ast.get_docstring(classdef)),
            "methods": _methods(classdef, class_id),
        }
    ]
    for node in classdef.body:
        if isinstance(node, ast.ClassDef):
            cards.extend(_collect_classes(node, relpath, parent=class_id))
    return cards


def parse_file(relpath: str, wanted) -> dict:
    path = os.path.join(REPO, relpath)
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if wanted is not None and node.name not in wanted:
                continue
            classes.extend(_collect_classes(node, relpath))
    return {"file": relpath, "classes": classes}


def main() -> None:
    modules = [parse_file(rel, wanted) for rel, wanted in TARGETS]
    all_classes = [c for m in modules for c in m["classes"]]
    all_methods = [md for c in all_classes for md in c["methods"]]
    n_nodoc = sum(1 for md in all_methods if not md["doc"]["has_doc"])
    n_cls_nodoc = sum(1 for c in all_classes if not c["doc"]["has_doc"])
    n_params = sum(1 for md in all_methods if md["doc"]["params"])

    formats = _formats()
    payload = {"formats": formats, "modules": modules}
    with open(os.path.join(HERE, "docs.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print("formatos: " + ", ".join(
        f"{f}[" + "".join(k[0] for k, v in row.items() if v) + "]"
        for f, row in formats.items()))
    print(
        f"modules: {len(modules)}  classes: {len(all_classes)}  "
        f"methods: {len(all_methods)}  nodoc: {n_nodoc}  "
        f"classes sin doc: {n_cls_nodoc}  con Parameters: {n_params}"
    )
    for m in modules:
        for c in m["classes"]:
            missing = [md["name"] for md in c["methods"] if not md["doc"]["has_doc"]]
            flag = "" if c["doc"]["has_doc"] else "  [clase sin docstring]"
            print(f"- {m['file']}::{c['id']}: {len(c['methods'])} metodos, "
                  f"sin doc: {missing if missing else 'ninguno'}{flag}")


if __name__ == "__main__":
    main()
