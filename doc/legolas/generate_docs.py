"""Generador de referencia API para la documentacion de LEGOLAS.

Misma maquinaria que ``doc/gandalf/generate_docs.py`` (de la que se copio el
parser de docstrings, numpydoc y estilo Google): lee el codigo con ``ast``,
sin importar el paquete, y vuelca ``docs.json``. Cada visor de ``doc/`` es
autocontenido, asi que una mejora del parser hay que portarla a mano.

Lo propio de LEGOLAS:

* **clases anidadas por alias**: ``LEGOLAS.Census = Census`` es una
  asignacion en el cuerpo de la clase, no un ``class`` anidado. Se
  resuelven siguiendo los ``from .modulo import Clase`` del paquete;
* **funciones de modulo** (``ordering``, ``topology``, lectores): se emiten
  como tarjetas de "modulo" con sus funciones;
* **formatos**: la lista sale de ``FORMATS`` en ``legolas/io/__init__.py``
  y la descripcion de cada uno es el docstring de su modulo lector. Nada de
  eso se escribe a mano.

Uso::

    python3 doc/legolas/generate_docs.py
"""

import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PKG = "FotR/characters/legolas"

#: Modulos cuyas funciones de nivel superior se documentan (tarjeta de modulo).
FUNCTION_MODULES = [
    f"{PKG}/io/__init__.py",
    f"{PKG}/ordering.py",
    f"{PKG}/topology.py",
    f"{PKG}/scene.py",
    f"{PKG}/bcfiles.py",
    f"{PKG}/quality.py",
]

#: Clases de nivel superior que se documentan, por fichero (None = todas).
CLASS_TARGETS = [
    (f"{PKG}/__init__.py", ["LEGOLAS"]),
    (f"{PKG}/mesh.py", None),
    (f"{PKG}/viewer.py", ["ViewerServer", "ViewerHandle"]),
]

# Secciones numpydoc que se parsean como lista de entradas "nombre : tipo".
DEF_SECTIONS = {
    "Parameters": "params",
    "Args": "params",
    "Arguments": "params",
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


_GOOGLE_HEAD = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Yields|Raises|Warns|Attributes|"
    r"Note|Notes|Example|Examples|Warning|Warnings|See Also):\s*$")
_GOOGLE_ENTRY = re.compile(r"^(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


def _google_split(lines):
    """Split a Google-style docstring: (lead, [(section, body_lines)]).

    Google sections are a bare ``Args:`` / ``Returns:`` line followed by an
    indented block, with no row of dashes underneath.
    """
    heads = [(i, _GOOGLE_HEAD.match(ln.strip()).group(1))
             for i, ln in enumerate(lines) if _GOOGLE_HEAD.match(ln.strip())]
    if not heads:
        return lines, []
    lead = lines[: heads[0][0]]
    sections = []
    for n, (i, title) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        sections.append((title, lines[i + 1 : end]))
    return lead, sections


def _parse_google_entries(body, type_only: bool):
    """Parse a Google ``name (type): description`` block."""
    body = _dedent(body)
    if not any(":" in ln for ln in body):
        text = " ".join(x.strip() for x in body if x.strip()).strip()
        return [{"name": "", "type": "", "desc": text}] if text else []

    entries, current = [], None
    for ln in body:
        m = _GOOGLE_ENTRY.match(ln.strip()) if ln.strip() else None
        starts = bool(m) and not ln.startswith((" ", "\t"))
        if starts:
            if current is not None:
                entries.append(current)
            name, typ, desc = m.group(1), m.group(2) or "", m.group(3)
            if type_only and not typ:
                name, typ = "", name
            current = {"name": name, "type": typ.strip(), "desc": [desc]}
        elif current is not None and ln.strip():
            current["desc"].append(ln.strip())
    if current is not None:
        entries.append(current)
    for e in entries:
        e["desc"] = " ".join(x for x in e["desc"] if x).strip()
    return entries


def _doc_info(doc: str) -> dict:
    """Parse a docstring (numpydoc, or Google style) into its sections."""
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
    google = False
    if not sections:                       # los rings usan estilo Google
        lead, sections = _google_split(lines)
        google = bool(sections)

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
            parse = _parse_google_entries if google else _parse_entries
            info[key].extend(parse(body, title in TYPE_ONLY))
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


#: Class-body assignments that bind a descriptor rather than a ``def``.
#: ``some_light = light.shortcut()`` is a real, documented entry point;
#: without this it would vanish from the reference, because the AST sees
#: an assignment and not a function.
_DESCRIPTOR_FACTORIES = {
    'shortcut': {
        'signature': 'some_light(target: str = None, *, show: bool = True) -> str',
        'returns': 'str',
        'summary': ("Shortcut to Earendil's Light: where a name lives and its "
                    "real signature."),
        'description': ("Bound to the instance when reached through one, so a "
                        "live object resolves its own delegation chain. See "
                        "``EarendilsLight.help``."),
    },
}


def _descriptor_methods(classdef: ast.ClassDef, class_id: str) -> list:
    """Emit entries for ``name = <expr>.<factory>()`` class attributes."""
    out = []
    for node in classdef.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        func = value.func
        if not isinstance(func, ast.Attribute):
            continue
        spec = _DESCRIPTOR_FACTORIES.get(func.attr)
        if spec is None:
            continue
        doc = _doc_info(None)
        doc.update(summary=spec['summary'], description=spec['description'],
                   has_doc=True)
        out.append({
            "id": f"{class_id}.{target.id}",
            "class": class_id,
            "name": target.id,
            "signature": spec['signature'],
            "returns": spec['returns'],
            "kind": "descriptor",
            "is_private": target.id.startswith('_'),
            "doc": doc,
        })
    return out


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
    return out + _descriptor_methods(classdef, class_id)


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


def _parse(relpath: str):
    path = os.path.join(REPO, relpath)
    with open(path, "r", encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _module_of(relpath: str, level: int, module: str) -> str:
    """Resolve a relative ``from .x import Y`` to a file path in the repo."""
    base = os.path.dirname(relpath)
    for _ in range(level - 1):
        base = os.path.dirname(base)
    target = os.path.join(base, *module.split(".")) if module else base
    if os.path.isdir(os.path.join(REPO, target)):
        return os.path.join(target, "__init__.py")
    return target + ".py"


def _imports(tree, relpath: str) -> dict:
    """{name: file} for the relative imports of a module."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            f = _module_of(relpath, node.level, node.module or "")
            for alias in node.names:
                out[alias.asname or alias.name] = f
    return out


def _class_cards(relpath: str, wanted=None) -> list:
    tree = _parse(relpath)
    cards = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if wanted is not None and node.name not in wanted:
                continue
            cards.extend(_collect_classes(node, relpath))
    return cards


def _nested_aliases(relpath: str, owner: str) -> list:
    """``Owner.Name = Name`` bindings resolved to their defining class."""
    tree = _parse(relpath)
    imports = _imports(tree, relpath)
    out = []
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == owner):
            continue
        for item in node.body:
            if (isinstance(item, ast.Assign) and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and isinstance(item.value, ast.Name)
                    and item.value.id in imports):
                src = imports[item.value.id]
                card = next((c for c in _class_cards(src, [item.value.id])
                             if c["id"] == item.value.id), None)
                if card is None:
                    continue
                card = dict(card, id=f"{owner}.{item.targets[0].id}",
                            defined_as=item.value.id)
                for m in card["methods"]:
                    m["class"] = card["id"]
                    m["id"] = f"{card['id']}.{m['name']}"
                out.append(card)
    return out


def _function_card(relpath: str) -> dict:
    """A pseudo-class card holding the top-level functions of a module."""
    tree = _parse(relpath)
    mod_id = relpath[len("FotR/characters/"):-3].replace("/", ".")
    mod_id = mod_id.replace(".__init__", "")
    fake = ast.ClassDef(name=mod_id, bases=[], keywords=[], body=[
        n for n in tree.body if isinstance(n, (ast.FunctionDef,
                                                ast.AsyncFunctionDef))
    ], decorator_list=[])
    methods = _methods(fake, mod_id)
    for m in methods:
        m["kind"] = "function"
    return {
        "id": mod_id,
        "file": relpath,
        "parent": None,
        "bases": [],
        "doc": _doc_info(ast.get_docstring(tree)),
        "methods": methods,
        "module": True,
    }


def _formats() -> list:
    """One entry per reader: name, module, module docstring, read()."""
    init = f"{PKG}/io/__init__.py"
    tree = _parse(init)
    formats = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "FORMATS"
                and isinstance(node.value, ast.Dict)):
            for k, v in zip(node.value.keys, node.value.values):
                formats[k.value] = v.value
    out = []
    for name, mod in formats.items():
        rel = _module_of(init, 1, mod.lstrip("."))
        card = _function_card(rel)
        out.append({"name": name, "file": rel, "card": card})
    return out


def main() -> None:
    classes = []
    for rel, wanted in CLASS_TARGETS:
        classes.extend(_class_cards(rel, wanted))
    nested = _nested_aliases(f"{PKG}/__init__.py", "LEGOLAS")
    modules = [_function_card(rel) for rel in FUNCTION_MODULES]
    formats = _formats()
    payload = {"classes": classes, "nested": nested, "modules": modules,
               "formats": formats}
    with open(os.path.join(HERE, "docs.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    cards = classes + nested + modules + [f["card"] for f in formats]
    print("clases: " + ", ".join(c["id"] for c in classes))
    print("anidadas: " + ", ".join(c["id"] for c in nested))
    print("formatos: " + ", ".join(f["name"] for f in formats))
    for c in cards:
        missing = [m["name"] for m in c["methods"]
                   if not m["doc"]["has_doc"] and not m["is_private"]]
        flag = "" if c["doc"]["has_doc"] else "  [sin docstring]"
        print(f"- {c['id']}: {len(c['methods'])} entradas, publicas sin doc: "
              f"{missing if missing else 'ninguna'}{flag}")


if __name__ == "__main__":
    main()
