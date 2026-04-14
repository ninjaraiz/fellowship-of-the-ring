import inspect
import importlib

import dataclasses
from typing import get_type_hints

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax

class EarendilsLight:
    """
    🌟 Eärendil's Light 🌟
    Sistema de ayuda introspectiva para frameworks modulares.

    - Muestra docstrings
    - Inspecciona firmas, inputs y outputs
    - Soporta dataclasses
    - Salida enriquecida con rich
    """

    def __init__(self, main_module):
        """
        Args:
            main_module (module or str): Módulo principal del framework.
        """
        
        if isinstance(main_module, str):
            main_module = importlib.import_module(main_module)

        self.main_module = main_module
        self.console = Console()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def help(self, target: str = None, *, verbose: bool = False):
        """
        Muestra ayuda introspectiva de una clase, método o función.

        Ejemplos:
            light.help()
            light.help("CODAReader")
            light.help("CODAReader.extract_input")
            light.help("extract_input", verbose=True)

        Args:
            target (str): Nombre del objeto a inspeccionar.
            verbose (bool): Si True, lista métodos públicos de clases.
        """

        if target is None:
            self._print_overview()
            return

        parts = [p for p in target.split(".") if p]

        obj = self._try_resolve(parts)
        if obj is not None:
            self._print_object(obj, parts, verbose)
            return

        matches = self._search_by_name(parts[-1])
        if matches:
            for path, member in matches:
                self._print_object(member, path.split("."), verbose)
            return

        self.console.print(
            f"[bold red]❌ No se encontró '{target}' en "
            f"{getattr(self.main_module, '__name__', str(self.main_module))}[/]"
        )

    # ------------------------------------------------------------------
    # Resolución de objetos
    # ------------------------------------------------------------------

    def _try_resolve(self, parts):
        obj = self.main_module
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return None
        return obj

    def _search_by_name(self, name, max_depth=10):
        matches = []
        visited = set()

        def recurse(root, path, depth):
            if depth < 0:
                return

            try:
                obj_id = id(root)
                if obj_id in visited:
                    return
                visited.add(obj_id)
            except Exception:
                return

            try:
                for attr, member in inspect.getmembers(root):
                    if attr.startswith("_"):
                        continue

                    fullpath = path + [attr]

                    # MATCH: nombre exacto
                    if attr == name:
                        matches.append((".".join(fullpath), member))

                    # RECURSIÓN: solo en módulos o clases
                    if inspect.ismodule(member) or inspect.isclass(member):
                        recurse(member, fullpath, depth - 1)

            except Exception:
                pass

        base = getattr(self.main_module, "__name__", "module")
        recurse(self.main_module, [base], max_depth)

        return matches


    # ------------------------------------------------------------------
    # Impresión principal
    # ------------------------------------------------------------------

    def _print_object(self, obj, parts, verbose):
        if inspect.isclass(obj):
            self._print_class(obj, verbose)
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            self._print_callable(obj, parts)
        elif inspect.ismodule(obj):
            self._print_module(obj)
        else:
            self._print_generic(obj, parts)

    # ------------------------------------------------------------------
    # Impresores específicos
    # ------------------------------------------------------------------

    def _print_module(self, module):
        name = getattr(module, "__name__", str(module))
        doc = inspect.getdoc(module) or "— sin docstring —"

        self.console.print(
            Panel(
                Text(doc, style="dim"),
                title=f"📦 Módulo {name}",
                border_style="cyan",
            )
        )

    def _print_class(self, cls, verbose):
        title = f"📘 Clase {cls.__name__}"

        body = []

        # Dataclass
        if dataclasses.is_dataclass(cls):
            body.append(self._render_dataclass(cls))

        # Constructor
        init = cls.__init__
        if init is not object.__init__:
            body.append(self._render_signature(init, title="🔧 Constructor (__init__)"))

        # Docstring
        doc = inspect.getdoc(cls)
        if doc:
            body.append(self._render_docstring(doc))

        panel = Panel.fit(
            "\n\n".join(body),
            title=title,
            border_style="cyan",
        )
        self.console.print(panel)

        if verbose:
            self._print_public_methods(cls)

    def _print_callable(self, fn, parts):
        owner = parts[-2] if len(parts) > 1 else ""
        name = parts[-1]

        blocks = [
            self._render_signature(fn, title=f"🔦 {owner}.{name}()")
        ]

        doc = inspect.getdoc(fn)
        if doc:
            blocks.append(self._render_docstring(doc))

        self.console.print(
            Panel.fit(
                "\n\n".join(blocks),
                border_style="blue",
            )
        )

    def _print_generic(self, obj, parts):
        name = parts[-1]
        doc = inspect.getdoc(obj)
        if doc:
            self.console.print(
                Panel(
                    Text(doc, style="dim"),
                    title=f"🔸 {name}",
                    border_style="white",
                )
            )
        else:
            self.console.print(f"[yellow]ℹ️ {name} no tiene docstring.[/]")

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _render_signature(self, fn, title="Firma"):
        try:
            sig = inspect.signature(fn)
            hints = get_type_hints(fn)
        except Exception:
            return "[red]⚠️ Firma no disponible[/]"

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Input")
        table.add_column("Tipo")
        table.add_column("Default")

        for name, param in sig.parameters.items():
            ptype = hints.get(name, "")
            default = (
                repr(param.default)
                if param.default is not inspect._empty
                else ""
            )
            table.add_row(name, str(ptype), default)

        ret = hints.get("return")

        text = Text()
        text.append(f"{title}\n", style="bold")

        text.append("\n📥 Inputs:\n", style="bold blue")
        self.console.print(text)
        self.console.print(table)

        if ret is not None:
            self.console.print(
                f"\n📤 [bold green]Output:[/] {ret}"
            )

        return ""

    def _render_docstring(self, doc):
        return Syntax(
            doc,
            "python",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )

    def _render_dataclass(self, cls):
        table = Table(
            title="📦 Dataclass fields",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Campo")
        table.add_column("Tipo")
        table.add_column("Default")

        for f in dataclasses.fields(cls):
            table.add_row(
                f.name,
                str(f.type),
                repr(f.default) if f.default is not dataclasses.MISSING else "",
            )

        self.console.print(table)
        return ""

    def _print_public_methods(self, cls):
        table = Table(
            title="🔍 Métodos públicos",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Nombre")
        table.add_column("Tipo")

        for name, member in inspect.getmembers(cls):
            if name.startswith("_"):
                continue
            if inspect.isfunction(member) or inspect.ismethod(member):
                table.add_row(name, "method")

        self.console.print(table)

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    def _print_overview(self):
        name = getattr(self.main_module, "__name__", str(self.main_module))
        self.console.print(
            Panel(
                f"💡 Eärendil's Light encendida sobre [bold]{name}[/]\n\n"
                "Usa:\n"
                "  • light.help('Clase')\n"
                "  • light.help('Clase.metodo')\n"
                "  • light.help('metodo', verbose=True)",
                border_style="green",
            )
        )
