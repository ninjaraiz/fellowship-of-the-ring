"""``LEGOLAS.Viewer``: a small web viewer for a mesh, with three.js.

Same pattern as :mod:`FotR.solver_tools.coda_monitor`: a process of its own
runs a read-only HTTP server on 127.0.0.1 and the page draws with WebGL in
the browser. Over VS Code Remote-SSH the port is forwarded by itself.
three.js 0.160.0 ships inside the package (``static/three/``), so the page
needs no internet.

The server sends surfaces, never the volume: boundaries by name, and the
cells around a plane (:func:`.scene.section_faces`). The page pings every
minute; with no request for ``idle_timeout`` seconds the server stops.

From a shell::

    python -m FotR.characters.legolas.viewer MESH/wt_g2_mesh.h5 --port 8766

HTTP API (all ``GET``):

* ``/`` -- the page; ``/static/three/...`` -- three.js.
* ``/api/info`` -- JSON: file, sizes, bounding box, ``origin``, zones (with
  triangle counts and whether they are shown at start), census, warnings.
* ``/api/surface?zone=<name>`` -- binary (:func:`.scene.pack`) faces of a
  boundary or pseudo-zone: ``positions``, ``indices``, ``tri_cell``,
  ``edges``.
* ``/api/section?axis=<0|1|2>&value=<x>&mode=<slice|clip>&side=<-1|1>`` --
  binary, the same arrays for the cells around a plane.
* ``/api/cell?id=<n>`` -- JSON, :func:`.scene.cell_info`.
* ``/api/field?name=<metric>`` -- binary, one float32 per cell
  (:class:`.quality.Quality`), with ``min``, ``max``, ``p01``, ``p99`` and
  ``better`` in the header; NaN where the metric is not defined.
* ``/api/worst?name=<metric>&n=<k>`` -- JSON, the k worst cells.
* ``/api/ping`` -- keeps the server alive.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import scene
from .quality import METRICS

__all__ = ['Viewer', 'ViewerServer', 'ViewerHandle', 'launch', 'main']

_HERE = os.path.dirname(os.path.abspath(__file__))
_HTML = os.path.join(_HERE, 'viewer.html')
_STATIC = os.path.join(_HERE, 'static')
_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'FotR',
                          'legolas')
_READY = 'LEGOLAS viewer: '
_PORT_TRIES = 20
DEFAULT_PORT = 8766


class ViewerServer:
    """HTTP server for one mesh, plus the idle watchdog.

    Parameters
    ----------
    mesh : Mesh
        The mesh to show.
    title : str
        Shown in the page header; default the file name.
    host : str
        Interface to listen on. Default ``'127.0.0.1'``.
    port : int
        First port to try; the next 20 are tried if it is taken, 0 lets the
        system choose. Default 8766.
    idle_timeout : float
        Seconds without any request after which the server stops by itself;
        0 never stops. Default 1800. An open page pings every 60 s.
    max_triangles : int
        Budget per request; bigger zones are decimated (every k-th face)
        and start hidden. Default :data:`.scene.MAX_TRIANGLES`.
    """

    def __init__(self, mesh, title: str | None = None,
                 host: str = '127.0.0.1', port: int = DEFAULT_PORT,
                 idle_timeout: float = 1800.0,
                 max_triangles: int = scene.MAX_TRIANGLES) -> None:
        from .census import Census
        from .quality import Quality

        self.mesh = mesh
        path = mesh.meta.get('path')
        self.title = title or (os.path.basename(path) if path else 'mesh')
        self.idle_timeout = float(idle_timeout)
        self.max_triangles = int(max_triangles)
        self.last_request = time.time()
        self._stop = threading.Event()
        self._serving = False
        self._cache: dict = {}
        self._lock = threading.Lock()
        with open(_HTML, 'rb') as fh:
            self.page = fh.read()
        self.static = self._static_files()
        # everything the page asks for first is computed before "ready"
        self.quality = Quality(SimpleNamespace(mesh=mesh))
        self.info = self._info(Census(SimpleNamespace(mesh=mesh)).summary())
        self.httpd = self._bind(host, int(port))
        self.httpd.viewer = self

    @staticmethod
    def _bind(host: str, port: int) -> ThreadingHTTPServer:
        """Listen on ``port`` or on the first free one after it."""
        last_error = None
        for p in ([0] if port == 0 else range(port, port + _PORT_TRIES + 1)):
            try:
                httpd = ThreadingHTTPServer((host, p), _Handler)
            except OSError as exc:
                last_error = exc
                continue
            httpd.daemon_threads = True
            return httpd
        raise OSError(f'No free port in {port}-{port + _PORT_TRIES} '
                      f'on {host}: {last_error}')

    @staticmethod
    def _static_files() -> dict:
        """Whitelist ``{url path: file}`` of everything under ``static/``."""
        out = {}
        for root, _, files in os.walk(_STATIC):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, _STATIC).replace(os.sep, '/')
                out['/static/' + rel] = full
        return out

    @property
    def url(self) -> str:
        """Address of the page."""
        host, port = self.httpd.server_address[:2]
        return f'http://{host}:{port}/'

    # -- payloads -----------------------------------------------------------
    def _info(self, census: dict) -> dict:
        m = self.mesh
        zones = scene.zone_list(m)
        for z in zones:
            z['visible'] = 0 < z['triangles'] <= self.max_triangles
        return {
            'title': self.title,
            'path': m.meta.get('path'),
            'format': m.meta.get('format'),
            'ngeo': m.ngeo,
            'n_points': m.n_points,
            'n_cells': m.n_cells,
            'cells': census['cells'],
            'bbox': census['bbox'],
            'origin': scene.origin(m).tolist(),
            'zones': zones,
            'census': {k: census[k] for k in (
                'exterior_faces', 'unassigned_faces', 'nonmanifold_faces',
                'orphan_nodes')},
            'warnings': census['warnings'],
            'max_triangles': self.max_triangles,
            'has_file_index': 'file_cell_index' in m.meta,
            'fields': [{'name': k, 'label': v['label'], 'better': v['better']}
                       for k, v in METRICS.items()],
        }

    def _geometry(self, elem, face, header: dict) -> bytes:
        g = scene.face_geometry(self.mesh, elem, face, self.max_triangles)
        header = dict(header, stride=g['stride'], n_faces=g['n_faces'],
                      triangles=int(len(g['indices'])))
        return scene.pack(header, {k: g[k] for k in (
            'positions', 'indices', 'tri_cell', 'edges')})

    def surface(self, zone: str) -> bytes:
        """Payload of ``/api/surface`` (cached per zone)."""
        with self._lock:
            if zone not in self._cache:
                elem, face = scene.zone_faces(self.mesh, zone)
                self._cache[zone] = self._geometry(elem, face, {'zone': zone})
            return self._cache[zone]

    def section(self, axis: int, value: float, mode: str, side: int) -> bytes:
        """Payload of ``/api/section``."""
        if axis not in (0, 1, 2):
            raise ValueError('axis must be 0, 1 or 2')
        with self._lock:
            elem, face = scene.section_faces(self.mesh, axis, value, mode,
                                             side)
            return self._geometry(elem, face, {
                'axis': axis, 'value': value, 'mode': mode, 'side': side})

    def cell(self, cell: int) -> dict:
        """Payload of ``/api/cell``."""
        with self._lock:
            info = scene.cell_info(self.mesh, cell)
            if self.quality._metrics is not None:
                info['quality'] = {
                    k: (None if not np.isfinite(v[cell]) else float(v[cell]))
                    for k, v in self.quality.metrics().items()}
            return info

    def field(self, name: str) -> bytes:
        """Payload of ``/api/field`` (a quality metric per cell)."""
        with self._lock:
            key = 'field:' + name
            if key not in self._cache:
                values = self.quality.field(name).astype(np.float32)
                stats = self.quality.summary().get(name)
                if stats is None:
                    stats = {k: 0.0 for k in ('min', 'max', 'p01', 'p99')}
                head = {'name': name, 'better': METRICS[name]['better']}
                head.update({k: stats[k] for k in ('min', 'max', 'p01',
                                                    'p99')})
                self._cache[key] = scene.pack(head, {'values': values})
            return self._cache[key]

    def worst(self, name: str, n: int) -> list:
        """Payload of ``/api/worst``."""
        with self._lock:
            return self.quality.worst(name, max(1, min(int(n), 1000)))

    # -- running ------------------------------------------------------------
    def _watch(self) -> None:
        while not self._stop.wait(1.0):
            if (self.idle_timeout > 0
                    and time.time() - self.last_request > self.idle_timeout):
                print(f'No request for {self.idle_timeout:.0f} s: stopping.',
                      flush=True)
                threading.Thread(target=self.httpd.shutdown,
                                 daemon=True).start()
                return

    def start(self) -> 'ViewerServer':
        """Serve from background threads and return at once."""
        self._serving = True
        for target in (self._watch, self.httpd.serve_forever):
            threading.Thread(target=target, daemon=True).start()
        return self

    def serve_forever(self) -> None:
        """Serve until :meth:`stop`, the idle timeout or Ctrl-C."""
        threading.Thread(target=self._watch, daemon=True).start()
        self._serving = True
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            self.httpd.server_close()

    def stop(self) -> None:
        """Stop the watchdog and the HTTP server."""
        self._stop.set()
        if self._serving:
            self._serving = False
            self.httpd.shutdown()
        self.httpd.server_close()


class _Handler(BaseHTTPRequestHandler):
    """Routes of :class:`ViewerServer`; nothing else is served."""

    server_version = 'FotR-LEGOLAS-viewer'

    def do_GET(self) -> None:
        """Answer the page, a static file or an API route."""
        v: ViewerServer = self.server.viewer
        v.last_request = time.time()
        url = urlparse(self.path)
        q = parse_qs(url.query)
        arg = lambda k, d=None: q.get(k, [d])[0]
        try:
            if url.path in ('/', '/index.html'):
                self._send(200, v.page, 'text/html; charset=utf-8')
            elif url.path in v.static:
                with open(v.static[url.path], 'rb') as fh:
                    body = fh.read()
                ctype = ('text/javascript' if url.path.endswith('.js') else
                         mimetypes.guess_type(url.path)[0] or 'text/plain')
                self._send(200, body, ctype, cache=True)
            elif url.path == '/api/info':
                self._json(v.info)
            elif url.path == '/api/ping':
                self._json({'ok': True})
            elif url.path == '/api/surface':
                self._send(200, v.surface(arg('zone', '')),
                           'application/octet-stream')
            elif url.path == '/api/section':
                self._send(200, v.section(int(arg('axis', '0')),
                                          float(arg('value', '0')),
                                          arg('mode', 'slice'),
                                          int(arg('side', '-1'))),
                           'application/octet-stream')
            elif url.path == '/api/cell':
                self._json(v.cell(int(arg('id', '-1'))))
            elif url.path == '/api/field':
                self._send(200, v.field(arg('name', '')),
                           'application/octet-stream')
            elif url.path == '/api/worst':
                self._json(v.worst(arg('name', 'scaled_jacobian'),
                                   int(arg('n', '10'))))
            else:
                self._json({'error': 'not found'}, 404)
        except KeyError as exc:
            self._json({'error': str(exc.args[0] if exc.args else exc)}, 404)
        except ValueError as exc:
            self._json({'error': str(exc)}, 400)

    def _json(self, payload, code: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        self._send(code, body, 'application/json')

    def _send(self, code: int, body: bytes, ctype: str,
              cache: bool = False) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control',
                         'max-age=86400' if cache else 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        """Keep the requests out of the server log."""


class ViewerHandle:
    """A viewer server running in a process of its own.

    Returned by :meth:`LEGOLAS.Viewer.show <Viewer.show>`. The process
    outlives the Python session that started it and stops by itself after
    ``idle_timeout`` seconds with no page open.

    Attributes
    ----------
    url : str
        Address of the page, on the machine that runs the server.
    pid : int
        Process id.
    log_path : str
        Output of the server process.
    """

    def __init__(self, url: str, process: subprocess.Popen,
                 log_path: str) -> None:
        self.url = url
        self.pid = process.pid
        self.log_path = log_path
        self._process = process

    def alive(self) -> bool:
        """True while the server process runs."""
        return self._process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the server process (kill it after ``timeout`` s)."""
        if not self.alive():
            return
        self._process.terminate()
        try:
            self._process.wait(timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout)

    def __repr__(self) -> str:
        state = 'running' if self.alive() else 'stopped'
        return f'<LEGOLAS viewer {self.url} pid={self.pid} {state}>'

    def _repr_html_(self) -> str:
        state = 'running' if self.alive() else 'stopped'
        return (f'LEGOLAS viewer <a href="{self.url}" target="_blank">'
                f'{self.url}</a> (pid {self.pid}, {state})')


def launch(manifest: dict, host: str = '127.0.0.1', port: int = DEFAULT_PORT,
           idle_timeout: float = 1800.0,
           max_triangles: int = scene.MAX_TRIANGLES,
           timeout: float = 120.0) -> ViewerHandle:
    """Start a :class:`ViewerServer` in a separate process.

    Parameters
    ----------
    manifest : dict
        ``path`` (mesh file), ``fmt`` and ``read_kwargs``; or ``npz`` (a
        :meth:`Mesh.save` file, deleted by the server once read). Optional
        ``title``.
    host, port, idle_timeout, max_triangles
        As in :class:`ViewerServer`.
    timeout : float
        Seconds to wait for the server to read the mesh and listen.
        Default 120.

    Returns
    -------
    ViewerHandle

    Raises
    ------
    RuntimeError
        If the process exits or does not listen in time; the message
        carries the end of its log.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    stem = os.path.join(_CACHE_DIR, f"viewer-{time.strftime('%Y%m%d-%H%M%S')}"
                                    f"-{os.getpid()}-{int(time.time() * 1e3) % 1000}")
    manifest_path, log_path = stem + '.json', stem + '.log'
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=1)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        p for p in (repo, env.get('PYTHONPATH')) if p)
    cmd = [sys.executable, '-m', 'FotR.characters.legolas.viewer',
           '--manifest', manifest_path, '--host', host, '--port', str(port),
           '--idle-timeout', str(idle_timeout),
           '--max-triangles', str(max_triangles)]
    with open(log_path, 'w') as log_fh:
        process = subprocess.Popen(cmd, stdout=log_fh,
                                   stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env,
                                   start_new_session=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        with open(log_path, errors='replace') as fh:
            text = fh.read()
        for line in text.splitlines():
            if line.startswith(_READY):
                return ViewerHandle(line[len(_READY):].strip(), process,
                                    log_path)
        if process.poll() is not None:
            raise RuntimeError(f'The viewer exited with code '
                               f'{process.returncode}:\n{text[-2000:]}')
        time.sleep(0.2)
    process.terminate()
    raise RuntimeError(f'The viewer did not start within {timeout} s; '
                       f'see {log_path}.')


class Viewer:
    """Look at the mesh in the browser: boundaries, element edges, cuts.

    Reached as ``lg.viewer``. :meth:`show` starts a small server in a
    process of its own and returns its address; open it in a browser (over
    VS Code Remote-SSH the port is forwarded by itself, see the *Ports*
    tab). three.js ships with FotR, so no internet is needed.

    The page offers: every boundary as a coloured surface with the element
    edges drawn over it; a crinkle *slice* (the layer of cells crossed by a
    plane) or *clip* (the cells on one side) to see inside the volume; and
    picking a cell to see its kind, zone, neighbours and boundary faces.

    Parameters
    ----------
    legolas : LEGOLAS
        The owner; the mesh is ``legolas.mesh``.
    """

    def __init__(self, legolas):
        self._lg = legolas
        self.handle = None

    def _manifest(self, title):
        source = getattr(self._lg, '_source', None)
        if source is not None:
            path, fmt, kwargs = source
            return {'path': path, 'fmt': fmt, 'read_kwargs': kwargs,
                    'title': title}
        os.makedirs(_CACHE_DIR, exist_ok=True)
        npz = os.path.join(_CACHE_DIR, f'mesh-{os.getpid()}-'
                                       f'{int(time.time() * 1e3)}.npz')
        self._lg.mesh.save(npz)
        return {'npz': npz, 'title': title or self._lg.name}

    def show(self, port: int = DEFAULT_PORT, host: str = '127.0.0.1',
             idle_timeout: float = 1800.0,
             max_triangles: int = scene.MAX_TRIANGLES,
             title: str | None = None, timeout: float = 120.0
             ) -> ViewerHandle:
        """Start the viewer in its own process and return its handle.

        A mesh read from a file is read again by the server process (it is
        usually faster than shipping it); a mesh built in memory is saved
        to ``~/.cache/FotR/legolas/`` first.

        Parameters
        ----------
        port : int
            First port to try (the next 20 too). Default 8766.
        host : str
            Default ``'127.0.0.1'``: only this machine can connect.
        idle_timeout : float
            Seconds without the page open before the server stops.
            Default 1800; 0 never.
        max_triangles : int
            Budget per request; bigger zones are decimated and start
            hidden. Default 1 000 000.
        title : str, optional
            Page title; default the file name.
        timeout : float
            Seconds to wait for the server. Default 120.

        Returns
        -------
        ViewerHandle
            ``url``, ``stop()``; shown as a link in Jupyter.

        Examples
        --------
        ::

            lg = LEGOLAS('MESH/Wind_turbine_disk_fluid_cgns_g2_mesh.h5')
            h = lg.viewer.show()
            h.url              # 'http://127.0.0.1:8766/'
            h.stop()
        """
        self.handle = launch(self._manifest(title), host=host, port=port,
                             idle_timeout=idle_timeout,
                             max_triangles=max_triangles, timeout=timeout)
        return self.handle

    def server(self, port: int = 0, host: str = '127.0.0.1',
               idle_timeout: float = 0.0,
               max_triangles: int = scene.MAX_TRIANGLES,
               title: str | None = None) -> ViewerServer:
        """A server in this process, bound but not started.

        Call ``.start()`` (background threads) or ``.serve_forever()``
        (blocking, Ctrl-C to stop). Useful in scripts and tests; in a
        notebook prefer :meth:`show`, which does not tie up the kernel.
        """
        return ViewerServer(self._lg.mesh, title=title, host=host, port=port,
                            idle_timeout=idle_timeout,
                            max_triangles=max_triangles)

    def stop(self) -> None:
        """Stop the server started by :meth:`show`, if any."""
        if self.handle is not None:
            self.handle.stop()

    def __repr__(self) -> str:
        state = repr(self.handle) if self.handle else 'not started'
        return f"<LEGOLAS.Viewer of {self._lg.name}: {state}>"


def main(argv=None) -> None:
    """Command line: ``python -m FotR.characters.legolas.viewer``."""
    parser = argparse.ArgumentParser(
        prog='python -m FotR.characters.legolas.viewer',
        description='Web viewer of a CFD mesh (HOPR .h5, CGNS, gmsh .msh, '
                    'HORSES3D .hmesh).')
    parser.add_argument('mesh', nargs='?', help='mesh file')
    parser.add_argument('--fmt', default=None)
    parser.add_argument('--manifest', help='JSON written by launch()')
    parser.add_argument('--title', default=None)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--idle-timeout', type=float, default=0.0)
    parser.add_argument('--max-triangles', type=int,
                        default=scene.MAX_TRIANGLES)
    args = parser.parse_args(argv)

    from .io import read
    from .mesh import Mesh

    title = args.title
    if args.manifest:
        with open(args.manifest) as fh:
            man = json.load(fh)
        title = title or man.get('title')
        if 'npz' in man:
            mesh = Mesh.load(man['npz'])
            try:
                os.remove(man['npz'])
            except OSError:
                pass
        else:
            mesh = read(man['path'], fmt=man.get('fmt'),
                        **(man.get('read_kwargs') or {}))
    elif args.mesh:
        mesh = read(args.mesh, fmt=args.fmt)
    else:
        parser.error('give a mesh file or --manifest')
    server = ViewerServer(mesh, title=title, host=args.host, port=args.port,
                          idle_timeout=args.idle_timeout,
                          max_triangles=args.max_triangles)
    print(f'{_READY}{server.url}', flush=True)
    print(f'{mesh!r}, pid {os.getpid()}. Ctrl-C to stop.', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
