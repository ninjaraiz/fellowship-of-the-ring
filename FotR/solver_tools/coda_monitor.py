"""Live web monitor for CODA runs: a small page that follows a list of cases.

A background thread re-reads each case's Slurm log every ``interval``
seconds (:class:`~FotR.solver_tools.coda_log.CodaCase`) and a small HTTP
server hands the state to ``coda_monitor.html``, which plots it with
Plotly. Standard library only, so the server runs in a process of its own
without pandas, torch or FRODO.

The usual entry point is :meth:`CODAResiduals.monitor_cases`, which picks
the cases from a FRODO database and calls :func:`launch`. From a shell::

    python -m FotR.solver_tools.coda_monitor outputs/aoa_4.3363_mach_0.7247 \\
        outputs/aoa_4.9831_mach_0.8085 --port 8765

HTTP API (all ``GET``, JSON):

* ``/`` -- the page.
* ``/api/config`` -- title, polling interval and the list of cases.
* ``/api/summary`` -- :meth:`CodaCase.summary` of every case.
* ``/api/case?i=<n>&since=<row>&log=<log id>`` -- summary of case ``n``
  plus its columns from row ``since`` on. When ``log`` is not the log the
  case follows now (it was restarted), ``reset`` is true and the columns
  start at row 0.

The server only listens on 127.0.0.1 by default. On a shared machine any
user can reach a local port, so it is read-only and serves nothing but the
page and the logs of the cases it was given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .coda_log import CodaCase

__all__ = ['MonitorServer', 'MonitorHandle', 'launch', 'main']

_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'coda_monitor.html')
_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'FotR',
                          'coda_monitor')
_READY = 'CODA monitor: '
_PORT_TRIES = 20


class MonitorServer:
    """HTTP server plus the thread that keeps the cases up to date.

    Parameters
    ----------
    cases : list[dict]
        One ``{'path': ..., 'label': ..., 'info': {...}}`` per case;
        ``label`` and ``info`` are optional (see :class:`CodaCase`).
    title : str
        Shown at the top of the page. Default ``'CODA monitor'``.
    interval : float
        Seconds between two reads of every log, and between two polls of
        the page. Default 5.
    stale_after : float
        Seconds without writes after which a running case is shown as
        ``'quiet'``. Default 900.
    idle_timeout : float
        Seconds without any request after which the server stops by
        itself; 0 never stops. Default 1800. An open page polls every
        ``interval`` seconds, so this only fires once nobody is watching.
    host : str
        Interface to listen on. Default ``'127.0.0.1'``.
    port : int
        First port to try; if it is taken, the next ones up to 20 more
        are tried. 0 lets the system choose. Default 8765.

    Examples
    --------
    ::

        srv = MonitorServer([{'path': 'outputs/aoa_4.3363_mach_0.7247'}],
                            port=0).start()
        print(srv.url)
        srv.stop()
    """

    def __init__(self, cases: list, title: str = 'CODA monitor',
                 interval: float = 5.0, stale_after: float = 900.0,
                 idle_timeout: float = 1800.0, host: str = '127.0.0.1',
                 port: int = 8765) -> None:
        """Build the cases and bind the port; nothing runs until
        :meth:`start` or :meth:`serve_forever`."""
        if not cases:
            raise ValueError('No case to monitor.')
        self.cases = [CodaCase(c['path'], c.get('label'), c.get('info'))
                      for c in cases]
        self.title = title
        self.interval = float(interval)
        self.stale_after = float(stale_after)
        self.idle_timeout = float(idle_timeout)
        self.last_request = time.time()
        self._stop = threading.Event()
        self._serving = False
        self._threads: list = []
        with open(_HTML, 'rb') as fh:
            self.page = fh.read()
        self.httpd = self._bind(host, int(port))
        self.httpd.monitor = self

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

    @property
    def url(self) -> str:
        """Address of the page."""
        host, port = self.httpd.server_address[:2]
        return f'http://{host}:{port}/'

    # ── running ───────────────────────────────────────────────────────────

    def _poll(self) -> None:
        """Re-read every case each ``interval`` seconds; stop when idle."""
        while not self._stop.is_set():
            for case in self.cases:
                if self._stop.is_set():
                    return
                try:
                    with case.lock:
                        case.update()
                except OSError:
                    pass
            if (self.idle_timeout > 0
                    and time.time() - self.last_request > self.idle_timeout):
                print(f'Idle for {self.idle_timeout:.0f} s: stopping.',
                      flush=True)
                threading.Thread(target=self.stop, daemon=True).start()
                return
            self._stop.wait(self.interval)

    def start(self) -> 'MonitorServer':
        """Serve from background threads and return at once."""
        self._serving = True
        for target in (self._poll, self.httpd.serve_forever):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def serve_forever(self) -> None:
        """Serve until :meth:`stop`, the idle timeout or Ctrl-C."""
        poller = threading.Thread(target=self._poll, daemon=True)
        poller.start()
        self._serving = True
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            self.httpd.server_close()

    def stop(self) -> None:
        """Stop the poller and the HTTP server."""
        self._stop.set()
        if self._serving:
            # shutdown() waits for serve_forever(), which may never have run.
            self._serving = False
            self.httpd.shutdown()
        self.httpd.server_close()

    # ── API payloads ──────────────────────────────────────────────────────

    def config(self) -> dict:
        """Payload of ``/api/config``."""
        return {
            'title': self.title, 'interval': self.interval,
            'stale_after': self.stale_after,
            'cases': [{'i': i, 'label': c.label, 'info': c.info}
                      for i, c in enumerate(self.cases)],
        }

    def summary(self) -> dict:
        """Payload of ``/api/summary``."""
        now = time.time()
        out = []
        for i, case in enumerate(self.cases):
            with case.lock:
                s = case.summary(now, self.stale_after)
            s['i'] = i
            out.append(s)
        return {'now': now, 'cases': out}

    def case(self, i: int, since: int = 0, log: str = '') -> dict:
        """Payload of ``/api/case``."""
        case = self.cases[i]
        with case.lock:
            current = case.log_id or ''
            n = case.log.n_rows if case.log else 0
            reset = log != current or since > n or since < 0
            start = 0 if reset else since
            return {
                'i': i, 'log': current, 'reset': reset, 'since': start,
                'n': n, 'columns': case.series(start),
                'summary': case.summary(time.time(), self.stale_after),
            }


class _Handler(BaseHTTPRequestHandler):
    """Routes of :class:`MonitorServer`; nothing else is served."""

    server_version = 'FotR-CODA-monitor'

    def do_GET(self) -> None:
        """Answer the page or one of the three API routes."""
        monitor: MonitorServer = self.server.monitor
        monitor.last_request = time.time()
        url = urlparse(self.path)
        try:
            if url.path in ('/', '/index.html'):
                self._send(200, monitor.page, 'text/html; charset=utf-8')
            elif url.path == '/api/config':
                self._json(monitor.config())
            elif url.path == '/api/summary':
                self._json(monitor.summary())
            elif url.path == '/api/case':
                q = parse_qs(url.query)
                i = int(q.get('i', ['-1'])[0])
                if not 0 <= i < len(monitor.cases):
                    self._json({'error': f'no case {i}'}, 404)
                    return
                self._json(monitor.case(i, int(q.get('since', ['0'])[0]),
                                        q.get('log', [''])[0]))
            else:
                self._json({'error': 'not found'}, 404)
        except ValueError as exc:
            self._json({'error': str(exc)}, 400)

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        self._send(code, body, 'application/json')

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        """Keep the polling out of the server log."""


class MonitorHandle:
    """A monitor server running in a process of its own.

    Returned by :func:`launch`. The process outlives the Python session
    that started it and stops by itself after ``idle_timeout`` seconds
    without the page open.

    Attributes
    ----------
    url : str
        Address of the page (on the machine that runs the server).
    pid : int
        Process id.
    log_path : str
        Output of the server process.
    manifest_path : str
        JSON with the cases it was given.
    """

    def __init__(self, url: str, process: subprocess.Popen,
                 log_path: str, manifest_path: str) -> None:
        self.url = url
        self.pid = process.pid
        self.log_path = log_path
        self.manifest_path = manifest_path
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
        return f'<CODA monitor {self.url} pid={self.pid} {state}>'

    def _repr_html_(self) -> str:
        state = 'running' if self.alive() else 'stopped'
        return (f'CODA monitor <a href="{self.url}" target="_blank">'
                f'{self.url}</a> (pid {self.pid}, {state})')


def launch(cases: list, title: str = 'CODA monitor', port: int = 8765,
           interval: float = 5.0, stale_after: float = 900.0,
           idle_timeout: float = 1800.0, host: str = '127.0.0.1',
           timeout: float = 30.0) -> MonitorHandle:
    """Start a :class:`MonitorServer` in a separate process.

    The cases go through a JSON manifest in ``~/.cache/FotR/coda_monitor/``,
    next to the log of the process.

    Parameters
    ----------
    cases : list[dict]
        As in :class:`MonitorServer`.
    title, port, interval, stale_after, idle_timeout, host
        As in :class:`MonitorServer`.
    timeout : float
        Seconds to wait for the server to listen. Default 30.

    Returns
    -------
    MonitorHandle

    Raises
    ------
    RuntimeError
        If the process exits or does not listen within ``timeout``; the
        message carries the end of its log.

    Examples
    --------
    ::

        mon = launch([{'path': 'outputs/aoa_4.3363_mach_0.7247'}])
        mon.url
        mon.stop()
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    stem = os.path.join(
        _CACHE_DIR, f"monitor-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    )
    manifest_path, log_path = stem + '.json', stem + '.log'
    with open(manifest_path, 'w') as fh:
        json.dump({'title': title, 'cases': cases}, fh, indent=1)

    # The child must import FotR even when this session found it through
    # sys.path rather than an installed package.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        p for p in (repo, env.get('PYTHONPATH')) if p
    )
    cmd = [sys.executable, '-m', 'FotR.solver_tools.coda_monitor',
           '--manifest', manifest_path, '--host', host, '--port', str(port),
           '--interval', str(interval), '--stale-after', str(stale_after),
           '--idle-timeout', str(idle_timeout)]
    with open(log_path, 'w') as log_fh:
        process = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, start_new_session=True,
        )

    deadline = time.time() + timeout
    while time.time() < deadline:
        with open(log_path, errors='replace') as fh:
            text = fh.read()
        for line in text.splitlines():
            if line.startswith(_READY):
                return MonitorHandle(line[len(_READY):].strip(), process,
                                     log_path, manifest_path)
        if process.poll() is not None:
            raise RuntimeError(
                f'The monitor exited with code {process.returncode}:\n'
                f'{text[-2000:]}'
            )
        time.sleep(0.2)
    process.terminate()
    raise RuntimeError(f'The monitor did not start within {timeout} s; '
                       f'see {log_path}.')


def main(argv: Optional[list] = None) -> None:
    """Command line: ``python -m FotR.solver_tools.coda_monitor``."""
    parser = argparse.ArgumentParser(
        prog='python -m FotR.solver_tools.coda_monitor',
        description='Live web monitor of CODA runs (one Slurm log per '
                    'case folder).',
    )
    parser.add_argument('cases', nargs='*', help='case folders')
    parser.add_argument('--manifest', help='JSON written by launch()')
    parser.add_argument('--title', default=None)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--interval', type=float, default=5.0)
    parser.add_argument('--stale-after', type=float, default=900.0)
    parser.add_argument('--idle-timeout', type=float, default=1800.0)
    args = parser.parse_args(argv)

    cases, title = [], 'CODA monitor'
    if args.manifest:
        with open(args.manifest) as fh:
            manifest = json.load(fh)
        cases = list(manifest['cases'])
        title = manifest.get('title') or title
    cases += [{'path': os.path.abspath(p)} for p in args.cases]
    if not cases:
        parser.error('give case folders or --manifest')

    server = MonitorServer(
        cases, title=args.title or title, interval=args.interval,
        stale_after=args.stale_after, idle_timeout=args.idle_timeout,
        host=args.host, port=args.port,
    )
    print(f'{_READY}{server.url}', flush=True)
    print(f'{len(server.cases)} case(s), pid {os.getpid()}. Ctrl-C to stop.',
          flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
