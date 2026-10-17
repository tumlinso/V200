#!/usr/bin/env python3
"""Tiny HTTP process-group fixture used only by CPU tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
pid_file = os.environ.get("V200_FAKE_CHILD_PID_FILE")
if pid_file:
    with open(pid_file, "w", encoding="utf-8") as stream:
        stream.write(str(child.pid))


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, value: object) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/completion":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if request.get("test_mode") == "malformed":
            self._send(200, {"content": "2"})
        else:
            self._send(200, {"content": " 2\n", "tokens_predicted": 1})

    def log_message(self, format: str, *args: object) -> None:
        return


server = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
if os.environ.get("V200_FAKE_EXIT_ON_TERM"):
    import signal
    signal.signal(signal.SIGTERM, lambda *_args: os._exit(139))
server.serve_forever(poll_interval=0.2)
