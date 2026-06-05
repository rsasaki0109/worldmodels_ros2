"""Reference remote World Model server (stdlib only, GPU-free).

This is the *other half* of the RemoteAdapter: a tiny HTTP server that speaks
the shared wire format. On a GPU box you swap the backing adapter from ``dummy``
to a heavy one (``cosmos``, ``dreamzero``, ...) and ROS 2 clients reach it
unchanged through the RemoteAdapter.

    world-model-server --adapter dummy --port 8080
    # POST /predict_future   -> FuturePrediction JSON
    # GET  /health           -> {"status": "ok", ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import wire
from .registry import load_model


def make_handler(adapter, default_horizon: int = 8):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence default stderr logging
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"status": "ok", "adapter": adapter.info()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/predict_future":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                obs = wire.observation_from_payload(payload)
                action = wire.action_from_payload(payload)
                horizon = int(payload.get("horizon", default_horizon))
                if action is not None and action.horizon > 0:
                    horizon = action.horizon
                pred = adapter.predict_future(obs, action, horizon=horizon)
                self._send(200, wire.prediction_to_response(pred))
            except Exception as exc:  # noqa: BLE001 -- report, do not crash the server
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def build_server(adapter_name: str = "dummy", host: str = "127.0.0.1", port: int = 8080, **adapter_kwargs):
    adapter = load_model(adapter_name, **adapter_kwargs)
    httpd = ThreadingHTTPServer((host, port), make_handler(adapter))
    return httpd, adapter


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="world-model-server", description="Reference remote World Model server")
    p.add_argument("--adapter", default="dummy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--url", default=None, help="remote url (when --adapter remote, i.e. chaining)")
    args = p.parse_args(argv)

    kwargs = {"url": args.url} if (args.adapter == "remote" and args.url) else {}
    httpd, adapter = build_server(args.adapter, args.host, args.port, **kwargs)
    print(f"world-model-server: adapter='{args.adapter}' on http://{args.host}:{args.port} "
          f"(POST /predict_future, GET /health)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
