"""Standalone HTTP server for the Spectra Scheduler demonstration GUI.

Serves static UI assets from web/static/ and handles JSON API requests.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Ensure src/ is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from web.api import (  # noqa: E402
    PRESETS,
    SCHEDULER_REGISTRY,
    get_available_scenarios,
    get_available_schedulers,
    run_simulation,
)
from web.timing_backend import TimingBusyError  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORTS_DIR = REPO_ROOT / "reports" / "generated"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


class SpectraConsoleHandler(BaseHTTPRequestHandler):
    server_version = "SpectraConsole/1.0"

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": message, "status": status}, status=status)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/api/scenarios":
            self._send_json(get_available_scenarios())
            return
        elif path == "/api/schedulers":
            self._send_json(get_available_schedulers())
            return
        elif path == "/api/presets":
            self._send_json(PRESETS)
            return
        elif path == "/api/reports":
            reports = []
            if REPORTS_DIR.exists():
                for p in REPORTS_DIR.glob("*.json"):
                    reports.append({"name": p.name, "size": p.stat().st_size})
            self._send_json(reports)
            return
        elif path.startswith("/api/report/"):
            report_name = path.replace("/api/report/", "")
            report_file = (REPORTS_DIR / report_name).resolve()
            if not report_file.is_relative_to(REPORTS_DIR.resolve()):
                self._send_error_json("Access forbidden", HTTPStatus.FORBIDDEN)
                return
            if report_file.exists() and report_file.is_file():
                try:
                    with open(report_file, encoding="utf-8") as f:
                        self._send_json(json.load(f))
                except Exception as e:
                    self._send_error_json(
                        f"Failed to read report: {e}",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
            else:
                self._send_error_json("Report not found", HTTPStatus.NOT_FOUND)
            return

        # Serve static files
        self._serve_static(path)

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/api/run":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._send_error_json("Invalid Content-Length")
                return
            if content_length > 16384:
                self._send_error_json(
                    "Request body is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                )
                return
            if content_length <= 0:
                self._send_error_json("Empty request body")
                return

            try:
                body = self.rfile.read(content_length).decode("utf-8")
                params = json.loads(body)
                if not isinstance(params, dict):
                    raise ValueError("Expected a JSON object")
                scenario_id = str(params.get("scenario", "frequency-agile"))
                baseline_id = str(params.get("baseline", "round-robin"))
                active_id = str(params.get("active", "track-aware"))
                seed = int(params.get("seed", 0))
                sensitivity_dbm = params.get("sensitivity_dbm")
                if sensitivity_dbm is not None:
                    sensitivity_dbm = float(sensitivity_dbm)
                noise_std_db = params.get("noise_std_db")
                if noise_std_db is not None:
                    noise_std_db = float(noise_std_db)
                false_alarm_prob = params.get("false_alarm_prob")
                if false_alarm_prob is not None:
                    false_alarm_prob = float(false_alarm_prob)
                retune_steps = params.get("retune_steps")
                if retune_steps is not None:
                    retune_steps = int(retune_steps)
                perturbation = params.get("perturbation")
                if perturbation:
                    perturbation = str(perturbation)
                if not 0 <= seed <= 99999:
                    raise ValueError("Seed must be in 0..99999")
                if sensitivity_dbm is not None and not -120 <= sensitivity_dbm <= -40:
                    raise ValueError("Sensitivity must be in -120..-40 dBm")
                if retune_steps is not None and not 0 <= retune_steps <= 10:
                    raise ValueError("Retune steps must be in 0..10")
                if noise_std_db is not None and not 0 <= noise_std_db <= 30:
                    raise ValueError("Noise standard deviation must be in 0..30 dB")
                if false_alarm_prob is not None and not 0 <= false_alarm_prob <= 1:
                    raise ValueError("False alarm probability must be in 0..1")
                if perturbation not in (None, "frequency-hop", "popup-threat"):
                    raise ValueError("Unknown environment variation")
            except ValueError as e:
                self._send_error_json(str(e), HTTPStatus.BAD_REQUEST)
                return
            except Exception as e:
                self._send_error_json(f"Malformed request parameters: {e}", HTTPStatus.BAD_REQUEST)
                return

            valid_scenarios = {s["id"] for s in get_available_scenarios()}
            if scenario_id not in valid_scenarios:
                self._send_error_json(
                    f"Unknown scenario ID: {scenario_id!r}", HTTPStatus.BAD_REQUEST
                )
                return

            if baseline_id not in SCHEDULER_REGISTRY:
                self._send_error_json(
                    f"Unknown baseline scheduler ID: {baseline_id!r}", HTTPStatus.BAD_REQUEST
                )
                return

            if active_id not in SCHEDULER_REGISTRY:
                self._send_error_json(
                    f"Unknown active scheduler ID: {active_id!r}", HTTPStatus.BAD_REQUEST
                )
                return

            try:
                result = run_simulation(
                    scenario_id=scenario_id,
                    baseline_id=baseline_id,
                    active_id=active_id,
                    seed=seed,
                    sensitivity_dbm=sensitivity_dbm,
                    noise_std_db=noise_std_db,
                    false_alarm_prob=false_alarm_prob,
                    retune_steps=retune_steps,
                    perturbation=perturbation,
                )
                self._send_json(result)
            except TimingBusyError as e:
                self._send_error_json(str(e), HTTPStatus.SERVICE_UNAVAILABLE)
            except ValueError as e:
                self._send_error_json(str(e), HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._send_error_json(
                    f"Simulation execution error: {e}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        self._send_error_json("Endpoint not found", HTTPStatus.NOT_FOUND)

    def _serve_static(self, path: str) -> None:
        clean_path = path.lstrip("/")
        if clean_path.startswith("static/"):
            clean_path = clean_path[len("static/") :]
        if not clean_path or clean_path == "/":
            clean_path = "index.html"

        file_path = (STATIC_DIR / clean_path).resolve()

        # Prevent directory traversal attacks
        if not file_path.is_relative_to(STATIC_DIR.resolve()):
            self._send_error_json("Access forbidden", HTTPStatus.FORBIDDEN)
            return

        if not file_path.exists() or not file_path.is_file():
            # Only fallback to index.html for extensionless navigation routes
            if "." not in clean_path:
                file_path = STATIC_DIR / "index.html"
            else:
                self._send_error_json(f"Asset not found: {clean_path}", HTTPStatus.NOT_FOUND)
                return

            if not file_path.exists():
                self._send_error_json("Not Found", HTTPStatus.NOT_FOUND)
                return

        suffix = file_path.suffix.lower()
        guess = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        content_type = MIME_TYPES.get(suffix, guess)

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._send_error_json(f"Failed to read asset: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, SpectraConsoleHandler)
    print(f"Spectra Scheduler demo: http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Spectra Console server...")
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectra Scheduler demonstration server")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--timing-model", type=Path, help="CUDA timing checkpoint or ensemble")
    args = parser.parse_args()
    if args.timing_model is not None:
        import os
        os.environ["SPECTRA_TIMING_CHECKPOINT"] = str(args.timing_model.resolve())
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
