"""Loopback-only dashboard. Workers run in separate Python processes."""

from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from urllib.parse import unquote, urlparse

from .storage import read_json, write_json
from .rewards import RewardConfig


WEB = Path(__file__).parent / "web"
RUN_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}$")


class Lab:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker = None
        self.job = None
        self.lock = threading.Lock()

    def directory(self, identifier):
        if not isinstance(identifier, str) or not RUN_ID.fullmatch(identifier):
            raise ValueError("Invalid experiment ID")
        path = self.root / identifier
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError("Invalid experiment directory")
        return path

    def listing(self):
        rows = []
        for path in sorted(self.root.iterdir(), reverse=True):
            if not path.is_dir() or path.is_symlink() or not RUN_ID.fullmatch(path.name):
                continue
            progress = read_json(path / "progress.json", {})
            rows.append({"id": path.name, "config": read_json(path / "config.json", {}),
                         "progress": {k: v for k, v in progress.items() if k != "history"},
                         "evaluation": read_json(path / "evaluation.json"),
                         "has_model": (path / "best.zip").exists() or (path / "final.zip").exists(),
                         "has_replay": (path / "replay.json").exists() or (path / "validation/replay.json").exists(),
                         "error": read_json(path / "error.json"),
                         "videos": [p.name for p in (path / "videos").glob("*.mp4")]})
        busy = self.worker is not None and self.worker.poll() is None
        return {"runs": rows, "job": self.job if busy else None}

    def launch(self, args, identifier, kind):
        with self.lock:
            if self.worker and self.worker.poll() is None:
                raise ValueError("A job is already running. Stop it or wait for it to finish.")
            log_dir = self.root / "_logs"
            log_dir.mkdir(exist_ok=True)
            log = (log_dir / f"{identifier}-{kind}.log").open("w")
            try:
                self.worker = subprocess.Popen([sys.executable, "-m", "rocketlab.cli", *args],
                                               stdout=log, stderr=subprocess.STDOUT)
            finally:
                log.close()
            self.job = {"id": identifier, "kind": kind, "pid": self.worker.pid}
            process = self.worker
            def watch():
                code = process.wait()
                write_json(log_dir / f"{identifier}-{kind}.json", {"exit_code": code})
            threading.Thread(target=watch, daemon=True).start()
        return self.job

    def start(self, data):
        mode = data.get("mode", "baseline")
        config = RewardConfig(mode=mode, time_cost=float(data.get("time_cost", .05)),
                              hover_bonus=float(data.get("hover_bonus", 1)),
                              potential_scale=float(data.get("potential_scale", 20)))
        steps = int(data.get("steps", 1000000))
        seed = int(data.get("seed", 7))
        if not 4096 <= steps <= 10000000 or not 0 <= seed < 2**31:
            raise ValueError("Use 4,096–10,000,000 steps and a seed from 0 to 2,147,483,647.")
        identifier = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + mode
        directory = self.directory(identifier)
        return self.launch(["train", "--output", str(directory), "--mode", mode,
                            "--steps", str(steps), "--seed", str(seed),
                            "--time-cost", str(config.time_cost), "--hover-bonus", str(config.hover_bonus),
                            "--potential-scale", str(config.potential_scale)], identifier, "train")

    def action(self, identifier, kind):
        directory = self.directory(identifier)
        model = directory / "best.zip"
        if not model.exists():
            model = directory / "final.zip"
        if not model.is_file():
            raise ValueError("This experiment has no saved model yet.")
        args = [kind, "--model", str(model), "--output", str(directory / "videos" if kind == "record" else directory)]
        if kind == "record":
            report = read_json(directory / "evaluation.json", {})
            success = next((r for r in report.get("results", []) if r["landed"]), None)
            args += ["--seed", str(success["seed"] if success else 20000)]
        return self.launch(args, identifier, kind)


def handler_for(lab):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path == "/api/runs":
                    return self.send(200, lab.listing())
                match = re.fullmatch(r"/api/run/([^/]+)/(progress|replay|csv|video/[^/]+)", path)
                if match:
                    directory = lab.directory(match[1])
                    asset = match[2]
                    if asset == "progress":
                        return self.send(200, read_json(directory / "progress.json", {}))
                    if asset == "replay":
                        replay = read_json(directory / "replay.json") or read_json(directory / "validation/replay.json")
                        return self.send(200 if replay else 404, replay or {"error": "No replay yet. The first validation runs at 50,000 steps."})
                    file = directory / "episodes.csv" if asset == "csv" else directory / "videos" / asset[6:]
                    if file.is_file() and not file.is_symlink() and file.suffix in {".csv", ".mp4"}:
                        return self.send(200, file.read_bytes(), "text/csv" if file.suffix == ".csv" else "video/mp4")
                if path in {"/", "/app.js", "/style.css", "/dm-sans.ttf", "/space-grotesk.ttf"}:
                    file = WEB / ("index.html" if path == "/" else path[1:])
                    return self.send(200, file.read_bytes(), mimetypes.guess_type(str(file))[0])
                self.send(404, {"error": "Not found"})
            except (ValueError, OSError) as error:
                self.send(400, {"error": str(error)})

        def do_POST(self):
            # Reject cross-origin browser requests to this local process launcher.
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"} or (origin and origin not in {f"http://{host}"}):
                return self.send(403, {"error": "Local same-origin requests only"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 4096:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(size) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                path = urlparse(self.path).path
                if path == "/api/train":
                    return self.send(202, lab.start(data))
                if path == "/api/stop":
                    with lab.lock:
                        if lab.worker and lab.worker.poll() is None:
                            lab.worker.send_signal(signal.SIGINT)
                    return self.send(200, {"stopping": True})
                if path in {"/api/evaluate", "/api/record"}:
                    return self.send(202, lab.action(data.get("id"), path[5:]))
                return self.send(404, {"error": "Not found"})
            except (ValueError, TypeError, OSError) as error:
                self.send(400, {"error": str(error)})
    return Handler


def serve(root, port=8765):
    lab = Lab(root)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(lab))
    print(f"Lunar Lander Lab → http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Experiments: {lab.root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if lab.worker and lab.worker.poll() is None:
            lab.worker.send_signal(signal.SIGINT)
    finally:
        server.server_close()
