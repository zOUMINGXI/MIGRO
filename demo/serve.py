"""Local web demo. Start after configuring the runtime assets in docs/runtime.md."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from run import validate

HERE = Path(__file__).resolve().parent
run_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, media="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", media + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.reply(200, self.server.page, "text/html")
        elif self.path == "/api/progress":
            try:
                progress = json.loads(self.server.progress_path.read_text())
            except FileNotFoundError:
                progress = {"stage": "idle"}
            self.reply(200, progress)
        elif self.path == "/api/info":
            self.reply(200, {"live": True})
        else:
            self.reply(404, {"error": "Page not found"})

    def do_POST(self):
        if self.path != "/api/run":
            self.reply(404, {"error": "Unknown endpoint"})
            return
        if self.headers.get_content_type() != "application/json":
            self.reply(415, {"error": "Expected a JSON session"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 24000:
                raise ValueError("Session must fit within 24 KB")
            case = json.loads(self.rfile.read(size))
            if not isinstance(case, dict):
                raise ValueError("Expected a session object")
            validate(case)
        except (ValueError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})
            return
        if not run_lock.acquire(blocking=False):
            self.reply(409, {"error": "A session is generating. Try again when it finishes."})
            return
        try:
            temporary = self.server.progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"stage":"loading"}))
            temporary.replace(self.server.progress_path)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request = root / "session.json"
                output = root / "trace.json"
                request.write_text(json.dumps(case))
                try:
                    subprocess.run([sys.executable, str(HERE / "run.py"), "--case", str(request),
                                    "--output", str(output), "--progress", str(self.server.progress_path)], check=True, timeout=3600)
                except (subprocess.SubprocessError, OSError):
                    self.reply(503, {"error": "Generation failed. Check the server log and runtime asset paths."})
                    return
                self.reply(200, output.read_bytes())
        finally:
            self.server.progress_path.unlink(missing_ok=True)
            run_lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18764)
    parser.add_argument("--trace", type=Path, help="Open a trace produced by run.py")
    args = parser.parse_args()
    os.environ.setdefault("MIGRO_DEVICE", "cpu")
    with tempfile.TemporaryDirectory() as scratch, ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.progress_path = Path(scratch) / "progress.json"
        page = (HERE / "index.html").read_text().replace("</head>", '<meta name="migro-live"></head>', 1)
        if args.trace:
            before, rest = page.split('<script type="application/json" id="capture">', 1)
            _, after = rest.split("</script>", 1)
            trace = json.dumps(json.loads(args.trace.read_text()), ensure_ascii=False).replace("<", "\\u003c")
            page = before + '<script type="application/json" id="capture">' + trace + "</script>" + after
        server.page = page.encode()
        print(f"MIGRO demo at http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()
