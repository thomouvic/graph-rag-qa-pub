"""
Local OpenAI-compatible embedding server using all-MiniLM-L6-v2.
No external deps beyond sentence-transformers + Python stdlib.

Usage:
    python embedding_server.py              # default port 8000
    python embedding_server.py --port 9000  # custom port

Then point any OpenAI-compatible client at http://localhost:8000/v1/embeddings
"""

import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
_encode_lock = threading.Lock()  # serialize model.encode() calls
print(f"Loaded {MODEL_NAME} (dim={model.get_sentence_embedding_dimension()})")


class ThreadingHTTPServer(HTTPServer):
    """Handle each request in a new thread so stuck connections don't block."""
    def process_request(self, request, client_address):
        t = threading.Thread(target=self.process_request_thread,
                             args=(request, client_address))
        t.daemon = True
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


class EmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not self.path.rstrip("/").endswith("/embeddings"):
            self._reply(404, {"error": f"Not found: {self.path}"})
            return

        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        inp = body.get("input", "")

        if isinstance(inp, str):
            texts = [inp]
        elif isinstance(inp, list):
            texts = [t if isinstance(t, str) else str(t) for t in inp]
        else:
            texts = [str(inp)]

        with _encode_lock:
            vectors = model.encode(texts).tolist()

        self._reply(200, {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec, "index": i}
                for i, vec in enumerate(vectors)
            ],
            "model": body.get("model", MODEL_NAME),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })

    def _reply(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        # Keep logs minimal
        if args and "200" not in str(args[1:2]):
            print(fmt % args)


if __name__ == "__main__":
    port = 8000
    for i, a in enumerate(sys.argv[1:]):
        if a == "--port" and i + 1 < len(sys.argv) - 1:
            port = int(sys.argv[i + 2])

    print(f"Embedding server on http://localhost:{port}/v1/embeddings")
    ThreadingHTTPServer(("127.0.0.1", port), EmbeddingHandler).serve_forever()
