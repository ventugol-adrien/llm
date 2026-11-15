import http.server
import json
import socketserver
import sys


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"model name is missing from the request"}}')
            return
        self.server.requests.append(body)
        prompt = body["messages"][-1]["content"]
        if not isinstance(prompt, str):
            prompt = prompt[0]["text"]
        if prompt == "http-error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"mock failure"}}')
            return
        self.send_response(200)
        self.end_headers()
        if body["stream"]:
            if prompt == "bad-stream":
                self.wfile.write(b"data: not-json\n\n")
                return
            for text in ("Hello", " world"):
                chunk = {"choices": [{"delta": {"content": text}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\r\n\r\n".encode())
                self.wfile.flush()
            if prompt != "truncated-stream":
                self.wfile.write(b"data: [DONE]\r\n\r\n")
        elif prompt == "api-error":
            self.wfile.write(b'{"error":{"message":"API failure"}}')
        elif prompt == "bad-response":
            self.wfile.write(b'{"choices":[]}')
        else:
            self.wfile.write(
                json.dumps({"choices": [{"message": {"content": prompt}}]}).encode()
            )


if __name__ == "__main__":
    if "--host" in sys.argv:
        path = sys.argv[sys.argv.index("--host") + 1]
    else:
        path = sys.argv[1]
    with UnixHTTPServer(path, Handler) as server:
        server.requests = []
        server.serve_forever()
