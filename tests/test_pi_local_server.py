"""Exercise the real pi CLI against a loopback provider, with no API spend."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading

import pytest

from hillclimb.agents.base import OperatorRequest
from hillclimb.agents.pi_cli import PiCliAgent


@pytest.mark.skipif(shutil.which("pi") is None, reason="requires an installed pi CLI")
def test_real_pi_sampling_tools_fork_and_provider_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if payload.get("temperature") == 0.99:
                body = json.dumps({"error": {"message": "temperature is deprecated for this model", "type": "invalid_request_error"}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            messages = payload["messages"]
            if payload.get("tools") and messages[-1]["role"] != "tool":
                delta = {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": f"call_{len(requests)}", "type": "function",
                    "function": {"name": "write", "arguments": json.dumps({
                        "path": "marker.txt", "content": f"request {len(requests)}",
                    })},
                }]}
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "pong"}
                finish = "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            ):
                chunk.update(id=f"response_{len(requests)}", object="chat.completion.chunk", model="test-model")
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("sandbox disallows binding a localhost test server")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        models = tmp_path / "models.json"
        models.write_text(json.dumps({"providers": {"hillclimb-test": {
            "baseUrl": f"http://127.0.0.1:{server.server_port}/v1",
            "api": "openai-completions", "apiKey": "test-local",
            "models": [{"id": "test-model"}],
        }}}))
        agent = PiCliAgent(auth="api-key", models_file=models)
        parent_dir = tmp_path / "search" / "candidates" / "c001"
        request = OperatorRequest(
            operator="draft", prompt="Write marker.txt then reply pong.",
            candidate_dir=parent_dir, timeout_s=30, model="hillclimb-test/test-model",
            sampling={"temperature": 0.7, "top_k": 40},
        )
        parent = agent.invoke(request)
        assert parent.ok, parent.error_message
        assert parent.session_id
        assert parent.total_tokens == 30
        assert parent_dir.joinpath("marker.txt").read_text() == "request 1"
        assert requests[0]["temperature"] == 0.7
        assert requests[0]["top_k"] == 40
        assert isinstance(requests[0]["top_k"], int)

        child_dir = parent_dir.with_name("c002")
        child = agent.invoke(request.model_copy(update={
            "operator": "debug", "candidate_dir": child_dir,
            "resume_session_id": parent.session_id,
        }))
        assert child.ok, child.error_message
        assert child.session_id != parent.session_id
        assert child_dir.joinpath("marker.txt").read_text() == "request 3"
        assert parent_dir.joinpath("marker.txt").read_text() == "request 1"
        assert sum(m["role"] == "user" for m in requests[2]["messages"]) == 2

        rejected = agent.preflight(request.model_copy(update={
            "candidate_dir": tmp_path / "preflight", "sampling": {"temperature": 0.99},
        }))
        assert not rejected.ok
        assert rejected.error_kind == "error"
        assert "temperature is deprecated" in rejected.error_message
        assert not requests[-1].get("tools")
        assert json.loads(Path(rejected.raw_output_path).read_text())["returncode"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
