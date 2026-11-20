from http.server import ThreadingHTTPServer
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from rocketlab.server import Lab, handler_for
from rocketlab.storage import write_json


@pytest.fixture
def server(tmp_path):
    lab = Lab(tmp_path)
    http = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(lab))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield lab, f"http://127.0.0.1:{http.server_port}"
    http.shutdown()
    http.server_close()


def test_dashboard_and_run_api(server):
    lab, url = server
    write_json(lab.root / "test-run/evaluation.json", {"mean":123})
    with urlopen(url) as response:
        assert b"Teach a rocket to land" in response.read()
    with urlopen(url+"/api/runs") as response:
        assert json.load(response)["runs"][0]["evaluation"]["mean"] == 123


def test_cross_origin_cannot_start_process(server):
    lab, url = server
    req = Request(url+"/api/train", data=b"{}", headers={"Origin":"https://evil.example"})
    with pytest.raises(HTTPError) as error:
        urlopen(req)
    assert error.value.code == 403
    assert lab.worker is None


@pytest.mark.parametrize("name", ["..", "../outside", "/tmp", "a/b", None])
def test_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        Lab(tmp_path).directory(name)


def test_invalid_training_request_does_not_launch(server):
    lab, url = server
    req = Request(url+"/api/train", data=b'{"steps":-1}', headers={"Content-Type":"application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(req)
    assert error.value.code == 400
    assert lab.worker is None
