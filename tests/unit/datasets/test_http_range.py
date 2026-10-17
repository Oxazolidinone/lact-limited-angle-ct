"""HTTP Range helpers of the dataset downloaders, against a local server that honours Range requests."""
from __future__ import annotations

import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lact.datasets.detect.zenodo import read_member, zip_index
from lact.datasets.http_range import RemoteFile, download_resumable


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("slice00001/mode2/sinogram.tif", bytes(range(256)) * 400)
        z.writestr("slice00001/mode1/sinogram.tif", b"not needed")
        z.writestr("slice00001/mode2/flat1.tif", b"\x01\x02" * 1000)
    return buf.getvalue()


@pytest.fixture(scope="module")
def server():
    files = {"/data.zip": _zip_bytes()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _body(self) -> tuple[int, bytes, dict]:
            data = files[self.path]
            rng = self.headers.get("Range")
            if not rng:
                return 200, data, {}
            a, _, b = rng.removeprefix("bytes=").partition("-")
            lo, hi = int(a), int(b) if b else len(data) - 1
            return 206, data[lo:hi + 1], {"Content-Range": f"bytes {lo}-{hi}/{len(data)}"}

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(files[self.path])))
            self.end_headers()

        def do_GET(self):
            status, body, extra = self._body()
            self.send_response(status)
            for k, v in {"Content-Length": str(len(body)), **extra}.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", files
    httpd.shutdown()


def test_remote_file_reads_ranges_and_zip_members(server, tmp_path):
    base, files = server
    data = files["/data.zip"]
    r = RemoteFile(base + "/data.zip", retries=1, timeout=5)
    assert r.size == len(data) and r.get(10, 19) == data[10:20] and r.get(len(data) - 3, 10**9) == data[-3:]
    idx = zip_index(r, tmp_path / "index.json", keep=lambda name: "/mode2/" in name)
    assert set(idx) == {"slice00001/mode2/sinogram.tif", "slice00001/mode2/flat1.tif"}
    assert (tmp_path / "index.json").is_file() and zip_index(r, tmp_path / "index.json") == idx     # cached
    full = bytes(range(256)) * 400
    assert read_member(r, idx["slice00001/mode2/sinogram.tif"]) == full
    assert read_member(r, idx["slice00001/mode2/sinogram.tif"], max_out=1000) == full[:1000]       # leading bytes only


def test_download_resumes_a_partial_file(server, tmp_path):
    base, files = server
    data = files["/data.zip"]
    dst = tmp_path / "out" / "data.zip"
    dst.parent.mkdir()
    dst.with_name("data.zip.partial").write_bytes(data[:100])                 # an interrupted earlier download
    assert download_resumable(base + "/data.zip", dst, len(data), retries=2, timeout=5).read_bytes() == data
    assert not dst.with_name("data.zip.partial").exists()
    assert download_resumable(base + "/data.zip", dst, len(data)) == dst                       # complete: no request
