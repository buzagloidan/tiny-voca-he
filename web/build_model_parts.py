"""Write the files the demo page loads next to index.html.

    python web/build_model_parts.py [path/to/tiny-voca-he.onnx]

Splits the ONNX model (default: models/he-v3/tiny-voca-he.onnx) into
model-1.b64.txt and model-2.b64.txt, and downloads onnxruntime-web's
ort-wasm-simd-threaded.wasm from jsDelivr (the version the page uses). The
page lists each file's size in its FILES constant; this script checks they
match, which they do for he-v3.

The model is split into base64 text only so the page can be hosted where raw
binary files are not served. On an ordinary web server you could load the
.onnx directly instead.
"""

import base64
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
model = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "models" / "he-v3" / "tiny-voca-he.onnx"
template = (HERE / "page.template.html").read_text(encoding="utf-8")
ort = re.search(r'const ORT_DIST = "([^"]+)"', template).group(1)
expected = dict(re.findall(r'\["([\w.-]+)", (\d+)\]', template))

data = model.read_bytes()
half = len(data) // 2 // 3 * 3              # a multiple of 3: each part is valid base64 on its own
for name, part in (("model-1.b64.txt", data[:half]), ("model-2.b64.txt", data[half:])):
    (HERE / name).write_bytes(base64.b64encode(part))

wasm = HERE / "ort-wasm-simd-threaded.wasm"
if not wasm.exists():
    with urllib.request.urlopen(ort + wasm.name) as response:
        wasm.write_bytes(response.read())

for name, size in expected.items():
    actual = (HERE / name).stat().st_size
    note = "" if actual == int(size) else f"  <- the page expects {int(size):,}: update FILES in page.template.html"
    print(f"{name}: {actual:,} bytes{note}")
