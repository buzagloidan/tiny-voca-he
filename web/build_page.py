"""Write web/index.html: page.template.html with tokenize.js and presets.json inlined.

    python web/build_page.py

Publish index.html with model-1.b64.txt, model-2.b64.txt and ort-wasm-simd-threaded.wasm
next to it (see build_presets.py for the presets, tiny_voca.export for the model).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

page = (HERE / "page.template.html").read_text(encoding="utf-8")
tokenize = (HERE / "tokenize.js").read_text(encoding="utf-8")
presets = json.dumps(json.loads((HERE / "presets.json").read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))
presets = presets.replace("</", "<\\/")
for marker, value in (("/*TOKENIZE*/", tokenize), ("/*PRESETS*/", presets)):
    if page.count(marker) != 1:
        raise SystemExit(f"{marker} must appear once in the template")
    page = page.replace(marker, value)
(HERE / "index.html").write_text(page, encoding="utf-8")
print(f"wrote {HERE / 'index.html'} ({len(page.encode()):,} bytes)")
