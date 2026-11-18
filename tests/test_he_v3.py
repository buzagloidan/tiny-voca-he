"""The released he-v3 model: it loads, speaks, exports, and the browser tokenizer agrees.

Needs ``renikud-plus`` (its G2P model downloads on first use); the ONNX case also
needs ``onnx`` and ``onnxruntime``; the browser case needs Node.js.
"""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import torch

from tiny_voca import text
from tiny_voca.infer import load_model, speak_text

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "he-v3"
try:
    import renikud_onnx  # noqa: F401
    HAVE_RENIKUD = True
except ImportError:
    HAVE_RENIKUD = False
try:
    import onnxruntime  # noqa: F401
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False


@unittest.skipUnless(HAVE_RENIKUD, "needs renikud-plus")
class ReleasedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text.set_frontend("renikud")
        cls.model = load_model(MODEL / "inference.pt")

    def test_the_checkpoint_matches_the_default_front_end(self):
        metadata = json.loads((MODEL / "tiny-voca-he.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["frontend_fingerprint"], text.frontend_fingerprint())
        self.assertEqual(metadata["voices"], ["AH"])

    def test_it_speaks_a_sentence(self):
        wave, spoken = speak_text(self.model, "שלום, מה שלומך היום?", "AH", 1.0, 0.0)
        seconds = len(wave) / self.model.config.sample_rate
        self.assertTrue(torch.isfinite(wave).all())
        self.assertTrue(0.8 < seconds < 4, seconds)
        self.assertGreater(float(wave.abs().max()), 0.05)
        self.assertEqual(len(spoken), 1)

    @unittest.skipUnless(HAVE_ORT, "needs onnx and onnxruntime")
    def test_the_released_onnx_model_speaks(self):
        import numpy as np
        session = onnxruntime.InferenceSession(str(MODEL / "tiny-voca-he.onnx"))
        phones = text.phonemize(text.to_ipa("בוקר טוב"))
        feeds = {"tokens": np.array([phones.ids], dtype=np.int64), "stress": np.array([phones.stress], dtype=np.int64),
                 "voice": np.array([0], dtype=np.int64), "speed": np.array([1.0], dtype=np.float32),
                 "pitch_shift": np.array([0.0], dtype=np.float32), "pitch_range": np.array([1.0], dtype=np.float32)}
        audio = session.run(["audio"], feeds)[0]
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(audio.size / 24000, 0.3)


@unittest.skipUnless(shutil.which("node"), "needs Node.js")
class BrowserTokenizerTests(unittest.TestCase):
    def test_tokenize_js_gives_the_ids_and_stress_of_the_python_front_end(self):
        presets = json.loads((ROOT / "web" / "presets.json").read_text(encoding="utf-8"))
        script = """
const fs = require("fs");
eval(fs.readFileSync(process.argv[1], "utf8") + ";globalThis.tokenizeIpa = tokenizeIpa;");
const presets = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let bad = 0, total = 0;
for (const p of presets) for (const s of p.sentences) {
  const got = tokenizeIpa(s.ipa); total++;
  if (JSON.stringify(got.ids) !== JSON.stringify(s.ids) || JSON.stringify(got.stress) !== JSON.stringify(s.stress)) bad++;
}
console.log(JSON.stringify({bad, total}));
"""
        result = subprocess.run(["node", "-e", script, str(ROOT / "web" / "tokenize.js"), str(ROOT / "web" / "presets.json")],
                                capture_output=True, text=True, check=True)
        counts = json.loads(result.stdout)
        self.assertGreater(counts["total"], 10)
        self.assertEqual(counts["bad"], 0)


if __name__ == "__main__":
    unittest.main()
