# tiny-voca-he

A tiny Hebrew text-to-speech model: 4.5M parameters, one male voice, 24 kHz audio.
The ONNX file is 21 MB and runs on a CPU, in Python or in the browser
(about 30× faster than real time with onnxruntime-web on a laptop).

מודל הקראה קטן לעברית: 4.5 מיליון פרמטרים, קול גברי אחד, רץ על מעבד רגיל או בדפדפן.

This is a Hebrew port of [tiny-voca](https://github.com/n4ze3m/tiny-voca) by
Muhammed Nazeem. The released model, **he-v3**, was trained from scratch for
104k steps on about 9 hours of [ivrit.ai](https://www.ivrit.ai) crowd-recital
speech, converted into a single voice with kNN-VC.

## What's in the repo

| Path | What |
|---|---|
| `models/he-v3/inference.pt` | PyTorch checkpoint (18 MB) |
| `models/he-v3/tiny-voca-he.onnx` | The same model as ONNX (21 MB), plus `tiny-voca-he.json` with its symbols and settings |
| `src/tiny_voca/` | Model, Hebrew front end, training, alignment, inference, ONNX export |
| `web/` | A static browser demo (Hebrew UI) that runs the ONNX model with onnxruntime-web |
| `recipes/recital-v3/` | How the training audio was built from ivrit.ai crowd-recital |
| `scripts/` | Data preparation, local and RunPod GPU training |

## Quick start

Python 3.10 or newer.

```bash
git clone https://github.com/buzagloidan/tiny-voca-he
cd tiny-voca-he
pip install -e .
```

```bash
tiny-voca-speak --checkpoint models/he-v3/inference.pt --text "שלום, מה שלומך היום?" --output hello.wav
```

Hebrew is written without vowels, so the text first goes through
[RenikudPlus](https://github.com/maxmelichov/RenikudPlus), which guesses the
pronunciation and returns stressed IPA. Its model (about 300 MB) downloads from
Hugging Face the first time you run it. Numbers, times and dates are read out
in words.

Other options: `--text-file long.txt` reads any length of text sentence by
sentence, `--speed 1.1`, `--pitch-shift 0.058` (about one semitone up),
`--pitch-range 1.2` (livelier intonation).

From Python:

```python
import soundfile as sf
from tiny_voca.infer import load_model, speak_text

model = load_model("models/he-v3/inference.pt")
wave, _ = speak_text(model, "בוקר טוב! ישנת טוב בלילה?", "AH")
sf.write("morning.wav", wave.numpy(), 24000)
```

## Running the ONNX model

`pip install -e ".[onnx]"`. The model takes token ids and stress marks from the
front end:

```python
import numpy as np, onnxruntime, soundfile as sf
from tiny_voca import text

session = onnxruntime.InferenceSession("models/he-v3/tiny-voca-he.onnx")
phones = text.phonemize(text.to_ipa("הלחץ שלך היום 130 על 85."))
audio, durations = session.run(None, {
    "tokens": np.array([phones.ids], dtype=np.int64),
    "stress": np.array([phones.stress], dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),
    "speed": np.array([1.0], dtype=np.float32),
    "pitch_shift": np.array([0.0], dtype=np.float32),
    "pitch_range": np.array([1.0], dtype=np.float32),
})
sf.write("bp.wav", audio.reshape(-1), 24000)
```

Feed one sentence at a time. The model was trained on utterances of up to 217
tokens; much longer input drifts. `speak_text` in `src/tiny_voca/infer.py`
shows how to split text and join the audio.

## Browser demo

The demo page runs fully in the browser. RenikudPlus is too large for a web
page, so the demo speaks a set of preset texts whose IPA was computed ahead of
time (`web/presets.json`); `web/tokenize.js` turns IPA into model tokens exactly
like the Python code.

```bash
python web/build_model_parts.py   # model parts + onnxruntime-web wasm next to index.html
python -m http.server -d web 8000
```

Then open http://localhost:8000. To change the texts, edit `web/build_presets.py`
and run it, then `python web/build_page.py`.

## Training your own

The steps that produced he-v3. You need about 10 GB of disk, ffmpeg, and a GPU
for training (he-v3 took under 5 hours on one RTX 4090, about $4 on RunPod).

1. **Target voice.** Put clean 16 kHz mono WAV clips of the voice you want in
   `recipes/recital-v3/target-voice/` (he-v3 used about 20 minutes; kNN-VC
   works with less, but more gives a closer match). Only use a voice you have permission to clone.
2. **Training audio.** If Hugging Face asks, accept the terms of
   [ivrit-ai/crowd-recital](https://huggingface.co/datasets/ivrit-ai/crowd-recital)
   on Hugging Face and log in with `hf auth login`. Then, with `torch`,
   `torchaudio` and `soundfile` installed:
   ```bash
   python recipes/recital-v3/build.py select     # male readers, quality score >= 0.9
   python recipes/recital-v3/build.py download
   python recipes/recital-v3/build.py cut        # 1.2-10 s utterances
   python recipes/recital-v3/build.py convert    # kNN-VC into the target voice, 24 kHz FLAC
   python recipes/recital-v3/label.py            # RenikudPlus IPA, splits by document
   ```
3. **Features and alignment.** `pip install -e ".[prepare,annotate]"`, then
   ```bash
   PYTHONPATH=src python scripts/hebrew_prepare.py --dataset recipes/recital-v3/dataset --output work/features-v3
   PYTHONPATH=src python -m tiny_voca.annotate --features work/features-v3 --output work/annotations-v3 \
       --dataset-revision recital-v3 --device cuda --max-weak-phone-fraction 0.33
   ```
   About three quarters of the clips pass the alignment checks.
4. **Train** for 104k steps, on this machine:
   ```bash
   ANNOTATIONS=work/annotations-v3 DEVICE=cuda scripts/train_local.sh runs/my-voice
   ```
   or on a rented GPU with the RunPod scripts (see `scripts/runpod/README.md`):
   ```bash
   FEATURES=work/features-v3 ANNOTATIONS=work/annotations-v3 scripts/runpod/launch.sh runs/my-voice
   ```
5. **Export** to ONNX:
   ```bash
   PYTHONPATH=src python -m tiny_voca.export --checkpoint runs/my-voice/inference.pt --output tiny-voca-he.onnx
   ```

## Tests

```bash
pip install -e ".[onnx]"
python -m unittest discover -s tests
```

They load the released model, synthesize a sentence in PyTorch and ONNX, and
check that the browser tokenizer agrees with the Python front end. The
upstream English test suite was not carried over.

## Limits

- One voice. The model speaks only the voice it was trained on.
- Pronunciation is only as good as RenikudPlus's guess for unvocalized text.
  Rare words, names and foreign words are sometimes wrong.
- `tiny_voca.evaluate` is still the upstream English evaluation (English
  Whisper and text normalization); it does not score Hebrew.

## Licenses and credits

- **Code**: Apache-2.0 (`LICENSE`). Based on
  [tiny-voca](https://github.com/n4ze3m/tiny-voca) by Muhammed Nazeem,
  Apache-2.0; see `NOTICE` for what changed.
- **he-v3 weights** (`models/he-v3`): CC BY 4.0 (`models/he-v3/LICENSE`).
  If you use them, credit this project and **ivrit.ai**.
- **Training data**: [ivrit.ai crowd-recital](https://huggingface.co/datasets/ivrit-ai/crowd-recital),
  used under the ivrit.ai License v2, which allows training AI models with
  credit to ivrit.ai. Thank you to ivrit.ai and the volunteers who recorded it.
  The recorded readers' own voices are not reproduced: all training audio was
  converted into one target voice first. No audio is included in this repository.
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) (Apache-2.0) for
  Hebrew pronunciation, [kNN-VC](https://github.com/bshall/knn-vc) (MIT) for
  voice conversion, [ONNX Runtime Web](https://onnxruntime.ai) (MIT) for the demo.

Please don't use the model to deceive people into thinking they are hearing a
real person.
