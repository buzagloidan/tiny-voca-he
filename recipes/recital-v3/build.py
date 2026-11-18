"""he-v3 training audio: ivrit.ai crowd-recital readings converted to one target voice.

Stages (run in order; each one resumes; convert needs torch, torchaudio and soundfile):

    select    male readers, session quality_score >= 0.9  -> sessions.json
    download  their audio.mka + transcript.aligned.json   -> $RECITAL_DIR/<session>/
    cut       Whisper-aligned segments merged into 1.2-10 s utterances, 16 kHz -> work/src/, cut.jsonl
    convert   kNN-VC (MIT) into the target voice ($TARGET_VOICE_DIR/*.wav), then 24 kHz FLAC -> dataset/audio/

``label.py`` then adds RenikudPlus IPA and writes
dataset/manifest.jsonl for scripts/hebrew_prepare.py.

License: ivrit.ai License v2 permits training AI models (CC BY 4.0: credit
ivrit.ai) but not imitating the people recorded, so only converted audio is
used for training.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
# Where the ivrit.ai crowd-recital sessions are (or will be) downloaded.
RECITAL = Path(os.environ.get("RECITAL_DIR", HERE / "crowd-recital"))
# Clean 16 kHz mono speech of the voice to convert into, ideally 5+ minutes in total.
# You need the speaker's permission to clone their voice.
TARGET_VOICE = Path(os.environ.get("TARGET_VOICE_DIR", HERE / "target-voice"))
WORK = HERE / "work"
DATASET = HERE / "dataset"
SR, OUT_SR = 16000, 24000
MIN_QUALITY, MIN_SEGMENT_PROBABILITY = 0.9, 0.9
MIN_SECONDS, MAX_SECONDS, MAX_GAP = 1.2, 10.0, 0.6
_LATIN = re.compile(r"[A-Za-z]")
_SENTENCE_END = re.compile(r"[.?!]['\"]?$")


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if path.exists() else []


def select(args):
    sessions = []
    for path in sorted(RECITAL.glob("*/metadata.json")):
        meta = json.load(open(path))
        if meta.get("biological_sex") == "male" and meta.get("quality_score", 0) >= MIN_QUALITY:
            sessions.append({"session": meta["session_id"], "reader": meta["user_id"][:8],
                             "seconds": meta["session_duration"], "title": meta.get("document_title")})
    json.dump(sessions, open(HERE / "sessions.json", "w"), ensure_ascii=False, indent=0)
    print(f"{len(sessions)} sessions, {sum(s['seconds'] for s in sessions) / 3600:.1f} h, "
          f"{len({s['reader'] for s in sessions})} readers")


def sessions(args):
    chosen = json.load(open(HERE / "sessions.json"))
    return chosen[: args.limit] if args.limit else chosen


HF = os.environ.get("HF_BIN", "hf")  # the huggingface-hub command-line client


def download(args):
    missing = [s["session"] for s in sessions(args) if not (RECITAL / s["session"] / "audio.mka").exists()]
    print(f"downloading {len(missing)} sessions")
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        patterns = [f"{s}/{name}" for s in batch for name in ("audio.mka", "transcript.aligned.json")]
        subprocess.run([HF, "download", "ivrit-ai/crowd-recital", "--repo-type", "dataset", "--local-dir", str(RECITAL),
                        "--max-workers", "8", "--quiet", "--include", *patterns], check=True, stdout=subprocess.DEVNULL)
        print(f"  {min(start + 100, len(missing))}/{len(missing)}", flush=True)


def utterances(segments, scores):
    """Merge consecutive good segments into utterances of MIN..MAX seconds."""
    groups, current = [], []

    def close():
        if current and MIN_SECONDS <= current[-1]["end"] - current[0]["start"] <= MAX_SECONDS:
            groups.append(list(current))
        current.clear()

    for index, segment in enumerate(segments):
        text = segment["text"].strip()
        probability = scores[index]["probability"] if index < len(scores) else 0
        if not text or _LATIN.search(text) or probability < MIN_SEGMENT_PROBABILITY or segment["end"] <= segment["start"]:
            close()
            continue
        if current and (segment["start"] - current[-1]["end"] > MAX_GAP or segment["end"] - current[0]["start"] > MAX_SECONDS):
            close()
        current.append({"start": segment["start"], "end": segment["end"], "text": text, "index": index})
        if _SENTENCE_END.search(text) and current[-1]["end"] - current[0]["start"] >= 2.0:
            close()
    close()
    return groups


def cut(args):
    import soundfile as sf
    (WORK / "src").mkdir(parents=True, exist_ok=True)
    done = {row["session"] for row in jsonl(WORK / "cut.jsonl")}
    with open(WORK / "cut.jsonl", "a", encoding="utf-8") as out:
        for number, item in enumerate(sessions(args), 1):
            session = item["session"]
            folder = RECITAL / session
            if session in done or not (folder / "audio.mka").exists():
                continue
            meta = json.load(open(folder / "metadata.json"))
            segments = json.load(open(folder / "transcript.aligned.json"))["segments"]
            full = WORK / "full.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(folder / "audio.mka"), "-ac", "1", "-ar", str(SR), str(full)], check=True)
            audio, _ = sf.read(full, dtype="float32")
            rows = []
            for n, group in enumerate(utterances(segments, meta.get("per_segment_quality_scores", []))):
                first, last = group[0]["index"], group[-1]["index"]
                # Pad into the silence around the utterance, never past the middle of a gap.
                before = segments[first - 1]["end"] if first > 0 else 0.0
                after = segments[last + 1]["start"] if last + 1 < len(segments) else len(audio) / SR
                start = max(group[0]["start"] - 0.15, (before + group[0]["start"]) / 2, 0)
                end = min(group[-1]["end"] + 0.2, (group[-1]["end"] + after) / 2, len(audio) / SR)
                clip_id = f"{session}_{n:03d}"
                sf.write(WORK / "src" / f"{clip_id}.wav", audio[int(start * SR):int(end * SR)], SR)
                rows.append({"id": clip_id, "session": session, "reader": item["reader"], "seconds": round(end - start, 3),
                             "text": " ".join(s["text"] for s in group), "title": item["title"]})
            for row in rows or [{"session": session, "empty": True}]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if number % 50 == 0:
                print(f"cut {number} sessions", flush=True)
    rows = [r for r in jsonl(WORK / "cut.jsonl") if "id" in r]
    print(f"{len(rows)} utterances, {sum(r['seconds'] for r in rows) / 3600:.2f} h")


def convert(args):
    import soundfile as sf
    import torch
    import torchaudio
    torch.set_num_threads(args.threads)
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device="cpu")
    cache = WORK / "matching.pt"
    if cache.exists():
        matching = torch.load(cache)
    else:
        # The target-voice clips should already be cut to speech; VAD at the default level empties quiet ones.
        matching = knn.get_matching_set([str(p) for p in sorted(TARGET_VOICE.glob("*.wav"))], vad_trigger_level=0)
        torch.save(matching, cache)
    (DATASET / "audio").mkdir(parents=True, exist_ok=True)
    rows = [r for r in jsonl(WORK / "cut.jsonl") if "id" in r]
    if args.limit:
        keep = {s["session"] for s in sessions(args)}
        rows = [r for r in rows if r["session"] in keep]
    todo = [r for r in rows if not (DATASET / "audio" / f"{r['id']}.flac").exists()]
    print(f"converting {len(todo)} of {len(rows)} utterances", flush=True)
    for number, row in enumerate(todo, 1):
        wave, _ = sf.read(WORK / "src" / f"{row['id']}.wav", dtype="float32")
        query = knn.get_features(torch.from_numpy(wave)[None])
        out = knn.match(query, matching, topk=4)
        out = torchaudio.functional.resample(out[None], SR, OUT_SR)[0].numpy()
        peak = abs(out).max()
        if peak > 0.95:
            out *= 0.95 / peak
        temporary = DATASET / "audio" / f"{row['id']}.tmp.flac"
        sf.write(temporary, out, OUT_SR, format="FLAC")
        temporary.rename(DATASET / "audio" / f"{row['id']}.flac")
        if number % 500 == 0:
            print(f"converted {number}/{len(todo)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("select", "download", "cut", "convert"))
    parser.add_argument("--limit", type=int, help="first N selected sessions only (smoke test)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
