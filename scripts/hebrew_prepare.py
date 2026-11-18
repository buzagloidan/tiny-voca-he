"""Acoustic preparation for the Hebrew dataset, in parallel.

Reads ``DATASET/manifest.jsonl`` (written by ``recipes/recital-v3/label.py``) and
writes the same files as ``tiny_voca.prepare`` (``<split>.jsonl``,
``<split>.preparation.json``, ``features.json`` and one ``.pt`` per clip), so
``tiny_voca.annotate`` can read them unchanged. The record ``text`` is the
manifest's ``ipa`` column; the Hebrew sentence is kept in the provenance.

Splits: research_main -> train, research_validation -> validation,
research_test -> test.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch

from tiny_voca.config import Config
from tiny_voca.journal import atomic_json, file_hash
from tiny_voca.prepare import MIN_FRAMES, extract

SPLITS = {"research_main": "train", "research_validation": "validation", "research_test": "test"}
KEEP = ("text_id", "origin", "source_id", "source_license", "source_url", "model", "reference_audio", "category")


def work(job):
    row, root, output, config = job
    split = SPLITS[row["split"]]
    name = hashlib.sha256(row["id"].encode()).hexdigest() + ".pt"
    target = output / split / name
    try:
        wave, rate = sf.read(root / row["audio"], dtype="float32")
        if wave.ndim != 1:
            raise ValueError("expected mono audio")
        if rate != config.sample_rate:
            raise ValueError(f"expected {config.sample_rate} Hz; got {rate}")
        frames = math.ceil(len(wave) / config.hop)
        if not MIN_FRAMES <= frames <= config.max_frames:
            raise ValueError("audio duration outside admission limits")
        if not target.exists():
            sample = extract(wave, config)
            sample.update(speaker=config.speakers.index(row["voice_id"]))
            temporary = target.with_suffix(".tmp")
            torch.save(sample, temporary)
            os.replace(temporary, target)
    except Exception as error:
        return split, None, {"id": row["id"], "reason": f"{type(error).__name__}: {error}"}
    text = row["ipa"].strip()
    entry = {"id": row["id"], "path": f"{split}/{name}", "acoustic": config.acoustic_fingerprint,
             "text_hash": hashlib.sha256(text.encode()).hexdigest(), "text": text, "speaker": row["voice_id"],
             "source_file": str(root / "manifest.jsonl"), "frames": frames,
             "provenance": {**{k: row[k] for k in KEEP if k in row}, "hebrew": row["text"]}}
    return split, entry, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Directory with manifest.jsonl and the audio it lists")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit", type=int, help="Only the first N rows of each split (smoke test)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = Config()
    manifest = args.dataset / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        counts, kept = {}, []
        for row in rows:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
            if counts[row["split"]] <= args.limit:
                kept.append(row)
        rows = kept
    for split in set(SPLITS.values()):
        (args.output / split).mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "features.json", {"fingerprint": config.acoustic_fingerprint, "fields": config.acoustic_config})
    results = {split: ([], []) for split in SPLITS.values()}
    jobs = [(row, args.dataset, args.output, config) for row in rows]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for count, (split, entry, rejected) in enumerate(pool.map(work, jobs, chunksize=8), 1):
            (results[split][0] if entry else results[split][1]).append(entry or rejected)
            if count % 500 == 0:
                print(json.dumps({"prepared": count, "of": len(jobs)}), flush=True)
    identity = {"acoustic": config.acoustic_fingerprint, "sources": [{"path": str(manifest.resolve()), "sha256": file_hash(manifest)}],
                "text_column": "ipa", "speaker_column": "voice_id", "speakers": list(config.speakers), "limit": args.limit,
                "admission": {"max_frames": config.max_frames, "min_frames": MIN_FRAMES}}
    for split, (accepted, rejected) in results.items():
        accepted.sort(key=lambda r: r["id"])
        for index, entry in enumerate(accepted):
            entry["record_key"] = f"0:{index}"
        for name, payload in ((split + ".jsonl", accepted), (split + ".rejected.jsonl", rejected)):
            (args.output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload), encoding="utf-8")
        atomic_json(args.output / (split + ".preparation.json"),
                    {"identity": identity, "complete": True, "accepted": len(accepted), "rejected": len(rejected)})
        hours = sum(r["frames"] for r in accepted) * config.hop / config.sample_rate / 3600
        print(json.dumps({"split": split, "accepted": len(accepted), "rejected": len(rejected), "hours": round(hours, 3)}), flush=True)


if __name__ == "__main__":
    main()
