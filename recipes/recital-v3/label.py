"""RenikudPlus IPA for the converted recital utterances -> dataset/manifest.jsonl.

Run with the tiny-voca-he environment after ``build.py convert``. Rows the front end
cannot fully map (unknown symbols) are dropped and listed in work/label-dropped.jsonl.
Splits are by document (several readers read the same documents), about 300
utterances each for validation and test, the rest train. A held-out sentence
that also occurs in train under another document title is dropped.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from tiny_voca import text as frontend  # noqa: E402

DATASET = HERE / "dataset"
HELD_OUT = 300
PROVENANCE = {"origin": "ivrit-ai/crowd-recital, voice-converted to one target voice with kNN-VC",
              "source_license": "ivrit.ai License v2 (CC BY 4.0 for training AI models; credit ivrit.ai)",
              "source_url": "https://huggingface.co/datasets/ivrit-ai/crowd-recital", "category": "recital"}


def main():
    frontend.set_frontend("renikud")
    rows = [json.loads(line) for line in open(HERE / "work" / "cut.jsonl", encoding="utf-8")]
    rows = [r for r in rows if "id" in r and (DATASET / "audio" / f"{r['id']}.flac").exists()]
    def group(row):
        return row["title"] or row["session"]
    order = sorted({group(r) for r in rows}, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    split, counts = {}, {"research_test": 0, "research_validation": 0}
    per_group = {}
    for r in rows:
        per_group[group(r)] = per_group.get(group(r), 0) + 1
    for name in order:
        target = next((n for n in counts if counts[n] < HELD_OUT), "research_main")
        split[name] = target
        if target in counts:
            counts[target] += per_group[name]
    kept, dropped = [], []
    train_texts = {r["text"] for r in rows if split[group(r)] == "research_main"}
    seen = {}
    for number, row in enumerate(rows, 1):
        name = split[group(row)]
        if name != "research_main" and (row["text"] in train_texts or seen.get(row["text"], name) != name):
            dropped.append({**row, "reason": "sentence also in another split"})
            continue
        seen[row["text"]] = name
        ipa = frontend.to_ipa(row["text"])
        result = frontend.phonemize(ipa)
        if result.unknown:
            dropped.append({**row, "ipa": ipa, "unknown": sorted(set(result.unknown))})
            continue
        kept.append({"id": f"recital_{row['id']}", "audio": f"audio/{row['id']}.flac", "split": name,
                     "voice_id": "AH", "text": row["text"], "ipa": ipa, "seconds": row["seconds"],
                     "source_id": row["session"], "reader": row["reader"], "title": row["title"], **PROVENANCE})
        if number % 1000 == 0:
            print(f"labelled {number}/{len(rows)}", flush=True)
    with open(DATASET / "manifest.jsonl", "w", encoding="utf-8") as out:
        out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    with open(HERE / "work" / "label-dropped.jsonl", "w", encoding="utf-8") as out:
        out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in dropped)
    summary = {}
    for r in kept:
        item = summary.setdefault(r["split"], [0, 0.0])
        item[0] += 1
        item[1] += r["seconds"] / 3600
    print(json.dumps({"kept": len(kept), "dropped": len(dropped),
                      "splits": {k: {"utterances": n, "hours": round(h, 2)} for k, (n, h) in summary.items()}}))


if __name__ == "__main__":
    main()
