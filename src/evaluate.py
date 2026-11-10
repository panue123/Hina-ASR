"""Use a held-out test set AFTER selecting the model on validation."""
import argparse
import csv
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from transformers import pipeline
from common import normalize, read_manifest

p = argparse.ArgumentParser()
p.add_argument("--model", default="vinai/PhoWhisper-base")
p.add_argument("--manifest", required=True)
p.add_argument("--output", default="outputs/predictions.csv")
args = p.parse_args()
rows = read_manifest(args.manifest)
pipe = pipeline("automatic-speech-recognition", model=args.model,
                device=0 if torch.cuda.is_available() else -1)
predictions = []
for row in rows:
    array, sr = sf.read(row["audio"], dtype="float32")
    result = pipe({"array": array, "sampling_rate": sr},
                  generate_kwargs={"language": "vietnamese", "task": "transcribe"})
    predictions.append({"audio": row["audio"], "reference": row["text"],
                        "prediction": result["text"], "speaker_id": row["speaker_id"]})
refs = [normalize(x["reference"]) for x in predictions]
hyps = [normalize(x["prediction"]) for x in predictions]
print(f"N={len(predictions)} WER={jiwer.wer(refs, hyps):.4f} CER={jiwer.cer(refs, hyps):.4f}")
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["audio", "reference", "prediction", "speaker_id"])
    w.writeheader()
    w.writerows(predictions)
print("Predictions:", out)
