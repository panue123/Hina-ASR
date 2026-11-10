"""Shared loading/evaluation utilities for short Vietnamese command audio."""
import csv
import unicodedata
from pathlib import Path

import soundfile as sf

REQUIRED = {"audio_path", "text", "speaker_id", "session_id"}


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if not row["audio_path"] or not row["text"].strip():
                raise ValueError(f"{path.name}:{line}: audio_path/text must not be empty")
            if not row["speaker_id"] or not row["session_id"]:
                raise ValueError(f"{path.name}:{line}: speaker_id/session_id required")
            audio = (path.parent / row["audio_path"]).resolve()
            if not audio.is_file():
                raise FileNotFoundError(f"{path.name}:{line}: {audio}")
            info = sf.info(str(audio))
            if info.duration <= 0 or info.duration > 30:
                raise ValueError(f"{audio}: clip must be between 0 and 30 seconds")
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(f"{audio}: convert to mono, 16 kHz WAV first")
            rows.append({"audio": str(audio), "text": row["text"].strip(),
                         "speaker_id": row["speaker_id"], "session_id": row["session_id"]})
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def normalize(text):
    # Reuse exactly this normalization for before/after tests; keep Vietnamese accents.
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def check_splits(splits):
    speakers = {name: {r["speaker_id"] for r in rows} for name, rows in splits.items()}
    for name, people in speakers.items():
        for other, others in speakers.items():
            if name < other and people & others:
                raise ValueError(f"Speaker leakage between {name} and {other}: {people & others}")
    paths = {name: {r["audio"] for r in rows} for name, rows in splits.items()}
    for name, files in paths.items():
        for other, other_files in paths.items():
            if name < other and files & other_files:
                raise ValueError(f"Duplicate audio across {name} and {other}")
