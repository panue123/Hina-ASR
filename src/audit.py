import argparse
from common import check_splits, read_manifest

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--val", required=True)
p.add_argument("--test", required=True)
args = p.parse_args()
splits = {name: read_manifest(getattr(args, name)) for name in ("train", "val", "test")}
check_splits(splits)
for name, rows in splits.items():
    print(f"{name}: {len(rows)} clips, {len({r['speaker_id'] for r in rows})} distinct speakers")
print("PASS: valid WAV manifests and no cross-split speaker/file leakage")
