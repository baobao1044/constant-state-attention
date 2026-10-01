"""Regenerate MANIFEST.json (sha256 of every tracked artifact).

    python tools/make_manifest.py
    python tools/make_manifest.py --check    # CI: fail if stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache", "data"}
SKIP_SUFFIX = (".b64", ".pyc")
MANIFEST = ROOT / "MANIFEST.json"


def collect() -> dict:
    artifacts = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if str(rel) == "MANIFEST.json" or str(rel).endswith(SKIP_SUFFIX):
            continue
        data = p.read_bytes()
        artifacts.append({"path": str(rel).replace("\\", "/"), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})

    checkpoints = {}
    for art in artifacts:
        if art["path"].startswith("weights/"):
            checkpoints[Path(art["path"]).stem] = {"file": art["path"],
                                                   "bytes": art["bytes"],
                                                   "sha256": art["sha256"]}
    return {"repo": "constant-state-attention",
            "note": "sha256 of every artifact; verify with tests/test_checkpoint_load.py",
            "artifacts": artifacts, "checkpoints": checkpoints}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    new = json.dumps(collect(), indent=2, sort_keys=False) + "\n"
    if args.check:
        old = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
        if old != new:
            print("MANIFEST.json is stale — run: python tools/make_manifest.py",
                  file=sys.stderr)
            sys.exit(1)
        print("MANIFEST.json is up to date")
        return

    MANIFEST.write_text(new, encoding="utf-8", newline="\n")
    n = len(collect()["artifacts"])
    print(f"wrote {MANIFEST} ({n} artifacts)")


if __name__ == "__main__":
    main()
