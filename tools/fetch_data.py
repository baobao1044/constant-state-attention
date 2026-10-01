"""Download the tiny-shakespeare corpus into ``data/``.

    python tools/fetch_data.py
    python tools/fetch_data.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import DATA  # noqa: E402

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")
SHA256 = None  # not pinned upstream; recorded after download instead
DEST = DATA / "tiny_shakespeare.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and not args.force:
        data = DEST.read_bytes()
        print(f"{DEST} already present ({len(data):,} bytes, "
              f"sha256 {hashlib.sha256(data).hexdigest()[:16]})")
        return

    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
    DEST.write_bytes(data)
    print(f"wrote {DEST} ({len(data):,} bytes, "
          f"sha256 {hashlib.sha256(data).hexdigest()})")


if __name__ == "__main__":
    main()
