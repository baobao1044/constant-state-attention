"""Report any tracked-ish file whose working-tree bytes contain CRLF.

MANIFEST.json stores sha256 of working-tree bytes, and CI checks it out on Linux where
line endings are LF.  A single CRLF file would make the manifest check fail there.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache", "data"}
BINARY_SUFFIX = (".pt", ".png", ".jpg")

crlf, total = [], 0
for p in sorted(ROOT.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        continue
    if rel.suffix in BINARY_SUFFIX:
        continue
    total += 1
    b = p.read_bytes()
    if bytes([13, 10]) in b:
        crlf.append((str(rel), b.count(bytes([13, 10]))))

print(f"scanned {total} text files")
if crlf:
    print("CRLF FOUND:")
    for name, n in crlf:
        print(f"  {name}: {n} CRLF lines")
    sys.exit(1)
print("no CRLF anywhere — manifest hashes will match on Linux checkout")
