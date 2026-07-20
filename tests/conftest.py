import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Make both repo-root modules (trading/) and research-layer modules importable
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
