"""
execute_all.py  (superseded)
----------------------------
The old monolithic generator/pipeline/eval script is retired: it fabricated a
synthetic "trained" model and mock COLMAP data inside the real output paths.
The genuine replacement is scripts/run_all.py (unit tests + a geometrically
consistent fixture study + printed ablation), and the real multi-view
pipeline is `python -m integration.run_semantic_pipeline`.

This shim keeps old muscle-memory commands working by delegating.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    print("execute_all.py is superseded -- delegating to scripts/run_all.py "
          "(tests + real fixture study).\n", file=sys.stderr)
    from scripts.run_all import main
    main()
