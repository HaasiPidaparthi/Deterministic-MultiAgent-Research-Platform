#!/usr/bin/env python3
"""Root wrapper for RAG population utilities.

This script delegates to src/engine/tools/rag_populator.py with a runtime path
adjustment so it can be executed from the repository root.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from engine.tools.rag_populator import main

if __name__ == "__main__":
    sys.exit(main())
