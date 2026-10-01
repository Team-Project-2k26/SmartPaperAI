"""
conftest.py — shared pytest fixtures and path setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is always on sys.path when running tests
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
