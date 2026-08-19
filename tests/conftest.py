"""
pytest conftest.py — ensures project root is on sys.path for all test imports.
This allows 'from src.xxx import ...' in integration tests to resolve correctly.
"""

import sys
from pathlib import Path

# Insert project root (parent of 'tests/') onto path so `src` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
