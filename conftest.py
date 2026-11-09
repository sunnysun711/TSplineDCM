"""Make the repo root importable for pytest and any script run from elsewhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
