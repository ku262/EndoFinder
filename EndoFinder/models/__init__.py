import sys
from pathlib import Path
# Make sure EndoFinder is in PYTHONPATH.
base_path = str(Path(__file__).resolve().parent.parent.parent)
if base_path not in sys.path:
    sys.path.append(base_path)