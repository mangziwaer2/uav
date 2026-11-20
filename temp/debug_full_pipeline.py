"""Preserved debug filename; delegates to the single active entry."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    runpy.run_path(str(root/'src/run_experiment.py'), run_name='__main__')
