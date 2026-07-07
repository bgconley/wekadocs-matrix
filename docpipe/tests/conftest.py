"""Make the ``docpipe`` package importable without installing it.

The project root (the directory that CONTAINS the ``docpipe`` package) is added
to sys.path, mirroring the repo's ``pythonpath = .`` convention.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
