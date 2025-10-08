# tests/conftest.py
# Ensure repo-root is importable when running `pytest` from anywhere.

import os
import sys

_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.dirname(_THIS)  # one level up from tests/
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
