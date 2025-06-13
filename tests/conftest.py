# tests/conftest.py
import sys, os

# project root (one level up from tests/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# package folder
PKG  = os.path.join(ROOT, "package")

# ensure both are on sys.path
for p in (ROOT, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)
