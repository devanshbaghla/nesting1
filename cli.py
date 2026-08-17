#!/usr/bin/env python3
"""Command-line entry point (the web app is `uvicorn app.main:app`)."""
import sys
from app.core.nest_base import main
if __name__ == "__main__":
    sys.exit(main())
