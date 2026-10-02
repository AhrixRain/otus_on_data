#!/usr/bin/env python
"""Execute a Jupyter notebook in place (nbclient), without nbconvert.

Usage:
    python scripts_joint/execute_notebook.py path/to/notebook.ipynb [--timeout 3600]

The executed notebook is written back with its outputs, so it opens with the
results already rendered. Requires nbformat + nbclient + ipykernel.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def execute(path: Path, *, timeout: int, kernel_name: str) -> int:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    started = time.time()
    client.execute()
    nbformat.write(notebook, path)
    print(f"executed {path} in {time.time() - started:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    return execute(args.notebook.expanduser().resolve(), timeout=args.timeout, kernel_name=args.kernel)


if __name__ == "__main__":
    raise SystemExit(main())
