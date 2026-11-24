r"""Run the upload checks on files on this computer, without uploading them.

    python -m scripts.check_upload "C:\Users\Ibrahim\ncert"
    python -m scripts.check_upload notes.pdf "haloalkanes and haloarenes.pdf"

Folders are searched for PDFs, .txt and .md files. Each file prints OK or
the reason a student would be told when uploading it. Useful to check the
checks do not refuse real study material.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ai.ingestion.extract import SUPPORTED_SUFFIXES
from ai.ingestion.scan import UploadRefused, check


def _files(args: list[str]):
    """(file, name to show): inside a folder, the path from that folder,
    so a file in a subfolder can be found again."""
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield p, str(p.relative_to(path))
        else:
            yield path, path.name


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    refused = 0
    for path, shown in _files(argv):
        try:
            check(path)
            print(f"  OK       {shown}")
        except UploadRefused as exc:
            refused += 1
            print(f"  REFUSED  {shown}: {exc}")
        except OSError as exc:
            refused += 1
            print(f"  MISSING  {path}: {exc.strerror}")
    print(f"\n  {refused} refused.")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
