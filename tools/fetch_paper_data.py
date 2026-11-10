"""Download the upstream LeafByte test images used as fixtures / accuracy ground truth.

The 30 paper photos and the ``leafWithScale`` fixture belong to the upstream
LeafByte repository (GPL-3.0), so they are **not** committed here; this script
fetches them on demand with plain HTTPS (no git, no extra dependencies).

Usage::

    python tools/fetch_paper_data.py           # fetch what is missing
    python tools/fetch_paper_data.py --force   # re-download everything

Targets:
    data/paper/images/*.png       30 paper photos  (accuracy ground truth)
    data/fixtures/leafWithScale.png  upstream thresholding fixture
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_BASE = ("https://raw.githubusercontent.com/TheBeruriahIncident/leafbyte/main/"
            "iOS/LeafByteTests/Images")

#: The 30 photos shipped with the upstream test bundle.
PAPER_FILES = [
    "AI2", "CA1", "CO1", "CO2", "CO5", "EA2", "GB1", "GB2", "GB4", "GB5",
    "GB6", "IE1", "MP5", "OR2", "PB2", "SH2", "SH4", "SH5", "SP5", "SS1",
    "SS2", "SS3", "SS4", "TH3", "TH4", "TH5", "TM1", "TM2", "TM4", "TM5",
]

USER_AGENT = "auto-leafbyte-fetch/1.0 (+https://github.com)"


def _download(url: str, target: Path, retries: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if len(payload) < 1000:
                raise ValueError(f"suspiciously small response ({len(payload)} bytes)")
            target.write_bytes(payload)
            print(f"  ok   {target.relative_to(ROOT)}  ({len(payload) / 1e6:.2f} MB)")
            return
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"could not download {url}: {last_error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    paper_dir = ROOT / "data" / "paper" / "images"
    fixture_dir = ROOT / "data" / "fixtures"

    print(f"upstream LeafByte test images -> {paper_dir}")
    missing = 0
    for name in PAPER_FILES:
        target = paper_dir / f"{name}.png"
        if target.exists() and not args.force:
            continue
        missing += 1
        _download(f"{RAW_BASE}/Paper%20Data/{name}.png", target)

    fixture = fixture_dir / "leafWithScale.png"
    if not fixture.exists() or args.force:
        missing += 1
        _download(f"{RAW_BASE}/leafWithScale.png", fixture)

    if missing == 0:
        print("nothing to do - all fixtures already present")
    else:
        print(f"done: {missing} file(s) downloaded")
    print("\nnext: python -m unittest discover -s tests -t . -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
