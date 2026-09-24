"""Download datasets into data/ and verify their MD5 checksums.

Usage: python scripts/fetch_data.py [key ...]
Default is the step-classifier sets (D15): uwash + zhang_who. ablutomania (~4.4 GB) and
ocdetect (31.6 GB) are fetched only when named. Files already present with the right
checksum are skipped; a zip is extracted next to itself after a fresh download.
Datasets are never committed or redistributed (docs/DATASETS.md).
"""

import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA = Path(__file__).resolve().parent.parent / "data"
GDRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
ZENODO = "https://zenodo.org/api/records/{}/files/{}/content"
RDR = "https://rdr.kuleuven.be/api"


def _rdr_files(doi: str, version: str) -> list[tuple[str, str, str]]:
    """List a KU Leuven RDR (Dataverse) dataset version: (path, url, md5) per file."""
    r = requests.get(
        f"{RDR}/datasets/:persistentId/versions/{version}/files",
        params={"persistentId": doi},
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for f in r.json()["data"]:
        df = f["dataFile"]
        assert df["checksum"]["type"] == "MD5", df["checksum"]
        rel = "/".join(p for p in (f.get("directoryLabel"), df["filename"]) if p)
        url = f"{RDR}/access/datafile/{df['id']}?format=original"
        out.append((rel, url, df["checksum"]["value"]))
    return out


def _zenodo(record: str, files: dict[str, str]) -> list[tuple[str, str, str]]:
    return [(name, ZENODO.format(record, name), md5) for name, md5 in files.items()]


# key -> (folder under data/, lazy list of (relative path, url, md5)).
SOURCES = {
    "uwash": (
        "uwash",
        lambda: [
            (
                "Dataset_raw.zip",
                GDRIVE.format("1ZRdRiwXp4xbFUWIIjIQ0OEK6gK0cwODN"),
                "4088db4cdcb5f41b9951462d1d7a082f",
            )
        ],
    ),
    # Pinned to version 1.0; checksums come from the repository's file listing.
    "zhang_who": ("zhang-who", lambda: _rdr_files("doi:10.48804/XHPPC7", "1.0")),
    "ablutomania": (
        "ablutomania",
        lambda: _zenodo(
            "20094015",
            {
                "readme.zip": "039ab75be56b30b7a8a44473fcdd0cbc",
                "handwashing-2019.zip": "6c30635b97c00aa42d9225f35545118f",
                "handwashing-2020.zip": "f88da6f7afa21e986a7707ebe3149fe9",
                "hwseminar-2019.zip": "92d1c90d02e2b16217dd4f178bef828b",
            },
        ),
    ),
    "ocdetect": (
        "ocdetect",
        lambda: _zenodo("13924901", {"OCDetect_dataset.zip": "897d665e6f9c6f5fbd302e08362baad0"}),
    ),
}
DEFAULT = ("uwash", "zhang_who")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, md5: str) -> bool:
    """Download ``url`` to ``dest`` and verify it. Return False if it was already there."""
    if dest.exists() and md5sum(dest) == md5:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.md5()
    # ponytail: no resume; a dropped 31 GB ocdetect download restarts from zero.
    with requests.get(url, stream=True, timeout=60) as r, open(tmp, "wb") as f:
        r.raise_for_status()
        size = int(r.headers.get("content-length", 0))
        with tqdm(total=size, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
                bar.update(len(chunk))
    if h.hexdigest() != md5:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {dest}: got {h.hexdigest()}, want {md5}")
    tmp.replace(dest)
    return True


def main(keys: list[str]) -> int:
    unknown = set(keys) - set(SOURCES)
    if unknown:
        print(f"unknown dataset(s) {sorted(unknown)}; choose from {sorted(SOURCES)}")
        return 2
    for key in keys or DEFAULT:
        folder, files = SOURCES[key]
        root = DATA / folder
        for rel, url, md5 in files():
            dest = root / rel
            if fetch(url, dest, md5):
                if dest.suffix == ".zip":
                    with zipfile.ZipFile(dest) as z:
                        z.extractall(dest.parent)
                print(f"OK   {key}: {rel}")
            else:
                print(f"skip {key}: {rel} (checksum matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
