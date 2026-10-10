#!/usr/bin/env python3
"""Fetch pinned public policies/configs/robot assets; verify every selected file."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--archive-dir", type=Path, help="Use previously downloaded NAME.tar.gz files")
    args = parser.parse_args()
    sources = json.loads((ROOT / "assets/manifest.json").read_text())["sources"]
    for source in sources:
        destination = ROOT / "assets/public" / source["name"]
        missing = [item for item in source["files"]
                   if not (destination/item["path"]).is_file()
                   or digest(destination/item["path"]) != item["sha256"]]
        if missing and args.verify_only:
            raise SystemExit("Missing or changed assets: " + source["name"] + ": " + ", ".join(x["path"] for x in missing[:3]))
        if missing:
            cache = ROOT / ".cache/downloads"
            cache.mkdir(parents=True, exist_ok=True)
            archive = (args.archive_dir / (source["name"]+".tar.gz") if args.archive_dir
                       else cache / (source["name"]+".tar.gz"))
            if not archive.is_file() or digest(archive) != source["archive_sha256"]:
                if args.archive_dir:
                    raise SystemExit("Missing or changed archive: " + str(archive))
                print("Downloading", source["url"], flush=True)
                partial = archive.with_suffix(".partial")
                try:
                    with urllib.request.urlopen(source["url"], timeout=90) as response, partial.open("wb") as out:
                        shutil.copyfileobj(response, out)
                    if digest(partial) != source["archive_sha256"]:
                        raise ValueError("Archive checksum mismatch")
                    partial.replace(archive)
                finally:
                    if partial.exists():
                        partial.unlink()
            prefix = source["name"] + "-" + source["revision"] + "/"
            with tarfile.open(archive, "r:gz") as tar:
                for item in missing:
                    relative = Path(item["path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe path in manifest")
                    member = tar.getmember(prefix + relative.as_posix())
                    if not member.isfile():
                        raise ValueError("Expected a regular archive member")
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                        temp_path = Path(tmp.name)
                        with tar.extractfile(member) as stream:
                            shutil.copyfileobj(stream, tmp)
                    try:
                        if digest(temp_path) != item["sha256"]:
                            raise ValueError("File checksum mismatch: " + str(relative))
                        temp_path.replace(target)
                    finally:
                        if temp_path.exists():
                            temp_path.unlink()
        print(source["name"], source["revision"], len(source["files"]), "files verified")


if __name__ == "__main__":
    main()
