"""Safely extract official COCO zip archives into the configured root."""
import argparse, zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", help="train2017.zip val2017.zip annotations_trainval2017.zip")
    parser.add_argument("--output", default="/data")
    args = parser.parse_args(); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    for archive in args.archives:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                destination = (root / member.filename).resolve()
                if root.resolve() not in destination.parents and destination != root.resolve():
                    raise ValueError(f"unsafe archive path: {member.filename}")
            zf.extractall(root)
    print(f"COCO extracted to {root}")


if __name__ == "__main__": main()
