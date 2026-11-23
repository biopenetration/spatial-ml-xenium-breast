"""Download the public 10x Genomics Xenium FFPE Human Breast Cancer (Rep 1) files used in this project.

Usage:  python scripts/download_data.py [--out data/raw]
Check the dataset page on 10xgenomics.com for the current license / terms before redistributing any derived data.
"""
import argparse
import urllib.request
from pathlib import Path

BASE = ("https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/"
        "Xenium_FFPE_Human_Breast_Cancer_Rep1")
FILES = {"cell_feature_matrix.h5": f"{BASE}_cell_feature_matrix.h5", "cells.csv.gz": f"{BASE}_cells.csv.gz"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/raw"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"already present: {dest}"); continue
        print(f"downloading {name} ..."); urllib.request.urlretrieve(url, dest)
        if dest.stat().st_size == 0:
            raise RuntimeError(f"empty download: {name}")
        print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
