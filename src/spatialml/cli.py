"""Command line interface: `spatialml demo` runs the synthetic end-to-end pipeline."""
import argparse
import json

from . import __version__
from .config import load_config
from .pipeline import run_demo


def main(argv=None):
    p = argparse.ArgumentParser(prog="spatialml", description="Leakage-safe spatial ML for Xenium data")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the synthetic end-to-end pipeline")
    d.add_argument("--out", default="demo_output"); d.add_argument("--cells", type=int, default=6000)
    d.add_argument("--seed", type=int, default=0); d.add_argument("--bootstrap", type=int, default=200)
    c = sub.add_parser("check-config", help="validate a configuration file against the leakage rules")
    c.add_argument("path")
    a = p.parse_args(argv)
    if a.cmd == "demo":
        s = run_demo(a.out, a.cells, a.seed, a.bootstrap)
        print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2))
    else:
        load_config(a.path); print("configuration OK")


if __name__ == "__main__":
    main()
