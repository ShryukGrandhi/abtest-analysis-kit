"""Command line entry point.

    abkit simulate events.csv --users 20000 --lift treatment=1.05
    abkit analyze events.csv --control control --ratio revenue/sessions
"""

import argparse
import json
import sys

from .data import load_events
from .report import analyze, to_markdown
from .simulate import SimConfig, write_events_csv


def _parse_lift(items):
    out = {}
    for item in items or []:
        name, _, val = item.partition("=")
        out[name] = float(val)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="abkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a synthetic event log")
    s.add_argument("path")
    s.add_argument("--users", type=int, default=20000)
    s.add_argument("--variants", default="control,treatment")
    s.add_argument("--lift", action="append", help="variant=multiplier on conversion")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dirty", type=float, default=0.0, help="fraction of bad rows to inject")

    a = sub.add_parser("analyze", help="analyse an event log")
    a.add_argument("path")
    a.add_argument("--control", required=True)
    a.add_argument("--metric", action="append", help="metric to analyse (default: all)")
    a.add_argument("--ratio", action="append", default=[], help="num/den ratio metric")
    a.add_argument("--alpha", type=float, default=0.05)
    a.add_argument("--no-cuped", action="store_true")
    a.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "simulate":
        cfg = SimConfig(n_users=args.users, variants=tuple(args.variants.split(",")),
                        conversion_lift=_parse_lift(args.lift))
        write_events_csv(args.path, cfg, seed=args.seed, dirty_fraction=args.dirty)
        return 0

    table = load_events(args.path)
    ratios = [tuple(r.split("/", 1)) for r in args.ratio]
    result = analyze(table, args.control, metrics=args.metric, ratios=ratios,
                     alpha=args.alpha, use_cuped=not args.no_cuped)
    if args.json:
        json.dump(result.as_dict(), sys.stdout, indent=2)
        print()
    else:
        print(to_markdown(result))
    return 2 if result.srm_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
