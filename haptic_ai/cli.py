"""CLI entrypoints. Model commands land at M2."""

import argparse
import sys

from haptic_ai import schema

_LATER = {
    "train": "M2",
    "export": "M2",
    "golden": "M2",
}


def _validate_schema(args: argparse.Namespace) -> int:
    try:
        df = schema.read_canonical_csv(args.csv)
    except (schema.SchemaError, ValueError, OSError) as e:
        print(f"INVALID {args.csv}: {e}")
        return 1
    print(f"OK {args.csv}: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    from haptic_ai import corpus

    df = corpus.build()
    print(f"OK {corpus.PROCESSED}: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    return 0


def _report_corpus(args: argparse.Namespace) -> int:
    from haptic_ai import corpus

    s = corpus.report(corpus.load())
    print(f"OK {corpus.REPORTS / 'M1_corpus.md'}: {s['rows']} rows, {s['subjects']} subjects")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from haptic_ai import evaluate

    r = evaluate.run(args.config)
    for key, v in r["results"].items():
        cells = [
            f"{tag} {s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f}"
            for tag, s in v["loso"].items()
        ]
        print(f"{key:12} LOSO macro-F1  " + "  ".join(cells))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="haptic-ai")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("validate-schema", help="Validate a canonical sensor CSV (SPEC 8.1)")
    p.add_argument("csv", help="Path to a canonical CSV")
    p.set_defaults(func=_validate_schema)

    p = sub.add_parser("ingest", help="Build the M1 corpus in data/processed/")
    p.add_argument("--all", action="store_true", help="all M1 datasets (the only mode)")
    p.set_defaults(func=_ingest)
    p = sub.add_parser("report-corpus", help="Write reports/M1_corpus.{json,md}")
    p.set_defaults(func=_report_corpus)
    p = sub.add_parser("evaluate", help="LOSO evaluation, writes reports/M2_*.json")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_evaluate)
    for name in ("train", "export", "golden"):
        sub.add_parser(name).add_argument("--config", required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    if args.command in _LATER:
        print(f"'{args.command}' is not implemented until {_LATER[args.command]}")
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
