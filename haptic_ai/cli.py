"""CLI entrypoints. Only `validate-schema` is implemented at M0; the rest land later."""

import argparse
import sys

from haptic_ai import schema

_LATER = {
    "ingest": "M1",
    "train": "M2",
    "evaluate": "M2",
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="haptic-ai")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("validate-schema", help="Validate a canonical sensor CSV (SPEC 8.1)")
    p.add_argument("csv", help="Path to a canonical CSV")
    p.set_defaults(func=_validate_schema)

    sub.add_parser("ingest").add_argument("--all", action="store_true")
    for name in ("train", "evaluate", "export", "golden"):
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
