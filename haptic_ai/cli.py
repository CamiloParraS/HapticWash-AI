"""CLI entrypoints."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="haptic-ai")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # ingest --all
    ingest_parser = subparsers.add_parser("ingest", help="Ingest datasets")
    ingest_parser.add_argument("--all", action="store_true", help="Ingest all datasets")

    # train --config
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", required=True, help="Config file")

    # evaluate --config
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--config", required=True, help="Config file")

    # export --config
    export_parser = subparsers.add_parser("export", help="Export to TFLite")
    export_parser.add_argument("--config", required=True, help="Config file")

    # golden --config
    golden_parser = subparsers.add_parser("golden", help="Generate golden files")
    golden_parser.add_argument("--config", required=True, help="Config file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
