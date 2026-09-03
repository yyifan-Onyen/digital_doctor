"""CLI for exporting harness traces to SFT or skill-conditioned OPSD JSONL."""

from __future__ import annotations

import argparse

from ..harness.training import export_distillation_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", help="Harness session trace JSONL")
    parser.add_argument("output_path", help="Destination training JSONL")
    parser.add_argument("--format", choices=("sft", "opsd"), default="opsd")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    count = export_distillation_records(
        args.trace_path,
        args.output_path,
        output_format=args.format,
    )
    print(f"Exported {count} {args.format.upper()} records to {args.output_path}")


if __name__ == "__main__":
    main()
