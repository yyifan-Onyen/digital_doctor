from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    Path("data")
    / "knowledge_sources"
    / "Wilhelm and Steketee_2006_Cognitive therapy for OCD. A guide for professionals.docx"
)

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from digital_doctor.tools.knowledge_tree import (  # type: ignore
        BuildOptions,
        DEFAULT_CACHE_PATH,
        DEFAULT_MODEL,
        DEFAULT_OUTPUT_PATH,
        build_knowledge_tree,
        write_tree,
    )
else:
    from .knowledge_tree import (
        BuildOptions,
        DEFAULT_CACHE_PATH,
        DEFAULT_MODEL,
        DEFAULT_OUTPUT_PATH,
        build_knowledge_tree,
        write_tree,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PageIndex-backed knowledge tree for the OCD reference document.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the DOCX source file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the generated knowledge tree JSON.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Intermediate markdown path used before handing content to PageIndex.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model name to use for PageIndex summaries.",
    )
    parser.add_argument(
        "--max-chapters",
        type=int,
        default=None,
        help="Optional cap on how many chapters to keep when preprocessing DOCX inputs.",
    )
    parser.add_argument(
        "--max-section-chars",
        type=int,
        default=2400,
        help="Maximum characters of section text to send to the model per section.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=120,
        help="HTTP timeout for OpenAI requests.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip PageIndex summary generation and fill metadata heuristically.",
    )
    parser.add_argument(
        "--exclude-source-text",
        action="store_true",
        help="Omit full source_text fields from the output JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = BuildOptions(
        input_path=args.input,
        output_path=args.output,
        cache_path=args.cache,
        model=args.model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        max_chapters=args.max_chapters,
        max_section_chars=args.max_section_chars,
        timeout_s=args.timeout_s,
        skip_llm=args.skip_llm,
        include_source_text=not args.exclude_source_text,
    )
    payload = build_knowledge_tree(options)
    write_tree(payload, args.output)
    stats = payload["stats"]
    print(f"Wrote knowledge tree to {args.output}")
    print(
        "Stats:"
        f" chapters={stats['chapter_count']}"
        f" sections={stats['section_count']}"
        f" nodes={stats['node_count']}"
    )


if __name__ == "__main__":
    main()
