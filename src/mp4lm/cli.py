from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._utils import parse_chunk_size
from .pytorch_bridge import export_pytorch_state_dict, pack_pytorch_state_dict
from .reader import Mp4LMBundle
from .writer import build_from_spec_file, pack_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mp4lm", description="Mp4LM v0.1 reference CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Build an .mp4lm bundle")
    pack_parser.add_argument("--spec", type=Path, help="JSON spec describing artifacts to bundle")
    pack_parser.add_argument("--input-dir", type=Path, help="Directory to pack as file artifacts")
    pack_parser.add_argument("--input", type=Path, help="Single input artifact such as a PyTorch checkpoint")
    pack_parser.add_argument(
        "--input-format",
        choices=["pytorch-state-dict"],
        help="Interpret --input using a bridge format",
    )
    pack_parser.add_argument("--output", type=Path, required=True, help="Output .mp4lm path")
    pack_parser.add_argument("--chunk-size", default="8MiB", help="Chunk size, for example 8MiB")
    pack_parser.add_argument("--deterministic", action="store_true", help="Force deterministic timestamps and IDs")

    inspect_parser = subparsers.add_parser("inspect", help="Print a short bundle summary")
    inspect_parser.add_argument("bundle", type=Path)

    manifest_parser = subparsers.add_parser("manifest", help="Print the canonical manifest as JSON")
    manifest_parser.add_argument("bundle", type=Path)

    verify_parser = subparsers.add_parser("verify", help="Verify checksums")
    verify_parser.add_argument("bundle", type=Path)

    cat_parser = subparsers.add_parser("cat", help="Stream one artifact to stdout")
    cat_parser.add_argument("bundle", type=Path)
    cat_parser.add_argument("artifact", help="Artifact name or artifact id")

    unpack_parser = subparsers.add_parser("unpack", help="Extract all or selected artifacts")
    unpack_parser.add_argument("bundle", type=Path)
    unpack_parser.add_argument("output", type=Path)
    unpack_parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        help="Artifact name or id to extract; may be repeated",
    )
    unpack_parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write bundle.manifest.json alongside extracted payloads",
    )
    unpack_parser.add_argument(
        "--as",
        dest="output_format",
        choices=["directory", "pytorch-state-dict"],
        default="directory",
        help="Extraction target format",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "pack":
        chunk_size = parse_chunk_size(args.chunk_size)
        sources = [
            args.spec is not None,
            args.input_dir is not None,
            args.input is not None or args.input_format is not None,
        ]
        if sum(sources) != 1:
            parser.error("pack requires exactly one source: --spec, --input-dir, or --input with --input-format")
        if args.spec:
            builder = build_from_spec_file(args.spec, chunk_size=chunk_size)
            if args.deterministic:
                builder.deterministic = True
            builder.write(args.output)
        elif args.input_dir:
            pack_directory(
                args.input_dir,
                args.output,
                chunk_size=chunk_size,
                deterministic=args.deterministic,
            )
        else:
            if args.input is None or args.input_format is None:
                parser.error("--input requires --input-format")
            if args.input_format == "pytorch-state-dict":
                pack_pytorch_state_dict(
                    args.input,
                    args.output,
                    chunk_size=chunk_size,
                    deterministic=args.deterministic,
                )
        print(args.output)
        return 0

    bundle = Mp4LMBundle.open(args.bundle)

    if args.command == "inspect":
        manifest = bundle.manifest()
        summary = {
            "artifact_count": len(manifest["artifacts"]),
            "bundle_id": manifest["bundle_id"],
            "chunk_count": len(manifest["chunks"]),
            "created_at": manifest["created_at"],
            "model": manifest["model"],
            "spec": manifest["spec"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "manifest":
        print(json.dumps(bundle.manifest(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "verify":
        report = bundle.verify()
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "cat":
        sys.stdout.buffer.write(bundle.fetch_artifact(args.artifact))
        return 0

    if args.command == "unpack":
        if args.output_format == "pytorch-state-dict":
            export_pytorch_state_dict(bundle, args.output, artifacts=args.artifacts)
            written = [args.output]
        else:
            written = bundle.unpack(
                args.output,
                artifacts=args.artifacts,
                include_manifest=not args.no_manifest,
            )
        print(
            json.dumps(
                {
                    "bundle": str(args.bundle),
                    "ok": True,
                    "output": str(args.output),
                    "output_format": args.output_format,
                    "written": [str(path) for path in written],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
