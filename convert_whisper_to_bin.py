#!/usr/bin/env python3
"""Download a Hugging Face Whisper model and convert it to whisper.cpp GGML .bin."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "rishabbahal/whisper-small-nigerian-accent"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_repo(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", url, str(path)])


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def ensure_added_tokens(model_dir: Path) -> None:
    added_tokens = model_dir / "added_tokens.json"
    if added_tokens.exists():
        return
    added_tokens.write_text("{}\n", encoding="utf-8")


def validate_hf_model_dir(model_dir: Path) -> None:
    required = ["config.json", "vocab.json"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"Downloaded model is missing required file(s): {', '.join(missing)}"
        )

    has_weights = any(
        (model_dir / name).exists()
        for name in (
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "flax_model.msgpack",
        )
    )
    has_sharded_weights = any(model_dir.glob("*.safetensors")) or any(
        model_dir.glob("pytorch_model-*.bin")
    )
    if not (has_weights or has_sharded_weights):
        raise SystemExit("Downloaded model does not appear to include model weights.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face Whisper model to whisper.cpp GGML .bin."
    )
    parser.add_argument(
        "model_id",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id or local model directory. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "models"),
        help="Directory where the .bin file will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "downloads"),
        help="Directory used for downloaded Hugging Face snapshots.",
    )
    parser.add_argument(
        "--f32",
        action="store_true",
        help="Write float32 weights instead of the default float16 output.",
    )
    parser.add_argument(
        "--skip-repo-checkout",
        action="store_true",
        help="Do not auto-clone openai/whisper and whisper.cpp if vendor copies are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    vendor_dir = ROOT / "vendor"
    whisper_repo = vendor_dir / "whisper"
    whisper_cpp_repo = vendor_dir / "whisper.cpp"

    if not args.skip_repo_checkout:
        ensure_repo(whisper_repo, "https://github.com/openai/whisper")
        ensure_repo(whisper_cpp_repo, "https://github.com/ggml-org/whisper.cpp")

    converter = whisper_cpp_repo / "models" / "convert-h5-to-ggml.py"
    if not converter.exists():
        raise SystemExit(
            "Missing whisper.cpp converter. Run ./setup_env.sh or clone "
            "https://github.com/ggml-org/whisper.cpp into vendor/whisper.cpp."
        )
    if not (whisper_repo / "whisper" / "assets" / "mel_filters.npz").exists():
        raise SystemExit(
            "Missing OpenAI Whisper assets. Run ./setup_env.sh or clone "
            "https://github.com/openai/whisper into vendor/whisper."
        )

    maybe_local = Path(args.model_id).expanduser()
    if maybe_local.exists():
        model_dir = maybe_local.resolve()
        model_label = maybe_local.name
    else:
        print(f"Downloading {args.model_id} from Hugging Face...")
        model_dir = Path(
            snapshot_download(
                repo_id=args.model_id,
                cache_dir=cache_dir,
                local_dir=cache_dir / safe_name(args.model_id),
            )
        ).resolve()
        model_label = safe_name(args.model_id)

    ensure_added_tokens(model_dir)
    validate_hf_model_dir(model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("ggml-model*.bin"))
    cmd = [
        sys.executable,
        str(converter),
        str(model_dir),
        str(whisper_repo),
        str(output_dir),
    ]
    if args.f32:
        cmd.append("use-f32")
    run(cmd)

    generated_name = "ggml-model-f32.bin" if args.f32 else "ggml-model.bin"
    generated = output_dir / generated_name
    if not generated.exists() and not before:
        raise SystemExit("Conversion finished, but no ggml-model*.bin file was found.")

    suffix = "f32" if args.f32 else "f16"
    final_path = output_dir / f"{model_label}-{suffix}.bin"
    if generated.exists():
        if final_path.exists():
            final_path.unlink()
        shutil.move(str(generated), str(final_path))

    metadata_path = final_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_model": args.model_id,
                "format": "whisper.cpp GGML",
                "precision": suffix,
                "output_file": str(final_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nDone: {final_path}")


if __name__ == "__main__":
    main()
