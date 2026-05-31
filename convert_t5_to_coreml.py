#!/usr/bin/env python3
"""Convert a local Hugging Face T5 model directory to a Core ML package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoModelForSeq2SeqLM


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "downloads" / "t5"


class T5LogitsWrapper(torch.nn.Module):
    """Traceable T5 forward pass for app-driven autoregressive decoding."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            decoder_input_ids=decoder_input_ids.to(torch.long),
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits.to(torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Hugging Face T5ForConditionalGeneration checkpoint to "
            "Core ML. The exported model returns logits; your app should run "
            "the tokenizer and generation loop around it."
        )
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Local T5 model directory. Default: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "models" / "coreml"),
        help="Directory where the .mlpackage will be written.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output package name without extension. Defaults to the model directory name.",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=128,
        help="Static encoder input length used for conversion.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=64,
        help="Static decoder input length used for conversion.",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Keep Core ML compute precision at float32 instead of float16.",
    )
    parser.add_argument(
        "--minimum-target",
        choices=("ios16", "ios17", "macos13", "macos14"),
        default="ios16",
        help="Minimum Core ML deployment target for the mlprogram output.",
    )
    parser.add_argument(
        "--compute-units",
        choices=("all", "cpu-only", "cpu-and-gpu", "cpu-and-ne"),
        default="all",
        help="Compute units to use during conversion.",
    )
    return parser.parse_args()


def load_coremltools():
    try:
        import coremltools as ct
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: coremltools. Install the Core ML extras with:\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install coremltools sentencepiece"
        ) from exc
    return ct


def read_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing config.json in {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def find_weight_file(model_dir: Path) -> Path | None:
    preferred = (
        "model.safetensors",
        "pytorch_model.bin",
    )
    for name in preferred:
        path = model_dir / name
        if path.exists():
            return path

    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if safetensors_files:
        return safetensors_files[0]

    pytorch_bins = sorted(model_dir.glob("*.bin"))
    if pytorch_bins:
        return pytorch_bins[0]

    return None


def validate_t5_dir(model_dir: Path) -> tuple[dict, Path]:
    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    config = read_config(model_dir)
    model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    if model_type != "t5" or "T5ForConditionalGeneration" not in architectures:
        raise SystemExit(
            "This converter expects a Hugging Face T5ForConditionalGeneration "
            f"model. Found model_type={model_type!r}, architectures={architectures!r}."
        )

    weight_file = find_weight_file(model_dir)
    if weight_file is None:
        raise SystemExit(f"No supported T5 weight file found in {model_dir}")

    if not (model_dir / "spiece.model").exists():
        print(
            "Warning: spiece.model was not found. The Core ML model can still be "
            "created, but tokenization assets are missing.",
            file=sys.stderr,
        )

    return config, weight_file


def load_t5_model(model_dir: Path, weight_file: Path) -> torch.nn.Module:
    if weight_file.name == "model.safetensors":
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            local_files_only=True,
        )

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_config(config)
    if weight_file.suffix == ".safetensors":
        state_dict = load_safetensors(str(weight_file), device="cpu")
    else:
        state_dict = torch.load(weight_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Unsupported PyTorch checkpoint format: {weight_file}")
    if "shared.weight" in state_dict:
        state_dict.setdefault("encoder.embed_tokens.weight", state_dict["shared.weight"])
        state_dict.setdefault("decoder.embed_tokens.weight", state_dict["shared.weight"])
        state_dict.setdefault("lm_head.weight", state_dict["shared.weight"])
    model.load_state_dict(state_dict)
    return model.to(dtype=torch.float32)


def get_target(ct, name: str):
    targets = {
        "ios16": ct.target.iOS16,
        "ios17": ct.target.iOS17,
        "macos13": ct.target.macOS13,
        "macos14": ct.target.macOS14,
    }
    return targets[name]


def get_compute_units(ct, name: str):
    units = {
        "all": ct.ComputeUnit.ALL,
        "cpu-only": ct.ComputeUnit.CPU_ONLY,
        "cpu-and-gpu": ct.ComputeUnit.CPU_AND_GPU,
    }
    if name == "cpu-and-ne":
        if not hasattr(ct.ComputeUnit, "CPU_AND_NE"):
            raise SystemExit("This coremltools version does not support CPU_AND_NE.")
        return ct.ComputeUnit.CPU_AND_NE
    return units[name]


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_name = args.name or f"{model_dir.name}-t5-logits"
    output_path = output_dir / f"{output_name}.mlpackage"

    config, weight_file = validate_t5_dir(model_dir)
    ct = load_coremltools()

    print(f"Loading T5 model from {model_dir}...")
    print(f"Using weights: {weight_file.name}")
    model = load_t5_model(model_dir, weight_file)
    model.eval()

    wrapper = T5LogitsWrapper(model).eval()
    example_inputs = (
        torch.zeros((1, args.max_source_length), dtype=torch.int32),
        torch.ones((1, args.max_source_length), dtype=torch.int32),
        torch.full(
            (1, args.max_target_length),
            int(config.get("decoder_start_token_id", 0)),
            dtype=torch.int32,
        ),
    )

    print("Tracing PyTorch wrapper...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs, strict=False)

    print(f"Converting to Core ML at {output_path}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=get_target(ct, args.minimum_target),
        compute_units=get_compute_units(ct, args.compute_units),
        compute_precision=ct.precision.FLOAT32
        if args.float32
        else ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, args.max_source_length),
                dtype=int,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, args.max_source_length),
                dtype=int,
            ),
            ct.TensorType(
                name="decoder_input_ids",
                shape=(1, args.max_target_length),
                dtype=int,
            ),
        ],
        outputs=[ct.TensorType(name="logits")],
    )
    mlmodel.short_description = (
        "T5 conditional-generation logits model converted from Hugging Face."
    )
    mlmodel.input_description["input_ids"] = "Padded encoder token IDs."
    mlmodel.input_description["attention_mask"] = "1 for real source tokens, 0 for padding."
    mlmodel.input_description["decoder_input_ids"] = "Current decoder token IDs."
    mlmodel.output_description["logits"] = "Vocabulary logits for each decoder position."
    mlmodel.save(str(output_path))

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_model": str(model_dir),
                "format": "Core ML mlprogram",
                "architecture": config.get("architectures", ["T5"])[0],
                "model_type": config.get("model_type"),
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_target_length,
                "precision": "float32" if args.float32 else "float16",
                "output_file": str(output_path),
                "notes": (
                    "This package exports the T5 forward logits pass. Run "
                    "SentencePiece tokenization and autoregressive decoding in "
                    "the host app."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
