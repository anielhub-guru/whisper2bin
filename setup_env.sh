#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
VENDOR_DIR="$ROOT_DIR/vendor"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

mkdir -p "$VENDOR_DIR"

if [ ! -d "$VENDOR_DIR/whisper" ]; then
  git clone https://github.com/openai/whisper "$VENDOR_DIR/whisper"
fi

if [ ! -d "$VENDOR_DIR/whisper.cpp" ]; then
  git clone https://github.com/ggml-org/whisper.cpp "$VENDOR_DIR/whisper.cpp"
fi

echo
echo "Environment ready."
echo "Activate it with: source .venv/bin/activate"
echo "Convert the Nigerian accent model with:"
echo "  python convert_whisper_to_bin.py rishabbahal/whisper-small-nigerian-accent"
echo "Convert the local T5 model to Core ML with:"
echo "  python convert_t5_to_coreml.py downloads/t5"
echo "Convert the local T5-tiny model to Core ML with:"
echo "  python convert_t5_to_coreml.py downloads/t5-tiny"
