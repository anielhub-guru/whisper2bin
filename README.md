# Whisper HF To `.bin` And T5 Core ML Converter

This project converts Hugging Face fine-tuned Whisper models into the Whisper.cpp
GGML `.bin` format. It can also convert the local T5 models in `downloads/t5`
and `downloads/t5-tiny` into Core ML `.mlpackage` files.

The default model is:

```text
rishabbahal/whisper-small-nigerian-accent
```

## Setup

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates `.venv`, installs the Python dependencies, and clones:

- `openai/whisper`, needed for Whisper mel filter assets
- `ggml-org/whisper.cpp`, needed for the official HF-to-GGML converter

## Convert The Nigerian Accent Model

```bash
source .venv/bin/activate
python convert_whisper_to_bin.py rishabbahal/whisper-small-nigerian-accent
```

The output will be written to:

```text
models/rishabbahal__whisper-small-nigerian-accent-f16.bin
```

## Convert Another Hugging Face Whisper Model

```bash
source .venv/bin/activate
python convert_whisper_to_bin.py openai/whisper-small
```

## Convert The Local T5 Model To Core ML

The T5 models in this repo are at:

```text
downloads/t5
downloads/t5-tiny
```

They are Hugging Face `T5ForConditionalGeneration` checkpoints. The converter
supports both the `.safetensors` model in `downloads/t5` and the PyTorch
`pytorch_model.bin` model in `downloads/t5-tiny`.

```bash
source .venv/bin/activate
python convert_t5_to_coreml.py downloads/t5
python convert_t5_to_coreml.py downloads/t5-tiny
```

The outputs will be written to:

```text
models/coreml/t5-t5-logits.mlpackage
models/coreml/t5-tiny-t5-logits.mlpackage
```

This Core ML package exposes the T5 forward logits pass:

- `input_ids`: padded encoder token IDs
- `attention_mask`: source-token mask
- `decoder_input_ids`: current decoder token IDs
- `logits`: vocabulary logits for each decoder position

Tokenization and the autoregressive generation loop should run in the host app.
By default, the converter uses static shapes of 128 source tokens and 64 target
tokens. You can change those limits:

```bash
python convert_t5_to_coreml.py downloads/t5 --max-source-length 256 --max-target-length 128
```

## Use Float32 Instead Of Float16

```bash
python convert_whisper_to_bin.py rishabbahal/whisper-small-nigerian-accent --f32
```

Float16 is smaller and is the normal choice for Whisper.cpp. Float32 is larger
and may be useful for debugging or compatibility checks.

## Notes

- The converter downloads model files into `downloads/`.
- Converted `.bin` files are written into `models/`.
- Converted Core ML packages are written into `models/coreml/`.
- You can also pass a local model directory instead of a Hugging Face model id.
- This produces Whisper.cpp GGML `.bin` files, not llama.cpp GGUF files.
