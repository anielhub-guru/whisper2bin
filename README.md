# Whisper HF To `.bin` And T5 Core ML Converter

This project converts Hugging Face fine-tuned Whisper models into the Whisper.cpp
GGML `.bin` format. It can also convert local T5 models into Core ML `.mlpackage` files.


## Setup

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates `.venv`, installs the Python dependencies, and clones:

- `openai/whisper`, needed for Whisper mel filter assets
- `ggml-org/whisper.cpp`, needed for the official HF-to-GGML converter

## Convert A Whisper Model

You can provide any Hugging Face model repository ID or a local directory path.

```bash
source .venv/bin/activate
python convert_whisper_to_bin.py <model_folder_path>
```

Example:

```bash
source .venv/bin/activate
python convert_whisper_to_bin.py <model_folder_path>
```

The output will be written to:

```text
models/<model_name>-f16.bin
```

## Convert A T5 Model To Core ML

The converter supports both `.safetensors` and PyTorch `pytorch_model.bin` model files.
Point it to the local directory containing your Hugging Face T5 model.

```bash
source .venv/bin/activate
python convert_t5_to_coreml.py path/to/t5_model_directory
```

The output will be written to:

```text
models/coreml/<model_directory_name>-t5-logits.mlpackage
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
python convert_t5_to_coreml.py path/to/t5_model_directory --max-source-length 256 --max-target-length 128
```

## Use Float32 Instead Of Float16 (Whisper)

```bash
python convert_whisper_to_bin.py <model_folder_path> --f32
```

Float16 is smaller and is the normal choice for Whisper.cpp. Float32 is larger
and may be useful for debugging or compatibility checks.

## Notes

- The converter downloads Hugging Face model files into `downloads/`.
- Converted `.bin` files are written into `models/`.
- Converted Core ML packages are written into `models/coreml/`.
- You can also pass a Hugging Face model id instead of a local model folder path.
- This produces Whisper.cpp GGML `.bin` files, not llama.cpp GGUF files.
