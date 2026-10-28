# K1(LLM)e

K1(LLM)e is a Rust CLI for running local language models from GGUF and KMF files.

## Project goals

- Faster inference through efficient local execution.
- Cheaper inference by running models on user-controlled hardware.
- Smaller model files through compressed storage and, in future KMF revisions, variable-precision floating-point weights.

## v1.0.0

- Run GGUF models with a prompt.
- Run KMF models through the same inference path.
- Inspect model container headers.
- Compress GGUF to KMF and restore it to GGUF.

The first release establishes the CLI and KMF container. It does not yet implement native variable-precision KMF weights or claim speedups over the selected inference backend. The project also plans a graphical interface.

## Build requirements

Install Rust, a C/C++ compiler, and libclang. The llama_cpp Rust binding builds llama.cpp from source.

    cargo build --release

## Usage

    k1 run ./model.gguf --prompt "Explain why the sky is blue." --max-tokens 200
    k1 run ./model.kmf --prompt "Write a short greeting."
    k1 inspect ./model.kmf
    k1 compress ./model.gguf ./model.kmf
    k1 decompress ./model.kmf ./restored.gguf

## KMF v1 format

KMF v1 is a versioned container for one GGUF payload. It uses zstd compression at level 6 and stores the original payload length in a little-endian header.

| Offset | Size | Meaning |
| --- | ---: | --- |
| 0 | 8 bytes | Magic: K1KMF followed by three zero bytes |
| 8 | 2 bytes | Version (1) |
| 10 | 8 bytes | Uncompressed GGUF length |
| 18 | remaining | zstd-compressed GGUF bytes |

GGUF stores tensor types and precision. KMF v1 preserves those encodings and adds lossless container compression; it does not requantize weights. KMF v1 is a compatibility and packaging step toward smaller model files, not a native variable-precision encoding.

## Implementation

Inference and hardware support come from llama.cpp through its Rust API. Split GGUF shards are not bundled in KMF v1.