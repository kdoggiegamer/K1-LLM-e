# K1(LLM)e

K1(LLM)e is a Rust CLI for running local language models from GGUF and KMF files.

## Project goals

- Faster inference through efficient local execution.
- Cheaper inference by running models on user-controlled hardware.
- Smaller model files through compressed storage and, in future KMF revisions, variable-precision floating-point weights.

## Project goals

K1(LLM)e aims to make local AI inference faster, cheaper, and more accessible:

- **Faster inference:** improve generation speed through efficient execution and hardware-aware optimizations.
- **Cheaper inference:** let people run models on hardware they control, reducing reliance on paid hosted inference.
- **Smaller model files:** reduce storage and distribution costs while preserving useful output quality.
- **Practical access:** provide a simple CLI first, with a graphical interface planned for users who prefer visual workflows.

These are project goals, not measured results. K1(LLM)e v1.0.0 does not yet claim speed or quality improvements over its inference backend.

## Current release: v1.0.0

- Run GGUF models with a prompt.
- Run KMF models through the same inference path.
- Inspect model container headers.
- Compress GGUF to KMF and restore it to GGUF.

KMF v1 is a lossless container around GGUF. It preserves the GGUF tensor encodings and does not yet implement native variable-precision KMF weights.

## Future development

Planned work, in order of importance:

1. **Define native KMF tensor encoding:** specify per-tensor metadata and variable-precision floating-point representations, with compatibility and corruption checks.
2. **Measure size and quality tradeoffs:** compare KMF against GGUF using file size, load time, generation speed, memory use, and output-quality evaluations.
3. **Improve inference performance:** profile the current runtime and add optimizations only where benchmarks show a benefit.
4. **Broaden model and hardware support:** document supported architectures and improve CPU and accelerator options as the runtime allows.
5. **Improve the CLI and add a GUI:** add useful generation controls and, later, a graphical way to choose models and run prompts.

The roadmap is exploratory; features will be added as they are designed and measured.

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