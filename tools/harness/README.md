# Keg measurement harness

The single-pass, field-standard long-mode fidelity measurement behind the Keg
gate. Everything here is public and reproducible, so **anyone can re-derive the
reference** (and re-measure any submission) from the public corpus + the model.

## What it does

- **`keg_sample.cpp`** — tokenize the eval stream and sample the **stratified
  long-context token positions** (prose / code / multilingual / technical, with a
  floor per component). Loads the model **vocab-only** (fast). Emits
  `tokens.txt`, `positions.txt`, `components.json`.
- **`keg_extract.cpp`** — load a model (BF16 for the reference, a quant for a
  submission), process the stream once in n_ctx windows (KV reuse), and record
  the **top-k next-token log-probs** at the sampled positions. Emits
  `{positions: {<token_index>: {token: logprob}}, n}`.
- **`keg_compare.py`** — compute the fidelity report (per-component KL
  divergence + top-1 agreement) from a reference and a submission top-k.

The **reference** is the BF16 model's top-k at the sampled positions; every
submission is measured against it. Both use the same tokenizer + extractor, so
their KL is directly comparable.

## Building the C++ tools

They need a llama.cpp build that knows the model's architecture (muse-glimmer
is in mainline llama.cpp). Build llama.cpp with the right CUDA arch for the box,
then compile against it:

```sh
# build llama.cpp (muse-glimmer supported) for your GPU arch (80=A100, 86=4090, 90=H100, 120=Blackwell)
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j

# compile the harness against it
g++ -O2 -std=c++17 -I llama.cpp/include -I llama.cpp/ggml/include \
  keg_extract.cpp -o keg_extract -L llama.cpp/build/bin -lllama -Wl,-rpath,llama.cpp/build/bin
g++ -O2 -std=c++17 -I llama.cpp/include -I llama.cpp/ggml/include \
  keg_sample.cpp -o keg_sample -L llama.cpp/build/bin -lllama -Wl,-rpath,llama.cpp/build/bin
```

## Using it

```sh
# 1. tokenize the stream + sample stratified positions (vocab-only, ~seconds)
./keg_sample --model <any-gguf-of-the-lane> --corpus ../../corpus/production.txt \
  --components ../../corpus/production.components.txt --outdir <dir> --n 5000

# 2. reference: extract top-k from the BF16 model (single pass, ~minutes)
./keg_extract --model <bf16.gguf> --tokens <dir>/tokens.txt \
  --positions <dir>/positions.txt --out reference.json --n-ctx 8192 --top-k 1024

# 3. a submission: same positions, the quant
./keg_extract --model <quant.gguf> --tokens <dir>/tokens.txt \
  --positions <dir>/positions.txt --out sub_topk.json --n-ctx 8192 --top-k 1024

# 4. the report (per-component KL + top-1); the gate keys on kl_max_component
python3 keg_compare.py --ref reference.json --sub sub_topk.json \
  --components <dir>/components.json
```

## Determinism / trustlessness

Positions are a pure function of the corpus + components + tokenizer (seeded by
their hashes), so re-running `keg_sample` gives the same points. The reference is
hash-bound and recorded in every receipt. Because the corpus and this harness are
public, anyone can re-derive the reference and re-verify any submission — no
sealed corpus, no house authority.
