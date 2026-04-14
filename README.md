# mp4lm

`mp4lm` is a streamable, inspectable model artifact format and reference Python implementation.

It is designed for:

- exact payload recovery
- partial fetch and progressive loading
- human-readable manifests
- transport-friendly chunking
- optional runtime bridge support

## What is included

- `src/mp4lm`: the Python reference library
- `mp4lm pack`, `inspect`, `manifest`, `verify`, `cat`, `unpack`: the CLI
- `docs/`: a GitHub Pages landing page
- `examples/`: a notebook demo

## Install

This project targets Python 3.12.

```bash
python -m pip install -e .
```

If you want to use the PyTorch bridge, install `torch` in the same environment first.

```bash
python -m pip install torch
```

## Quick Start

### 1. Build a bundle from a directory

```bash
mp4lm pack \
  --input-dir ./artifacts \
  --output ./model.mp4lm \
  --chunk-size 8MiB \
  --deterministic
```

This packages every file under `./artifacts` into a single `.mp4lm` bundle.
File paths are stored relative to the input directory.

### 2. Inspect the bundle

```bash
mp4lm inspect ./model.mp4lm
```

This prints a summary with:

- artifact count
- chunk count
- bundle id
- created timestamp
- model metadata

### 3. Print the manifest

```bash
mp4lm manifest ./model.mp4lm
```

This prints the canonical JSON manifest, which is the main source of meaning for the bundle.

### 4. Verify checksums

```bash
mp4lm verify ./model.mp4lm
```

This validates:

- chunk checksums
- artifact payload checksums
- header/footer consistency
- manifest/index integrity

### 5. Extract one artifact

```bash
mp4lm cat ./model.mp4lm config.json > config.json
```

Use `mp4lm cat` when you want raw bytes for a single artifact.

### 6. Unpack the bundle

```bash
mp4lm unpack ./model.mp4lm ./output
```

By default, `unpack` writes:

- `bundle.manifest.json`
- `artifacts/<object_type>/<artifact_name>`

To extract only selected artifacts:

```bash
mp4lm unpack \
  ./model.mp4lm \
  ./output \
  --artifact config.json \
  --artifact tokenizer.json
```

To skip the manifest sidecar:

```bash
mp4lm unpack ./model.mp4lm ./output --no-manifest
```

## PyTorch Bridge

The reference implementation includes a PyTorch `state_dict` bridge.

### Pack a checkpoint

```bash
mp4lm pack \
  --input ./checkpoint.pt \
  --input-format pytorch-state-dict \
  --output ./checkpoint.mp4lm \
  --deterministic
```

This loads the checkpoint, extracts the `state_dict`, converts each tensor into canonical bytes, and writes a bundle.

### Restore a checkpoint

```bash
mp4lm unpack \
  ./checkpoint.mp4lm \
  ./restored.pt \
  --as pytorch-state-dict
```

This rebuilds a PyTorch-compatible `state_dict` file from the tensor artifacts in the bundle.

If `torch` is not installed, this mode is unavailable.

## Python API

### Build a bundle

```python
from mp4lm import BundleBuilder

builder = BundleBuilder(
    model={"name": "toy-model"},
    deterministic=True,
)

builder.add_tensor(
    "layers.0.weight",
    tensor_bytes,
    dtype="float32",
    shape=[2, 2],
)

builder.add_file("config.json", b'{"hidden_size": 2}')
builder.write("toy_model.mp4lm")
```

### Open and inspect

```python
from mp4lm import Mp4LMBundle

bundle = Mp4LMBundle.open("toy_model.mp4lm")
manifest = bundle.manifest()
artifacts = bundle.list_artifacts()
```

### Fetch exact bytes

```python
raw = bundle.fetch_artifact("config.json")
tensor = bundle.tensor("layers.0.weight")
```

### Verify

```python
report = bundle.verify()
assert report["ok"]
```

### Selective unpack

```python
bundle.unpack("./output", artifacts=["config.json"])
```

## PyTorch API

```python
from mp4lm import pack_pytorch_state_dict, load_pytorch_state_dict

pack_pytorch_state_dict(
    "./checkpoint.pt",
    "./checkpoint.mp4lm",
    deterministic=True,
)

state_dict = load_pytorch_state_dict("./checkpoint.mp4lm")
```

## GitHub Pages

The landing page lives in `docs/` and is ready to publish with GitHub Pages.

Suggested Pages source:

- branch: `main`
- folder: `/docs`

## Design Notes

The current reference implementation focuses on the v0.1 core:

- fixed-size header
- canonical JSON manifest
- binary index
- raw payloads
- SHA-256 verification
- selective fetch
- optional PyTorch bridge

The notebook demo in `examples/` is kept separate from the publishable surface so the repo stays clean and easy to browse.

## License

MIT