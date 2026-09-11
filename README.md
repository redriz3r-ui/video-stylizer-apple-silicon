# Video Stylizer for Apple Silicon

Stylize a video with your existing Mochi Diffusion DreamShaper Core ML model.
The default is a **5-second, 512×512, 8 fps preview** (40 frames).
Original audio is trimmed to the same segment and remuxed into the MP4.

## Setup on an M2 Mac

Requires macOS 13.1+, Apple command-line tools (`xcode-select --install`),
Python 3.10+ and FFmpeg (`brew install ffmpeg`).

```bash
cd /path/to/video-stylizer
./setup.sh
./preview.sh
```

`setup.sh` creates `.venv`, resolves the pinned Swift packages, compiles the
native backend, and checks model resources and FFmpeg. The first build requires
internet access. Core ML inference subsequently uses your local model offline.
The Core ML route needs no pip packages, PyTorch or model conversion.

Default model: `~/MochiDiffusion/models/dreamshaper-5_split-einsum`.
This contains compiled `TextEncoder`, `Unet`, `VAEEncoder`, and `VAEDecoder`
`.mlmodelc` directories plus `merges.txt` and `vocab.json`.

## Preview and resume

```bash
./preview.sh input/testclip_01.mov.mp4 --run-dir runs/preview
# Repeat the same command to resume interrupted generation.
./preview.sh input/testclip_01.mov.mp4 --run-dir runs/preview
# Five seconds starting at 10 seconds:
./preview.sh input/testclip_01.mov.mp4 --start 10 --run-dir runs/preview-at-10
# Explicitly request the full clip:
./preview.sh input/testclip_01.mov.mp4 --full --run-dir runs/full
```

Use a new run directory after changing settings, input, model path, or segment.
The manifest rejects mismatched resume settings. Do not replace model contents
in place while resuming. Completed PNGs are saved atomically by the Core ML worker.

The worker creates one pipeline for all frames, uses CPU + Neural Engine and
Apple's reduced-memory mode. Resources may be unloaded/reloaded between stages
to limit memory. First-time Core ML compilation can take several minutes.
The optional `compute_units` config accepts `cpu-and-neural-engine` (default)
or `cpu-and-gpu`. The installed 2023 model hit a Metal assertion in GPU testing
on macOS 26.0.1, so keep the Neural Engine default.
Frame generation is substantially slower than playback. Ctrl-C interrupts;
repeat with the same `--run-dir` to resume. Temporal blending and a fixed seed
reduce variation but do not guarantee flicker-free video.

## Checks without inference

```bash
.venv/bin/python src/pipeline.py --check
./preview.sh input/testclip_01.mov.mp4 --prepare-only --run-dir runs/media-check
.venv/bin/python -m unittest discover -s tests
```

`--prepare-only` makes `output/prepared_preview.mp4` from resized source frames
and audio. It is explicitly **not stylized**.
Actual generation writes `output/styled_final.mp4` (with audio) or
`output/styled_silent.mp4` (without audio). Runs live under this project's `runs/`.

## Backend compatibility

The installed Mochi DreamShaper 5 model supports native Core ML img2img. It has
no ControlNet resources or ControlNet-enabled UNet, so this route does not apply
Canny conditioning; the Canny fields in the shared config apply only to Diffusers.
It uses Apple's default PNDM scheduler. Settings are in `configs/default.json`.

The original SD 1.5 DreamShaper + Canny ControlNet route remains available for
`.safetensors` checkpoints and Diffusers model folders:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/pipeline.py --input input/testclip_01.mov.mp4 \
  --model-path /path/to/sd15-dreamshaper.safetensors --run-dir runs/diffusers-preview
```

That optional route uses MPS when available and downloads ControlNet and any
needed configuration/tokenizer assets on first use. It is separate from the
verified Core ML setup; SDXL/FLUX checkpoints are not supported by that pipeline.

Upstream: https://github.com/apple/ml-stable-diffusion and
https://github.com/MochiDiffusion/MochiDiffusion.
