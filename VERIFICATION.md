# Video Stylizer verification — September 11, 2026

Project updated in `/path/to/video-stylizer`.

## Passed

- Native arm64 Swift 6.1.2 release build, targeting macOS 13.1+.
- Python 3.11 virtual environment created in the project; Core ML route requires no pip packages.
- FFmpeg/ffprobe 7.1.1 present, H.264/AAC encoding tested.
- Existing Mochi DreamShaper 5 compiled Core ML resources detected, including VAEEncoder for img2img.
- Actual input clip: 38.016 seconds, 480×848, H.264 with AAC audio.
- Five-second media preview: 40 frames, 512×512, 8 fps, AAC audio, exact 5.000-second duration.
- Silent input extraction and MP4 assembly tested.
- Three validation tests passed; changed start time rejected when resuming an existing run.

## Disk space

The final verification attempt hit `No space left on device`. Temporary build files created during this task were removed. Free more disk space before retrying Core ML compilation; disk pressure may be contributing to the stall. The test process was stopped.

## Runtime limitation

Actual stylized output has NOT been verified. Sandboxed model execution ran CPU operations very slowly. Native CPU + Neural Engine execution waited in Apple's ANE compilation service. CPU + GPU failed with an Apple Metal assertion: `shape.count = 0 != strides.count = 3`. The Neural Engine remains the default. This is an older compiled model on macOS 26.0.1; compatibility is unresolved.

No ControlNet is applied by the Core ML route: the installed Mochi model lacks ControlNet resources and a ControlNet-enabled UNet. The optional SD 1.5 Diffusers + ControlNet route remains separate and its Python packages were not installed or inference-tested.

## Run

```bash
cd /path/to/video-stylizer
./preview.sh input/testclip_01.mov.mp4 --run-dir runs/preview
```

This defaults to five seconds. Repeat the same command to resume. Use `--full` for the whole video. `./setup.sh` rebuilds the backend; `.venv/bin/python src/pipeline.py --check` checks prerequisites. `--prepare-only` verifies video/audio without model inference.

If native loading continues to stall, restart macOS and verify the same model can generate an image in Mochi before retrying. A newer compatible Core ML conversion may be needed if Mochi also fails; no replacement model was downloaded.

Implementation references: [Apple Core ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion) and [Mochi Diffusion](https://github.com/MochiDiffusion/MochiDiffusion).
