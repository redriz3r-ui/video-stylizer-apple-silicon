#!/usr/bin/env python3

import argparse
import json
import shutil
import math
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / "MochiDiffusion/models/dreamshaper-5_split-einsum"


def model_backend(path):
    if path.is_dir() and (path / "TextEncoder.mlmodelc").is_dir():
        required = ["TextEncoder.mlmodelc", "VAEEncoder.mlmodelc", "VAEDecoder.mlmodelc", "merges.txt", "vocab.json"]
        if not (path / "Unet.mlmodelc").is_dir():
            required += ["UnetChunk1.mlmodelc", "UnetChunk2.mlmodelc"]
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            raise ValueError(f"Incomplete Core ML model: {missing}")
        return "coreml"
    if path.is_file() and path.suffix == ".safetensors":
        return "diffusers"
    if (path / "model_index.json").is_file():
        return "diffusers"
    raise ValueError("Expected a Mochi Core ML model folder, Diffusers folder, or .safetensors checkpoint")


def validate_config(cfg):
    if cfg.get("compute_units", "cpu-and-neural-engine") not in ("cpu-and-neural-engine", "cpu-and-gpu"):
        raise ValueError("compute_units must be cpu-and-neural-engine or cpu-and-gpu")
    for key in ("fps", "size", "steps"):
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["size"] % 8:
        raise ValueError("size must be divisible by 8")
    for key in ("strength", "temporal_mix"):
        if not math.isfinite(float(cfg[key])) or not 0 <= cfg[key] < 1:
            raise ValueError(f"{key} must be in [0, 1)")
    if int(cfg["steps"] * cfg["strength"]) < 1:
        raise ValueError("strength * steps must be at least 1")
    if not math.isfinite(float(cfg["guidance"])) or cfg["guidance"] < 0:
        raise ValueError("guidance must be finite and nonnegative")
    if not isinstance(cfg["seed"], int) or not 0 <= cfg["seed"] <= 4294967295:
        raise ValueError("seed must fit UInt32")


def preflight(model, backend, cfg):
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RuntimeError(f"Missing {executable}; run brew install ffmpeg")
    if backend == "coreml":
        if cfg["size"] != 512:
            raise ValueError("This DreamShaper Core ML backend requires size=512")
        binary = ROOT / "coreml/.build/release/video-stylizer-coreml"
        if not binary.is_file():
            raise RuntimeError("Run ./setup.sh to build the Core ML backend")
        print("[check] Core ML model resources and native backend found")
        print("[check] Core ML uses img2img; this Mochi model has no ControlNet support")
    else:
        import torch
        import cv2
        import diffusers
        from PIL import Image
        print(f"[check] torch={torch.__version__}; MPS={torch.backends.mps.is_available()}; diffusers={diffusers.__version__}")


def stylize_coreml(model, dirs, cfg, run_dir):
    job = {key: cfg[key] for key in ("prompt", "negative_prompt", "strength", "steps", "guidance", "seed", "temporal_mix")}
    job["compute_units"] = cfg.get("compute_units", "cpu-and-neural-engine")
    job.update(model=str(model), frames=str(dirs["frames"]), styled=str(dirs["styled"]))
    job_path = run_dir / "coreml-job.json"
    job_path.write_text(json.dumps(job, indent=2))
    run_cmd([str(ROOT / "coreml/.build/release/video-stylizer-coreml"), str(job_path)])


def run_cmd(cmd):
    print(">>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_run_dirs(run_dir: Path):
    dirs = {}
    for name in ["frames", "control", "styled", "output", "logs"]:
        p = run_dir / name
        p.mkdir(parents=True, exist_ok=True)
        dirs[name] = p
    return dirs


def ffprobe_has_audio(video_path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def ffprobe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def extract_frames(input_video: Path, out_dir: Path, fps: int, size: int, start=None, duration=None):
    pattern = out_dir / "frame_%06d.png"
    vf = f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size},setsar=1"
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(input_video)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", vf, str(pattern)]
    run_cmd(cmd)


def extract_audio(input_video: Path, audio_out: Path, start=None, duration=None):
    if not ffprobe_has_audio(input_video):
        return False
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(input_video)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-c:a", "aac", str(audio_out)]
    run_cmd(cmd)
    return True


def generate_canny(frames_dir: Path, control_dir: Path, low: int, high: int):
    import cv2
    frames = sorted(frames_dir.glob("frame_*.png"))
    for i, frame in enumerate(frames, 1):
        dst = control_dir / frame.name
        if dst.exists():
            continue
        img = cv2.imread(str(frame))
        if img is None:
            raise RuntimeError(f"Could not read frame: {frame}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(dst), rgb)
        if i % 25 == 0 or i == len(frames):
            print(f"[control] {i}/{len(frames)}")


def load_pipeline(model_path: str, controlnet_model: str):
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"[pipeline] device={device} dtype={dtype}")

    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)
    loader = StableDiffusionControlNetImg2ImgPipeline.from_pretrained if Path(model_path).is_dir() else StableDiffusionControlNetImg2ImgPipeline.from_single_file
    pipe = loader(
        model_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


def stylize_frames(pipe, frames_dir: Path, control_dir: Path, styled_dir: Path, cfg: dict):
    import torch
    from PIL import Image
    frames = sorted(frames_dir.glob("frame_*.png"))
    prev_styled = None

    for i, frame in enumerate(frames, 1):
        out = styled_dir / frame.name
        control = control_dir / frame.name

        if out.exists():
            print(f"[stylize] resume skip {out.name}")
            prev_styled = Image.open(out).convert("RGB")
            continue

        init_img = Image.open(frame).convert("RGB")
        control_img = Image.open(control).convert("RGB")

        temporal_mix = float(cfg.get("temporal_mix", 0.0))
        if prev_styled is not None and temporal_mix > 0:
            prev_resized = prev_styled.resize(init_img.size)
            init_img = Image.blend(init_img, prev_resized, temporal_mix)

        generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

        result = pipe(
            prompt=cfg["prompt"],
            negative_prompt=cfg.get("negative_prompt", ""),
            image=init_img,
            control_image=control_img,
            strength=float(cfg["strength"]),
            num_inference_steps=int(cfg["steps"]),
            guidance_scale=float(cfg["guidance"]),
            controlnet_conditioning_scale=float(cfg.get("conditioning_scale", 1.0)),
            generator=generator,
        ).images[0]

        result.save(out)
        prev_styled = result
        print(f"[stylize] {i}/{len(frames)} -> {out.name}")


def assemble_video(styled_dir: Path, output_dir: Path, fps: int, audio_path: Path | None):
    silent = output_dir / "styled_silent.mp4"
    final = output_dir / "styled_final.mp4"

    run_cmd([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(styled_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-crf", "18",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(silent),
    ])

    if audio_path and audio_path.exists():
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(silent),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            str(final),
        ])
        return final

    return silent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--check", action="store_true", help="Check tools and model without rendering")
    ap.add_argument("--prepare-only", action="store_true", help="Verify the video workflow without inference")
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL))
    ap.add_argument("--config", default=str(ROOT / "configs/default.json"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--preview-seconds", type=float, default=5.0)
    ap.add_argument("--full", action="store_true", help="Process full clip (or --duration)")
    args = ap.parse_args()

    if not args.check and not args.input:
        ap.error("--input is required unless --check is used")
    input_video = Path(args.input).expanduser().resolve() if args.input else None
    config_path = Path(args.config).expanduser().resolve()
    model_path = str(Path(args.model_path).expanduser().resolve())

    if input_video is not None and not input_video.is_file():
        raise FileNotFoundError(input_video)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    cfg = load_config(config_path)
    validate_config(cfg)
    backend = model_backend(Path(model_path))
    preflight(Path(model_path), backend, cfg)
    if args.check:
        print("Checks passed")
        return
    source_duration = ffprobe_duration(input_video)
    duration = args.duration if (args.full or args.duration is not None) else args.preview_seconds
    if not math.isfinite(args.start) or not 0 <= args.start < source_duration:
        ap.error("--start must fall within the video")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        ap.error("duration must be positive")
    duration = min(duration or source_duration, source_duration - args.start)

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "runs" / (stamp + "_" + datetime.now().strftime("%f"))
        run_dir = run_dir.resolve()

    run_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_run_dirs(run_dir)

    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_video),
        "source_duration_seconds": source_duration,
        "start_seconds": args.start,
        "duration_seconds": duration,
        "model_path": model_path,
        "backend": backend,
        "input_size": input_video.stat().st_size,
        "input_mtime_ns": input_video.stat().st_mtime_ns,
        "config": cfg,
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in manifest:
            if key != "started_at" and previous.get(key) != manifest[key]:
                raise ValueError(f"Cannot resume: {key} changed; choose a new --run-dir")
    elif any(dirs["styled"].iterdir()):
        raise ValueError("Refusing styled frames without a matching manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[1/6] extracting frames")
    extract_frames(input_video, dirs["frames"], int(cfg["fps"]), int(cfg["size"]), args.start, duration)

    print("\n[2/6] extracting audio")
    audio_path = run_dir / "audio.m4a"
    has_audio = extract_audio(input_video, audio_path, args.start, duration)

    if not list(dirs["frames"].glob("frame_*.png")):
        raise RuntimeError("FFmpeg extracted no frames")
    if args.prepare_only:
        final = assemble_video(dirs["frames"], dirs["output"], int(cfg["fps"]), audio_path if has_audio else None)
        prepared = dirs["output"] / "prepared_preview.mp4"
        final.replace(prepared)
        print(f"Prepared preview (not stylized): {prepared}")
        return
    if backend == "coreml":
        stylize_coreml(Path(model_path), dirs, cfg, run_dir)
    else:
        generate_canny(dirs["frames"], dirs["control"], int(cfg["canny_low"]), int(cfg["canny_high"]))
        pipe = load_pipeline(model_path, cfg["controlnet_model"])
        stylize_frames(pipe, dirs["frames"], dirs["control"], dirs["styled"], cfg)

    print("\n[6/6] assembling video")
    final_video = assemble_video(dirs["styled"], dirs["output"], int(cfg["fps"]), audio_path if has_audio else None)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["final_video"] = str(final_video)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDONE")
    print(final_video)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Resume with the same --run-dir.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
