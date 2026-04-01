"""
Build MSTDN-A deployment ZIP packages.

Usage:
    py -3 build_package.py --variant gpu
    py -3 build_package.py --variant cpu
    py -3 build_package.py --variant both
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).parent  # MSTDN_A/
ROOT = BASE.parent            # Bineetha - emoDet/
DIST = ROOT / "dist"
ASSETS = BASE / "dist_assets"

# InsightFace models in user home
INSIGHTFACE_SRC = Path.home() / ".insightface" / "models" / "buffalo_l"

# HuggingFace wav2vec2 cache
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--wav2vec2-base"

# Files to include from MSTDN_A/
INCLUDE_FILES = [
    # Model code
    "models/__init__.py",
    "models/distillation.py",
    "models/heads/__init__.py",
    "models/heads/output_heads.py",
    "models/student/__init__.py",
    "models/student/student_model.py",
    "models/student/deep_audio_branch.py",
    "models/student/prosodic_branch.py",
    "models/student/spectral_branch.py",
    "models/student/speaker_gru.py",
    "models/student/temporal_transformer.py",
    # Dashboard server
    "dashboard/__init__.py",
    "dashboard/server.py",
    "dashboard/database.py",
    "dashboard/face_id.py",
    "dashboard/face_emotion.py",
    "dashboard/capture.py",
    "dashboard/pdf_report.py",
    # Dashboard UI
    "dashboard/static/index.html",
    "dashboard/static/yt_test.html",
    # FERPlus ONNX
    "dashboard/models/emotion-ferplus-8.onnx",
]

CHECKPOINTS = [
    "checkpoints/english_finetune_r2.pt",
    "checkpoints/online_tuned.pt",
]


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return dst.stat().st_size


def build_variant(variant: str):
    name = f"MSTDN-A_{variant.upper()}"
    out_dir = DIST / name
    print(f"\n{'='*60}")
    print(f"  Building {name}")
    print(f"{'='*60}\n")

    # Clean previous build
    if out_dir.exists():
        print(f"  Removing old {name}/...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    total = 0

    # 1. Copy project files
    print("  [1/6] Copying project files...")
    for rel in INCLUDE_FILES:
        src = BASE / rel
        if not src.exists():
            print(f"    SKIP (missing): {rel}")
            continue
        sz = copy_file(src, out_dir / rel)
        total += sz
    print(f"    -> {len(INCLUDE_FILES)} files")

    # 2. Copy checkpoints
    print("  [2/6] Copying checkpoints...")
    for rel in CHECKPOINTS:
        src = BASE / rel
        if not src.exists():
            print(f"    SKIP (missing): {rel}")
            continue
        print(f"    {rel} ({src.stat().st_size / 1e6:.0f} MB)...")
        sz = copy_file(src, out_dir / rel)
        total += sz

    # 3. Copy InsightFace models
    print("  [3/6] Copying InsightFace models...")
    if INSIGHTFACE_SRC.exists():
        dst_if = out_dir / "insightface_models" / "buffalo_l"
        dst_if.mkdir(parents=True, exist_ok=True)
        for f in INSIGHTFACE_SRC.iterdir():
            if f.suffix == ".onnx":
                print(f"    {f.name} ({f.stat().st_size / 1e6:.0f} MB)...")
                sz = copy_file(f, dst_if / f.name)
                total += sz
    else:
        print("    WARNING: InsightFace models not found at", INSIGHTFACE_SRC)

    # 4. Copy wav2vec2 cache
    print("  [4/6] Copying wav2vec2-base cache...")
    if HF_CACHE.exists():
        dst_hf = out_dir / "hf_cache" / "models--facebook--wav2vec2-base"
        shutil.copytree(str(HF_CACHE), str(dst_hf))
        hf_size = sum(f.stat().st_size for f in dst_hf.rglob("*") if f.is_file())
        total += hf_size
        print(f"    -> {hf_size / 1e6:.0f} MB")
    else:
        print("    WARNING: wav2vec2 cache not found at", HF_CACHE)

    # 5. Copy variant-specific scripts
    print("  [5/6] Copying scripts...")
    copy_file(ASSETS / "MSTDN-A.bat", out_dir / "MSTDN-A.bat")
    if variant == "gpu":
        copy_file(ASSETS / "setup_gpu.bat", out_dir / "setup.bat")
        copy_file(ASSETS / "requirements-gpu.txt", out_dir / "requirements-gpu.txt")
    else:
        copy_file(ASSETS / "setup_cpu.bat", out_dir / "setup.bat")
        copy_file(ASSETS / "requirements-cpu.txt", out_dir / "requirements-cpu.txt")

    # 6. Create README
    readme = out_dir / "README.txt"
    readme.write_text(f"""
MSTDN-A Emotion Detection System ({variant.upper()} Version)
{'='*50}

INSTALLATION:
  1. Install Python 3.10 or 3.11 from https://www.python.org
     IMPORTANT: Check "Add Python to PATH" during installation!
  2. Double-click setup.bat and wait for it to finish.
  3. Double-click MSTDN-A.bat to start.

DASHBOARDS:
  Main (Single):  http://localhost:8000
  Main (Multi):   http://localhost:8000/multi
  YT (Single):    http://localhost:8000/yt_test
  YT (Multi):     http://localhost:8000/yt_test/multi

STOPPING:
  Press Ctrl+C in the server window, or close it.

NOTES:
  - First startup may take 30-60 seconds (loading models).
  - {"GPU acceleration is used automatically." if variant == "gpu" else "CPU-only mode. Inference may be slower than GPU version."}
  - Online teaching corrections are saved in checkpoints/online_tuned.pt
""", encoding="utf-8")

    # 7. Create ZIP
    print("  [6/6] Creating ZIP archive...")
    zip_path = DIST / name
    shutil.make_archive(str(zip_path), "zip", str(DIST), name)
    zip_file = Path(f"{zip_path}.zip")
    zip_size = zip_file.stat().st_size

    print(f"\n  Total uncompressed: {total / 1e9:.2f} GB")
    print(f"  ZIP size:           {zip_size / 1e9:.2f} GB")
    print(f"  Output:             {zip_file}")
    print(f"\n  Done! {name}.zip ready.\n")

    return zip_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MSTDN-A deployment package")
    parser.add_argument("--variant", choices=["gpu", "cpu", "both"], default="gpu",
                        help="Which variant to build")
    args = parser.parse_args()

    DIST.mkdir(exist_ok=True)

    if args.variant == "both":
        build_variant("gpu")
        build_variant("cpu")
    else:
        build_variant(args.variant)
