# %% [markdown]
# # ==============================================================================
# # 🚀 Phase 1: Baseline 4D Gaussian Splatting on D-NeRF "lego" Scene
# # ==============================================================================
# # This self-contained script executes Phase 1 of the Semantic 4D-GS pipeline:
# #   1. GPU environment validation (requires NVIDIA GPU, e.g., T4/V100/A100)
# #   2. Repository cloning and workspace configuration
# #   3. Installation of legacy PyTorch 1.13.1 + CUDA 11.7 (required by CUDA rasterizer)
# #   4. Dependencies and mmcv 1.6.0 compilation
# #   5. Custom CUDA rasterization & KNN submodules compilation
# #   6. D-NeRF dataset download & extraction (lego scene)
# #   7. Dataset structure verification (transforms_train.json)
# #   8. Baseline 4D-GS training (20,000 iterations: 3,000 coarse + 17,000 fine)
# #   9. Test set novel-view & 360-degree video rendering
# #  10. Quantitative metrics evaluation (PSNR, SSIM, LPIPS)
# #  11. Checkpoint & dataset backup to Google Drive
# #
# # Runtime Requirement: GPU runtime with CUDA support.
# # Config: arguments/dnerf/lego.py (inherits arguments/dnerf/dnerf_default.py)
# # K-Planes Resolution: [64, 64, 64, 25], Iterations: 20000, Coarse: 3000
# # ==============================================================================

# %% [Cell 1: GPU Check]
# ==============================================================================
# Cell 1: Check GPU Environment and CUDA Availability
# ==============================================================================
# 4D Gaussian Splatting relies on custom CUDA kernels that require a CUDA-capable GPU.
# This cell checks driver availability, GPU device model, and allocated memory.

!nvidia-smi

import os
import sys

try:
    import torch
    print(f"\n[GPU Check] PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_arch = torch.cuda.get_device_capability(0)
        print(f"✓ GPU Detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        print(f"✓ Compute Capability: {cuda_arch[0]}.{cuda_arch[1]}")
        print(f"✓ CUDA Available: True (CUDA version: {torch.version.cuda})")
    else:
        print("❌ WARNING: CUDA is not available! In Colab, go to 'Runtime' -> 'Change runtime type' and select GPU (T4 or A100).")
except ImportError:
    print("[GPU Check] PyTorch not yet imported; proceeding to setup.")

# %% [Cell 2: Clone & Setup]
# ==============================================================================
# Cell 2: Clone Repository & Set Working Directory
# ==============================================================================
# Clones the Semantic 4D-GS repository recursively to ensure submodules
# (depth-diff-gaussian-rasterization and simple-knn) are downloaded.

REPO_URL = "https://github.com/Tenisha5132/computer_vision-semantic-4dgs.git"
REPO_DIR = "/content/computer_vision-semantic-4dgs"

import os
import subprocess

if not os.path.exists(REPO_DIR):
    print(f"[Setup] Cloning repository recursively from {REPO_URL}...")
    !git clone --recursive {REPO_URL} {REPO_DIR}
else:
    print(f"[Setup] Repository directory already exists at {REPO_DIR}.")

# Navigate to repository directory
try:
    %cd {REPO_DIR}
except Exception:
    pass

import os
os.chdir(REPO_DIR)
print(f"✓ Working directory set to: {os.getcwd()}")

# Update submodules if necessary
if not os.path.exists(os.path.join(REPO_DIR, "submodules/simple-knn/setup.py")):
    print("[Setup] Initializing git submodules...")
    !git submodule update --init --recursive
else:
    print("✓ Submodules already initialized.")

# %% [Cell 3: Install Legacy PyTorch]
# ==============================================================================
# Cell 3: Install Legacy PyTorch 1.13.1 + CUDA 11.7
# ==============================================================================
# Google Colab defaults to PyTorch 2.x, but the 4D-GS custom rasterization CUDA kernels
# require PyTorch 1.13.1 + CUDA 11.7 to prevent C++ ABI and kernel dispatch mismatches.

print("[Install] Installing PyTorch 1.13.1 + cu117 wheels...")
!pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

import torch
print(f"\n✓ Installed PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})")
assert "1.13.1" in torch.__version__, f"Expected torch 1.13.1, got {torch.__version__}"
assert torch.cuda.is_available(), "CUDA must be available after torch installation!"

# %% [Cell 4: Install Dependencies]
# ==============================================================================
# Cell 4: Install Dependencies & MMCV 1.6.0
# ==============================================================================
# 1. Installs project requirements (plyfile, lpips, pytorch_msssim, open3d, etc.)
# 2. Re-verifies PyTorch 1.13.1 was not overridden by unpinned dependencies
# 3. Installs MMCV 1.6.0 pre-built wheel for cu117/torch1.13 (used for config parsing)

print("[Dependencies] Installing requirements from requirements.txt...")
!pip install -r requirements.txt

# Guard against accidental torch downgrade/upgrade from requirements
print("\n[Dependencies] Ensuring torch==1.13.1+cu117 remains active...")
!pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

print("\n[Dependencies] Installing pre-built MMCV 1.6.0 for CUDA 11.7...")
!pip install mmcv==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Verify MMCV installation
try:
    import mmcv
    print(f"✓ MMCV version: {mmcv.__version__}")
except ImportError as e:
    raise ImportError(f"❌ Failed to import mmcv: {e}")

# %% [Cell 5: Compile CUDA Extensions]
# ==============================================================================
# Cell 5: Compile Custom CUDA Extensions
# ==============================================================================
# Compiles:
#   1. depth-diff-gaussian-rasterization: Differentiable 4D Gaussian rasterizer
#   2. simple-knn: Fast nearest-neighbor computation for Gaussian initialization

print("[CUDA Extensions] Compiling submodules/depth-diff-gaussian-rasterization...")
!pip install -e submodules/depth-diff-gaussian-rasterization

print("\n[CUDA Extensions] Compiling submodules/simple-knn...")
!pip install -e submodules/simple-knn

# Verify compiled modules can be imported
try:
    import diff_gaussian_rasterization
    import simple_knn
    print("\n✓ Successfully imported 'diff_gaussian_rasterization' and 'simple_knn'!")
except ImportError as e:
    raise RuntimeError(f"❌ Failed to import compiled CUDA extensions: {e}")

# %% [Cell 6: Download D-NeRF Data]
# ==============================================================================
# Cell 6: Download D-NeRF Dataset and Extract "lego" Scene
# ==============================================================================
# Downloads the standard D-NeRF synthetic benchmark zip archive (~1.5 GB total),
# and selectively extracts only the 'lego' scene files into data/dnerf/lego/.

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path("data/dnerf/lego")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = Path("data/dnerf_data.zip")

DROPBOX_URL = "https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1"

transforms_file = DATA_DIR / "transforms_train.json"

if not transforms_file.exists():
    print(f"[Data] Downloading D-NeRF dataset from {DROPBOX_URL}...")
    try:
        # Download with progress indicator
        def _reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\r[Data] Progress: {percent:5.1f}% ({readsofar / (1024*1024):.1f} MB / {totalsize / (1024*1024):.1f} MB)"
                sys.stderr.write(s)
                if readsofar >= totalsize:
                    sys.stderr.write("\n")
            else:
                sys.stderr.write(f"\r[Data] Downloaded: {readsofar / (1024*1024):.1f} MB")

        urllib.request.urlretrieve(DROPBOX_URL, str(ZIP_PATH), _reporthook)
        print("✓ Download complete. Extracting 'lego' scene...")
    except Exception as e:
        print(f"Standard download failed ({e}), attempting fallback curl...")
        !curl -L -o {str(ZIP_PATH)} "https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1"

    # Extract only lego files into data/dnerf/lego
    print(f"[Data] Inspecting {ZIP_PATH} and extracting 'lego' directory...")
    extracted_count = 0
    with zipfile.ZipFile(str(ZIP_PATH), "r") as zf:
        for member in zf.namelist():
            # Find any path containing 'lego/'
            if "lego/" in member:
                # Strip leading path components before 'lego/'
                sub_path = member[member.find("lego/") + len("lego/"):]
                if not sub_path:
                    continue
                target_path = DATA_DIR / sub_path
                if member.endswith("/"):
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                    extracted_count += 1

    print(f"✓ Extracted {extracted_count} files into {DATA_DIR}")
    # Remove archive to save disk space
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print("✓ Removed temporary zip archive.")
else:
    print(f"✓ D-NeRF lego dataset already exists at {DATA_DIR}")

# %% [Cell 7: Verify Data]
# ==============================================================================
# Cell 7: Verify Dataset Integrity and Frame Counts
# ==============================================================================
# Validates that transforms_train.json, transforms_val.json, and transforms_test.json
# exist and contain valid camera trajectories and timestamps.

import json
from pathlib import Path

DATA_DIR = Path("data/dnerf/lego")

train_meta_file = DATA_DIR / "transforms_train.json"
test_meta_file = DATA_DIR / "transforms_test.json"
val_meta_file = DATA_DIR / "transforms_val.json"

assert train_meta_file.exists(), f"❌ Missing train file: {train_meta_file}"
assert test_meta_file.exists(), f"❌ Missing test file: {test_meta_file}"

with open(train_meta_file, "r") as f:
    train_meta = json.load(f)

with open(test_meta_file, "r") as f:
    test_meta = json.load(f)

val_count = 0
if val_meta_file.exists():
    with open(val_meta_file, "r") as f:
        val_count = len(json.load(f).get("frames", []))

train_frames = train_meta.get("frames", [])
test_frames = test_meta.get("frames", [])
camera_angle_x = train_meta.get("camera_angle_x", 0.0)

print("=" * 60)
print("📊 D-NeRF 'lego' Dataset Verification")
print("=" * 60)
print(f"  • Scene root: {DATA_DIR.resolve()}")
print(f"  • Training frames:   {len(train_frames)}")
print(f"  • Testing frames:    {len(test_frames)}")
print(f"  • Validation frames: {val_count}")
print(f"  • Camera Angle X:    {camera_angle_x:.4f} rad")
if train_frames:
    sample = train_frames[0]
    print(f"  • Sample frame path: {sample.get('file_path')}")
    print(f"  • Sample timestamp:  {sample.get('time', 0.0)}")
print("=" * 60)
print("✓ Dataset verification successful! Ready for training.")

# %% [Cell 8: Train]
%%time
# ==============================================================================
# Cell 8: Train Baseline 4D Gaussian Splatting
# ==============================================================================
# Launches 4D-GS training using the official D-NeRF lego configuration:
#   - Config: arguments/dnerf/lego.py (inherits arguments/dnerf/dnerf_default.py)
#   - Total Iterations: 20,000 (3,000 coarse deformation warmup + 17,000 fine)
#   - K-Planes Resolution: [64, 64, 64, 25] (spatial xyz + temporal t grid)
#   - Output Checkpoint: output/dnerf/lego/
import os

try:
    %cd /content/computer_vision-semantic-4dgs
except Exception:
    pass

print("\n" + "=" * 70)
print("🏋️ Starting Baseline 4D-GS Training on D-NeRF 'lego' scene...")
print("=" * 70)

!python train.py \
    -s data/dnerf/lego \
    --port 6017 \
    --expname "dnerf/lego" \
    --configs arguments/dnerf/lego.py

print("\n✓ Baseline training complete! Checkpoint saved in output/dnerf/lego/")

# %% [Cell 9: Render]
# ==============================================================================
# Cell 9: Render Novel Views and Test Trajectory
# ==============================================================================
# Evaluates the trained 4D Gaussians by rendering all camera viewpoints:
#   - Generates rendered RGB images for test cameras: output/dnerf/lego/test/ours_20000/renders/
#   - Generates ground-truth comparison images: output/dnerf/lego/test/ours_20000/gt/
#   - Compiles video_rgb.mp4 showing dynamic scene evolution over time

print("\n" + "=" * 70)
print("🎬 Rendering Test Set and Novel Views...")
print("=" * 70)

!python render.py \
    --model_path output/dnerf/lego/ \
    --skip_train \
    --configs arguments/dnerf/lego.py

print("\n✓ Rendering completed successfully!")

# %% [Cell 10: Evaluate]
# ==============================================================================
# Cell 10: Quantitative Metrics Evaluation (PSNR, SSIM, LPIPS)
# ==============================================================================
# Computes quantitative reconstruction quality on the held-out test split:
#   - PSNR (Peak Signal-to-Noise Ratio): Pixel-level reconstruction fidelity
#   - SSIM (Structural Similarity Index): High-frequency structural quality
#   - LPIPS (Learned Perceptual Image Patch Similarity): Perceptual fidelity
# Outputs are saved to output/dnerf/lego/results.json.

import json
from pathlib import Path

print("\n" + "=" * 70)
print("📊 Computing Quantitative Metrics on Test Split...")
print("=" * 70)

!python metrics.py --model_paths output/dnerf/lego/

results_json = Path("output/dnerf/lego/results.json")
if results_json.exists():
    with open(results_json, "r") as f:
        metrics = json.load(f)

    print("\n" + "=" * 60)
    print("🏆 Phase 1 Baseline Results (D-NeRF 'lego'):")
    print("=" * 60)
    for model_name, sub in metrics.items():
        for method, vals in sub.items():
            print(f"Method: {method}")
            print(f"  • PSNR:      {vals.get('PSNR', 0.0):.2f} dB")
            print(f"  • SSIM:      {vals.get('SSIM', 0.0):.4f}")
            print(f"  • LPIPS-vgg: {vals.get('LPIPS-vgg', 0.0):.4f}")
    print("=" * 60)
else:
    print(f"Note: results.json not found at {results_json}; verify metrics.py standard output above.")

# %% [Cell 11: Save to Drive]
# ==============================================================================
# Cell 11: Mount Google Drive and Backup Checkpoints & Dataset
# ==============================================================================
# Mounts Google Drive and bundles:
#   1. Trained baseline model checkpoint: output/dnerf/lego/
#   2. D-NeRF lego dataset: data/dnerf/lego/
# into /content/drive/MyDrive/4DGS_Backup/dnerf_lego_phase1.tar.gz for Phase 2 & 3.

import os
import tarfile
from pathlib import Path

try:
    from google.colab import drive
    print("[Drive] Mounting Google Drive...")
    drive.mount('/content/drive')
    backup_dir = Path("/content/drive/MyDrive/4DGS_Backup")
except Exception as e:
    print(f"[Drive] Non-Colab environment or mounting skipped: {e}")
    backup_dir = Path("backups")

backup_dir.mkdir(parents=True, exist_ok=True)
archive_path = backup_dir / "dnerf_lego_phase1.tar.gz"

print(f"[Drive] Archiving 'output/dnerf/lego' and 'data/dnerf/lego' to {archive_path}...")
with tarfile.open(archive_path, "w:gz") as tar:
    if os.path.exists("output/dnerf/lego"):
        print("  • Adding output/dnerf/lego/ ...")
        tar.add("output/dnerf/lego", arcname="output/dnerf/lego")
    if os.path.exists("data/dnerf/lego"):
        print("  • Adding data/dnerf/lego/ ...")
        tar.add("data/dnerf/lego", arcname="data/dnerf/lego")

archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
print(f"\n✓ Successfully saved Phase 1 backup to: {archive_path} ({archive_size_mb:.1f} MB)")
print("Phase 1 Baseline Complete! Proceed to Script 2 for SAM 2 + OpenCLIP Semantic Extraction.")
