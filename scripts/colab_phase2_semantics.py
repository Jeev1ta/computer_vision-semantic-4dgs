# %% [markdown]
# # ==============================================================================
# # 🧠 Phase 2: Semantic Feature Extraction (SAM 2 + OpenCLIP)
# # ==============================================================================
# # This self-contained script executes Phase 2 in a MODERN PyTorch runtime:
# #   1. GPU check & verification
# #   2. Clone Semantic 4D-GS repository
# #   3. Ensure Modern PyTorch (PyTorch >= 2.0, standard Google Colab runtime)
# #   4. Install SAM 2 (Segment Anything 2) and OpenCLIP dependencies
# #   5. Download official SAM 2 Hiera-Small model checkpoint
# #   6. Restore D-NeRF "lego" dataset from Google Drive (Phase 1 backup)
# #   7. Run Semantic Extraction on all training frames:
# #      Prompts: "background", "lego bulldozer body", "wheels", "arm", "bucket"
# #   8. Verify generated outputs (masks, labels, dense CLIP features, metadata.json)
# #   9. Visualize 3 sample semantic label maps with matplotlib class color legends
# #  10. Generate Ground-Truth (GT) evaluation masks for 15 test frames
# #  11. Archive and backup semantics data to Google Drive
# #
# # NOTE: Run this script in a separate Colab session with standard PyTorch 2.x!
# # (SAM 2 requires PyTorch >= 2.0 with scaled_dot_product_attention support)
# # ==============================================================================

# %% [Cell 1: GPU Check]
# ==============================================================================
# Cell 1: Check GPU Environment
# ==============================================================================
# Verifies that an NVIDIA GPU (T4, V100, A100, or L4) is active for SAM 2 + CLIP inference.

!nvidia-smi

import torch

print(f"\n[GPU Check] PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"✓ GPU Active: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"✓ Device Capability: {torch.cuda.get_device_capability(0)}")
else:
    raise RuntimeError("❌ CUDA GPU not detected! Go to Runtime -> Change runtime type -> Select T4 or A100 GPU.")

# %% [Cell 2: Clone Repo]
# ==============================================================================
# Cell 2: Clone Semantic 4D-GS Repository
# ==============================================================================
# Clones repository to /content/computer_vision-semantic-4dgs and sets working directory.

REPO_URL = "https://github.com/Tenisha5132/computer_vision-semantic-4dgs.git"
REPO_DIR = "/content/computer_vision-semantic-4dgs"

import os

if not os.path.exists(REPO_DIR):
    print(f"[Setup] Cloning repository from {REPO_URL}...")
    !git clone {REPO_URL} {REPO_DIR}
else:
    print(f"[Setup] Repository already present at {REPO_DIR}")

# Change directory
try:
    %cd {REPO_DIR}
except Exception:
    pass

os.chdir(REPO_DIR)
print(f"✓ Active directory: {os.getcwd()}")

# %% [Cell 3: Install Modern PyTorch]
# ==============================================================================
# Cell 3: Ensure Modern PyTorch (PyTorch >= 2.0)
# ==============================================================================
# Segment Anything 2 (SAM 2) requires PyTorch >= 2.0 for FlashAttention / SDPA.
# Colab defaults to PyTorch 2.x; this cell verifies the environment meets criteria.

import torch
from packaging import version

torch_ver = torch.__version__.split("+")[0]
print(f"[Environment] Current PyTorch version: {torch.__version__}")

if version.parse(torch_ver) < version.parse("2.0.0"):
    print("[Environment] Updating PyTorch to modern release >= 2.0...")
    !pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    print("✓ PyTorch upgraded. (Note: Runtime restart may be required if prompted by Colab).")
else:
    print("✓ PyTorch >= 2.0 is confirmed ready for SAM 2.")

# %% [Cell 4: Install SAM 2 + OpenCLIP]
# ==============================================================================
# Cell 4: Install SAM 2 and OpenCLIP Packages
# ==============================================================================
# Installs:
#   - segment-anything-2: Meta's SAM 2 foundation segmentation model
#   - open-clip-torch: OpenCLIP vision-language model for semantic classification
#   - matplotlib, tqdm, pillow: visualization and progress utilities

print("[Packages] Installing SAM 2, OpenCLIP, and utilities...")
!pip install -q segment-anything-2 open-clip-torch matplotlib tqdm pillow

# Test imports
try:
    import sam2
    import open_clip
    print("✓ Successfully imported 'sam2' and 'open_clip'!")
except ImportError as e:
    raise ImportError(f"❌ Failed to import SAM 2 or OpenCLIP: {e}")

# %% [Cell 5: Download SAM 2 Checkpoint]
# ==============================================================================
# Cell 5: Download SAM 2 Model Checkpoint
# ==============================================================================
# Downloads the SAM 2 Hiera-Small weights checkpoint (~180 MB) into checkpoints/.
# Architecture: sam2_hiera_small (efficient and high accuracy for 3D/4D lifting)

import os
import urllib.request
from pathlib import Path

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "sam2_hiera_small.pt"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"

if not CKPT_PATH.exists() or CKPT_PATH.stat().st_size < 100 * 1024 * 1024:
    print(f"[SAM 2 Checkpoint] Downloading from {CKPT_URL}...")
    try:
        def _reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                pct = readsofar * 100 / totalsize
                print(f"\r  • Download progress: {pct:.1f}% ({readsofar/(1024*1024):.1f} MB)", end="")
        urllib.request.urlretrieve(CKPT_URL, str(CKPT_PATH), _reporthook)
        print(f"\n✓ Downloaded SAM 2 checkpoint to {CKPT_PATH} ({CKPT_PATH.stat().st_size / (1024*1024):.1f} MB)")
    except Exception as e:
        print(f"urllib failed ({e}); trying curl...")
        !curl -L -o {str(CKPT_PATH)} {CKPT_URL}
else:
    print(f"✓ SAM 2 checkpoint already present: {CKPT_PATH} ({CKPT_PATH.stat().st_size / (1024*1024):.1f} MB)")

# %% [Cell 6: Restore Data from Drive]
# ==============================================================================
# Cell 6: Mount Google Drive and Restore Phase 1 Data
# ==============================================================================
# Restores data/dnerf/lego from the Phase 1 backup tar archive on Google Drive.

import os
import tarfile
from pathlib import Path

DATA_LEGO = Path("data/dnerf/lego")

# Check if dataset already extracted
if not (DATA_LEGO / "transforms_train.json").exists():
    try:
        from google.colab import drive
        print("[Drive] Mounting Google Drive...")
        drive.mount('/content/drive')
        backup_archive = Path("/content/drive/MyDrive/4DGS_Backup/dnerf_lego_phase1.tar.gz")
    except Exception:
        backup_archive = Path("backups/dnerf_lego_phase1.tar.gz")

    if backup_archive.exists():
        print(f"[Drive] Restoring dataset from {backup_archive}...")
        with tarfile.open(backup_archive, "r:gz") as tar:
            tar.extractall(".")
        print(f"✓ Restored dataset to {DATA_LEGO.resolve()}")
    else:
        print(f"⚠️ Warning: Backup archive not found at {backup_archive}!")
        print("Downloading D-NeRF lego dataset directly as fallback...")
        !curl -L -o data_tmp.zip "https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1"
        import zipfile
        DATA_LEGO.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile("data_tmp.zip", "r") as zf:
            for member in zf.namelist():
                if "lego/" in member:
                    sub = member[member.find("lego/") + len("lego/"):]
                    if not sub: continue
                    tp = DATA_LEGO / sub
                    if member.endswith("/"):
                        tp.mkdir(parents=True, exist_ok=True)
                    else:
                        tp.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as s, open(tp, "wb") as t:
                            t.write(s.read())
        if os.path.exists("data_tmp.zip"): os.remove("data_tmp.zip")
        print(f"✓ Extracted dataset to {DATA_LEGO}")
else:
    print(f"✓ D-NeRF lego dataset already present at {DATA_LEGO.resolve()}")

# %% [Cell 7: Run Semantic Extraction]
%%time
# ==============================================================================
# Cell 7: Execute SAM 2 + OpenCLIP Semantic Extraction Pipeline
# ==============================================================================
# Runs preprocess.extract_semantics across all training frames:
#   1. Automatic 2D mask proposals via SAM 2 Hiera-Small
#   2. Region-level feature encoding with OpenCLIP ViT-B/32
#   3. Zero-shot semantic assignment using text prompts:
#      - "background"
#      - "lego bulldozer body"
#      - "wheels"
#      - "arm"
#      - "bucket"
#   4. Dense per-pixel label map & feature map composition

print("\n" + "=" * 70)
print("🔬 Running Semantic Extraction on D-NeRF 'lego' Scene...")
print("=" * 70)

!python -m preprocess.extract_semantics \
    --dataset_path data/dnerf/lego \
    --text_prompts "background" "lego bulldozer body" "wheels" "arm" "bucket" \
    --sam2_checkpoint checkpoints/sam2_hiera_small.pt

print("\n✓ Semantic extraction completed!")

# %% [Cell 8: Verify Outputs]
# ==============================================================================
# Cell 8: Verify Semantic Extraction Artifacts
# ==============================================================================
# Confirms generated outputs:
#   - data/dnerf/lego/semantics/metadata.json: Pipeline configuration & class mapping
#   - data/dnerf/lego/semantics/labels/: PNG uint8 label maps
#   - data/dnerf/lego/semantics/masks/: NPZ segmentation arrays
#   - data/dnerf/lego/semantics/features/: NPY dense CLIP feature maps

import json
from pathlib import Path

SEM_DIR = Path("data/dnerf/lego/semantics")
meta_path = SEM_DIR / "metadata.json"
labels_dir = SEM_DIR / "labels"
masks_dir = SEM_DIR / "masks"
feats_dir = SEM_DIR / "features"

assert meta_path.exists(), f"❌ Missing metadata.json at {meta_path}"
assert labels_dir.exists(), f"❌ Missing labels directory at {labels_dir}"

label_files = list(labels_dir.glob("*.png"))
mask_files = list(masks_dir.glob("*.npz"))
feat_files = list(feats_dir.glob("*.npy"))

with open(meta_path, "r") as f:
    metadata = json.load(f)

print("=" * 60)
print("✅ Phase 2 Semantic Extraction Verification")
print("=" * 60)
print(f"  • Metadata file:     {meta_path.name} (Valid JSON)")
print(f"  • Label maps (.png): {len(label_files)} frames")
print(f"  • Mask files (.npz): {len(mask_files)} frames")
print(f"  • CLIP maps  (.npy): {len(feat_files)} frames")
print(f"  • Semantic Classes:  {metadata.get('class_mapping', {})}")
print(f"  • Statistics:        {metadata.get('statistics', {})}")
print("=" * 60)
assert len(label_files) > 0, "No label map PNGs found!"

# %% [Cell 9: Visualize Masks]
# ==============================================================================
# Cell 9: Visualize Sample Extracted Semantic Masks with Matplotlib
# ==============================================================================
# Loads 3 sample label maps from data/dnerf/lego/semantics/labels/ and displays
# them with vibrant class color overlays and labeled legend items.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path

SEM_DIR = Path("data/dnerf/lego/semantics")
labels_dir = SEM_DIR / "labels"
label_paths = sorted(list(labels_dir.glob("*.png")))[:3]

# Vibrant semantic palette:
# 0: Unlabeled/Black, 1: Background, 2: Body, 3: Wheels, 4: Arm, 5: Bucket
SEMANTIC_COLORS = np.array([
    [20, 20, 20],      # 0: Unlabeled
    [80, 80, 80],      # 1: Background
    [255, 215, 0],     # 2: Lego Bulldozer Body (Gold/Yellow)
    [30, 144, 255],    # 3: Wheels (Dodger Blue)
    [255, 69, 0],      # 4: Arm (Orange-Red)
    [50, 205, 50],     # 5: Bucket (Lime Green)
    [186, 85, 211],    # 6: Reserved
], dtype=np.uint8)

CLASS_NAMES = [
    "Unlabeled / Dark",
    "Background",
    "Lego Bulldozer Body",
    "Wheels",
    "Arm",
    "Bucket",
]

fig, axes = plt.subplots(1, len(label_paths), figsize=(18, 6))
if len(label_paths) == 1:
    axes = [axes]

for i, lpath in enumerate(label_paths):
    lbl_img = np.array(Image.open(lpath))
    # Map uint8 label indices to RGB colors
    c_indices = np.clip(lbl_img, 0, len(SEMANTIC_COLORS) - 1)
    colored_map = SEMANTIC_COLORS[c_indices]
    
    axes[i].imshow(colored_map)
    axes[i].set_title(f"Label Map: {lpath.name}\n(Classes present: {np.unique(lbl_img)})", fontsize=11)
    axes[i].axis("off")

# Add class legend
patches = [
    mpatches.Patch(color=SEMANTIC_COLORS[cid] / 255.0, label=CLASS_NAMES[cid])
    for cid in range(len(CLASS_NAMES))
]
fig.legend(handles=patches, loc="lower center", ncol=len(CLASS_NAMES), bbox_to_anchor=(0.5, -0.05), fontsize=10)
plt.suptitle("SAM 2 + OpenCLIP Semantic Segmentation Maps (D-NeRF Lego)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [Cell 10: Create GT Labels for Evaluation]
%%time
# ==============================================================================
# Cell 10: Generate Ground-Truth (GT) Semantic Labels on Held-Out Test Frames
# ==============================================================================
# To evaluate semantic mIoU in Phase 3, this cell creates pseudo-ground-truth
# annotations for 15 evenly-spaced held-out test frames from transforms_test.json:
#   1. Reads test frame paths and camera indices from transforms_test.json
#   2. Selects 15 evenly-spaced test views
#   3. Runs SAM 2 automatic mask generator on each test frame
#   4. Encodes masks with OpenCLIP and assigns class IDs (same prompts)
#   5. Saves single-channel PNG label maps to data/dnerf/lego/semantics_gt/labels/
#      with filenames matching render.py index format ({idx:05d}.png).

import os
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from preprocess.config import SemanticConfig
from preprocess.extract_semantics import SemanticExtractor

DATA_ROOT = Path("data/dnerf/lego")
GT_DIR = DATA_ROOT / "semantics_gt"
GT_LABELS_DIR = GT_DIR / "labels"
GT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

test_json_path = DATA_ROOT / "transforms_test.json"
assert test_json_path.exists(), f"❌ Missing {test_json_path}"

with open(test_json_path, "r") as f:
    test_meta = json.load(f)

test_frames = test_meta.get("frames", [])
num_total_test = len(test_frames)
assert num_total_test > 0, "No frames in transforms_test.json!"

# Pick 15 evenly spaced test frames
NUM_GT_FRAMES = 15
sample_indices = np.linspace(0, num_total_test - 1, NUM_GT_FRAMES, dtype=int)
print(f"[GT Generation] Generating GT labels for {NUM_GT_FRAMES} test frames out of {num_total_test} total...")
print(f"Selected test indices: {sample_indices.tolist()}")

# Initialize semantic pipeline
TEXT_PROMPTS = ["background", "lego bulldozer body", "wheels", "arm", "bucket"]
config = SemanticConfig(
    dataset_path=str(DATA_ROOT),
    output_dir=str(GT_DIR),
    sam2_checkpoint="checkpoints/sam2_hiera_small.pt",
    text_prompts=TEXT_PROMPTS,
    device="cuda" if torch.cuda.is_available() else "cpu",
    process_test_frames=False,
    verbose=False,
)

extractor = SemanticExtractor(config)
extractor.load_sam2()
extractor.load_clip()

# Save metadata for semantics_gt
gt_metadata = {
    "pipeline": "Phase 2 Evaluation Ground Truth Generation",
    "num_frames": NUM_GT_FRAMES,
    "test_indices": sample_indices.tolist(),
    "class_mapping": {str(i + 1): name for i, name in enumerate(TEXT_PROMPTS)},
    "config": {"text_prompts": TEXT_PROMPTS},
}
with open(GT_DIR / "metadata.json", "w") as f:
    json.dump(gt_metadata, f, indent=2)

print("\nProcessing GT test frames with SAM 2 + OpenCLIP...")
for test_idx in tqdm(sample_indices, desc="Generating GT Labels"):
    frame = test_frames[test_idx]
    rel_path = frame["file_path"]
    if not rel_path.endswith(".png"):
        rel_path += ".png"
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    
    img_path = DATA_ROOT / rel_path
    if not img_path.exists():
        continue
    
    # Load and prepare image (handle alpha if synthetic RGBA)
    pil_img = Image.open(img_path).convert("RGBA")
    rgba = np.array(pil_img)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    bg = np.ones_like(rgb) * 255.0
    composed_rgb = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    image_np = composed_rgb
    image_pil = Image.fromarray(image_np)
    
    H, W = image_np.shape[:2]
    
    # SAM 2 mask generation
    masks = extractor._generate_masks(image_np)
    
    # CLIP feature extraction and prompt assignment
    if masks:
        mask_features = extractor._extract_clip_features_for_masks(image_pil, masks)
        labels = extractor._assign_labels(mask_features)
        label_map = extractor._build_label_map(masks, labels, (H, W))
    else:
        label_map = np.zeros((H, W), dtype=np.uint8)
    
    # Save as 1-channel uint8 PNG with filename format matching render.py test index
    out_name = f"{test_idx:05d}.png"
    out_file = GT_LABELS_DIR / out_name
    Image.fromarray(label_map.astype(np.uint8), mode="L").save(out_file)
    
    # Also save with frame stem name for dual pairing compatibility
    stem_name = f"{Path(rel_path).stem}_labels.png"
    Image.fromarray(label_map.astype(np.uint8), mode="L").save(GT_LABELS_DIR / stem_name)

saved_gt_files = list(GT_LABELS_DIR.glob("*.png"))
print(f"\n✓ Successfully created {len(saved_gt_files)} GT label maps in {GT_LABELS_DIR}")

# %% [Cell 11: Upload to Drive]
# ==============================================================================
# Cell 11: Mount Google Drive and Backup Semantics Data
# ==============================================================================
# Bundles:
#   1. data/dnerf/lego/semantics/ (training masks, labels, CLIP features, metadata)
#   2. data/dnerf/lego/semantics_gt/ (15 test GT label maps for mIoU evaluation)
# into /content/drive/MyDrive/4DGS_Backup/dnerf_lego_phase2_semantics.tar.gz.

import os
import tarfile
from pathlib import Path

try:
    from google.colab import drive
    print("[Drive] Mounting Google Drive...")
    drive.mount('/content/drive')
    backup_dir = Path("/content/drive/MyDrive/4DGS_Backup")
except Exception:
    backup_dir = Path("backups")

backup_dir.mkdir(parents=True, exist_ok=True)
archive_path = backup_dir / "dnerf_lego_phase2_semantics.tar.gz"

print(f"[Drive] Archiving semantics data to {archive_path}...")
with tarfile.open(archive_path, "w:gz") as tar:
    if os.path.exists("data/dnerf/lego/semantics"):
        print("  • Adding data/dnerf/lego/semantics/ ...")
        tar.add("data/dnerf/lego/semantics", arcname="data/dnerf/lego/semantics")
    if os.path.exists("data/dnerf/lego/semantics_gt"):
        print("  • Adding data/dnerf/lego/semantics_gt/ ...")
        tar.add("data/dnerf/lego/semantics_gt", arcname="data/dnerf/lego/semantics_gt")

archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
print(f"\n✓ Successfully uploaded Phase 2 semantics backup to: {archive_path} ({archive_size_mb:.1f} MB)")
print("Phase 2 Complete! Proceed to Script 3 (Legacy PyTorch runtime) for Semantic Training & Editing.")
