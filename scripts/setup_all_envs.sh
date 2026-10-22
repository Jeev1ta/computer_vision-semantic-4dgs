#!/usr/bin/env bash
# scripts/setup_all_envs.sh
#
# Sets up all THREE separate environments recommended in guide section 5:
#   ENV A: gaussian_splatting   (official 3DGS, CUDA/torch)
#   ENV B: sam2                 (official SAM2, torch>=2.5.1)
#   ENV C: semantic3dgs-integration (this repo's glue code, no GPU needed)
#
# Run each stage independently -- don't run this blindly on a machine that
# doesn't have a matching GPU/CUDA setup. Read the official READMEs first;
# this script mirrors guide section 34's cheat sheet.

set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=============================================="
echo " Step 0: clone official repos (guide section 6)"
echo "=============================================="
mkdir -p third_party
if [ ! -d "third_party/gaussian-splatting" ]; then
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive third_party/gaussian-splatting
fi
if [ ! -d "third_party/sam2" ]; then
  git clone https://github.com/facebookresearch/sam2.git third_party/sam2
fi

echo "=============================================="
echo " ENV A: gaussian_splatting (guide section 7.1)"
echo "=============================================="
echo "Run manually on your GPU machine:"
echo "  cd third_party/gaussian-splatting"
echo "  conda env create --file environment.yml"
echo "  conda activate gaussian_splatting"

echo "=============================================="
echo " ENV B: sam2 (guide section 11.1)"
echo "=============================================="
echo "Run manually on your GPU machine:"
echo "  conda create -n sam2 python=3.10 -y"
echo "  conda activate sam2"
echo "  cd third_party/sam2 && pip install -e ."
echo "  cd checkpoints && bash download_ckpts.sh"

echo "=============================================="
echo " ENV C: semantic3dgs-integration (this repo)"
echo "=============================================="
conda create -n semantic3dgs-integration python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semantic3dgs-integration
pip install -r requirements-integration.txt

echo "Done. Three environments are set up (2 need manual GPU-specific steps above)."
