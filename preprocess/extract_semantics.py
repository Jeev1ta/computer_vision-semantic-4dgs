"""
Semantic Feature Extraction Pipeline using SAM 2 + CLIP.

Processes D-NeRF training (and optionally test) frames to produce:
  1. Per-frame object masks via SAM 2 automatic mask generation
  2. Per-mask CLIP image feature vectors
  3. Per-pixel semantic label maps (text-guided via CLIP similarity)
  4. Per-pixel dense CLIP feature maps for downstream supervision

Usage:
    python -m preprocess.extract_semantics \
        --dataset_path data/dnerf/bouncingballs \
        --text_prompts "ball" "floor" "background" \
        --sam2_checkpoint checkpoints/sam2_hiera_small.pt

    # Or from Python:
    from preprocess import SemanticConfig, SemanticExtractor
    config = SemanticConfig(dataset_path="data/dnerf/bouncingballs")
    extractor = SemanticExtractor(config)
    extractor.run()
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from preprocess.config import SemanticConfig


class SemanticExtractor:
    """Extracts semantic masks and CLIP features from D-NeRF dataset frames.

    This class orchestrates the full extraction pipeline:
    1. Loads frame metadata from transforms_train.json / transforms_test.json
    2. Runs SAM 2 automatic mask generation on each frame
    3. Encodes each masked region through CLIP's image encoder
    4. Optionally labels masks via text-prompt cosine similarity
    5. Saves masks, label maps, and dense CLIP feature maps to disk

    Attributes:
        config: SemanticConfig instance with all pipeline parameters.
        sam_model: Loaded SAM 2 model.
        sam_generator: SAM 2 automatic mask generator.
        clip_model: Loaded CLIP vision-language model.
        clip_preprocess: CLIP image preprocessing transform.
        tokenizer: CLIP text tokenizer.
        text_features: Precomputed CLIP text embeddings for prompts.
    """

    def __init__(self, config: SemanticConfig):
        self.config = config
        self.sam_model = None
        self.sam_generator = None
        self.clip_model = None
        self.clip_preprocess = None
        self.tokenizer = None
        self.text_features = None

        # Frame metadata loaded from transforms JSON
        self._train_frames: List[Dict] = []
        self._test_frames: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────────────────────────────

    def load_sam2(self):
        """Load SAM 2 model and create automatic mask generator."""
        print("[Phase 2] Loading SAM 2 model...")
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM 2 is not installed. Install with:\n"
                "  pip install segment-anything-2\n"
                "Or clone: https://github.com/facebookresearch/sam2"
            )

        sam2 = build_sam2(
            config_file=self.config.sam2_model_cfg,
            ckpt_path=self.config.sam2_checkpoint,
            device=self.config.device,
        )

        self.sam_model = sam2
        self.sam_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=self.config.sam_points_per_side,
            pred_iou_thresh=self.config.sam_pred_iou_thresh,
            stability_score_thresh=self.config.sam_stability_score_thresh,
            min_mask_region_area=self.config.min_mask_area,
        )
        print(f"  ✓ SAM 2 loaded ({self.config.sam2_model_cfg})")

    def load_clip(self):
        """Load CLIP model and precompute text features."""
        print("[Phase 2] Loading CLIP model...")
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is not installed. Install with:\n"
                "  pip install open-clip-torch"
            )

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
            device=self.config.device,
        )
        tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)

        self.clip_model = model.eval()
        self.clip_preprocess = preprocess
        self.tokenizer = tokenizer

        # Precompute text embeddings for semantic labeling
        if self.config.text_prompts:
            self._compute_text_features()

        print(f"  ✓ CLIP loaded ({self.config.clip_model_name}, "
              f"dim={self.config.clip_feature_dim})")

    def _compute_text_features(self):
        """Encode text prompts into normalized CLIP feature vectors."""
        prompts = self.config.text_prompts
        # Prepend "a photo of" for better CLIP zero-shot performance
        text_inputs = [f"a photo of {p}" for p in prompts]
        tokens = self.tokenizer(text_inputs).to(self.config.device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)

        self.text_features = text_features.cpu().numpy()  # (K, D)
        print(f"  ✓ Text features computed for {len(prompts)} prompts: {prompts}")

    # ─────────────────────────────────────────────────────────────────
    # Dataset Loading
    # ─────────────────────────────────────────────────────────────────

    def load_dataset_frames(self):
        """Read frame metadata from D-NeRF transforms JSON files."""
        dataset_path = Path(self.config.dataset_path)

        # Load training frames
        train_json = dataset_path / "transforms_train.json"
        if not train_json.exists():
            raise FileNotFoundError(
                f"transforms_train.json not found in {dataset_path}. "
                "Is this a valid D-NeRF dataset directory?"
            )

        with open(train_json, "r") as f:
            train_data = json.load(f)

        self._train_frames = []
        for frame in train_data["frames"]:
            file_path = frame["file_path"]
            # Resolve relative path and append extension if missing
            img_path = dataset_path / file_path
            if not img_path.suffix:
                img_path = img_path.with_suffix(".png")
            self._train_frames.append({
                "path": str(img_path),
                "name": Path(file_path).stem,
                "time": frame.get("time", 0.0),
            })

        print(f"  ✓ Loaded {len(self._train_frames)} training frames")

        # Optionally load test frames
        if self.config.process_test_frames:
            test_json = dataset_path / "transforms_test.json"
            if test_json.exists():
                with open(test_json, "r") as f:
                    test_data = json.load(f)
                self._test_frames = []
                for frame in test_data["frames"]:
                    file_path = frame["file_path"]
                    img_path = dataset_path / file_path
                    if not img_path.suffix:
                        img_path = img_path.with_suffix(".png")
                    self._test_frames.append({
                        "path": str(img_path),
                        "name": Path(file_path).stem,
                        "time": frame.get("time", 0.0),
                    })
                print(f"  ✓ Loaded {len(self._test_frames)} test frames")

    # ─────────────────────────────────────────────────────────────────
    # Core Processing
    # ─────────────────────────────────────────────────────────────────

    def _generate_masks(self, image_np: np.ndarray) -> List[Dict]:
        """Run mask generation on a single image via SAM 2 (or Grounding DINO + SAM 2).

        Args:
            image_np: RGB image as numpy array, shape (H, W, 3), uint8.

        Returns:
            List of mask dictionaries with segmentation, area, bbox, predicted_iou.
        """
        if getattr(self.config, 'use_grounding_dino', False):
            from preprocess.grounding_dino_sam import GroundingDINODetector, segment_with_grounding_dino_and_sam2
            if not hasattr(self, '_grounding_detector'):
                self._grounding_detector = GroundingDINODetector(
                    config_path=self.config.grounding_dino_config,
                    checkpoint_path=self.config.grounding_dino_checkpoint,
                    box_threshold=self.config.grounding_box_threshold,
                    text_threshold=self.config.grounding_text_threshold,
                    device=self.config.device,
                )
            # Create SAM2 predictor if needed
            if not hasattr(self, '_sam_predictor'):
                try:
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                    self._sam_predictor = SAM2ImagePredictor(self.sam_model)
                except Exception:
                    self._sam_predictor = None

            if self._sam_predictor is not None:
                text_query = " . ".join([p for p in self.config.text_prompts if p != "background"])
                gd_masks = segment_with_grounding_dino_and_sam2(
                    image=image_np,
                    text_prompt=text_query,
                    sam2_predictor=self._sam_predictor,
                    grounding_detector=self._grounding_detector,
                )
                if len(gd_masks) > 0:
                    for m in gd_masks:
                        if "bbox" not in m and "box" in m:
                            b = m["box"]
                            m["bbox"] = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                    return gd_masks

        masks = self.sam_generator.generate(image_np)

        # Filter by minimum area
        image_area = image_np.shape[0] * image_np.shape[1]
        min_area = max(
            self.config.min_mask_area,
            int(self.config.mask_threshold * image_area),
        )
        masks = [m for m in masks if m["area"] >= min_area]

        # Sort by predicted IoU (descending) and keep top-K
        masks.sort(key=lambda m: m["predicted_iou"], reverse=True)
        if len(masks) > self.config.max_masks_per_frame:
            masks = masks[: self.config.max_masks_per_frame]

        return masks

    def _extract_clip_features_for_masks(
        self,
        image_pil: Image.Image,
        masks: List[Dict],
    ) -> np.ndarray:
        """Extract CLIP image features for each masked region.

        For each mask, crops the image to the mask's bounding box,
        applies the mask to zero-out background pixels, and encodes
        the resulting crop through CLIP's image encoder.

        Args:
            image_pil: Original PIL image (RGB).
            masks: List of SAM mask dicts with 'segmentation' and 'bbox'.

        Returns:
            features: numpy array of shape (N_masks, D) with L2-normalized
                CLIP feature vectors.
        """
        image_np = np.array(image_pil)
        crops = []

        for mask_info in masks:
            seg = mask_info["segmentation"]  # (H, W) bool
            bbox = mask_info["bbox"]  # [x, y, w, h]
            x, y, w, h = [int(v) for v in bbox]

            # Crop image to bounding box
            crop = image_np[y : y + h, x : x + w].copy()

            # Apply mask — zero out background pixels
            mask_crop = seg[y : y + h, x : x + w]
            crop[~mask_crop] = 0

            # Convert to PIL for CLIP preprocessing
            crop_pil = Image.fromarray(crop)
            crops.append(crop_pil)

        if not crops:
            return np.zeros((0, self.config.clip_feature_dim), dtype=np.float32)

        # Batch encode through CLIP
        features_list = []
        for i in range(0, len(crops), self.config.batch_size):
            batch_crops = crops[i : i + self.config.batch_size]
            batch_tensors = torch.stack(
                [self.clip_preprocess(c) for c in batch_crops]
            ).to(self.config.device)

            with torch.no_grad():
                batch_features = self.clip_model.encode_image(batch_tensors)
                batch_features = F.normalize(batch_features, dim=-1)

            features_list.append(batch_features.cpu().numpy())

        return np.concatenate(features_list, axis=0)  # (N_masks, D)

    def _assign_labels(self, mask_features: np.ndarray) -> np.ndarray:
        """Assign semantic labels to masks via CLIP text similarity.

        Computes cosine similarity between each mask's CLIP feature
        and all text prompt features, assigning the argmax label.

        Args:
            mask_features: (N_masks, D) normalized CLIP features.

        Returns:
            labels: (N_masks,) integer label indices.
                Label 0 corresponds to text_prompts[0], etc.
        """
        if self.text_features is None or len(mask_features) == 0:
            return np.zeros(len(mask_features), dtype=np.int32)

        # Cosine similarity (features are already L2-normalized)
        similarity = mask_features @ self.text_features.T  # (N_masks, K)
        labels = similarity.argmax(axis=1)  # (N_masks,)
        return labels.astype(np.int32)

    def _build_label_map(
        self,
        masks: List[Dict],
        labels: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Compose per-mask labels into a per-pixel semantic label map.

        Resolves overlaps by priority: higher-IoU masks take precedence.
        Background (unlabeled) pixels receive label 0.

        Args:
            masks: List of SAM mask dicts (sorted by descending IoU).
            labels: (N_masks,) integer class labels.
            image_shape: (H, W) of the target label map.

        Returns:
            label_map: (H, W) uint8 array with per-pixel class IDs.
        """
        H, W = image_shape
        label_map = np.zeros((H, W), dtype=np.uint8)

        # Paint masks in reverse order (lowest IoU first), so highest
        # IoU masks overwrite and take priority
        for i in reversed(range(len(masks))):
            seg = masks[i]["segmentation"]
            label_map[seg] = labels[i] + 1  # +1 so 0 = unlabeled/background

        return label_map

    def _build_feature_map(
        self,
        masks: List[Dict],
        mask_features: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Build a dense per-pixel CLIP feature map from mask features.

        Each pixel receives the CLIP feature vector of the mask it
        belongs to. Overlapping regions use the highest-IoU mask's feature.
        Unlabeled pixels receive a zero vector.

        Args:
            masks: List of SAM mask dicts (sorted by descending IoU).
            mask_features: (N_masks, D) normalized CLIP features.
            image_shape: (H, W) of the target feature map.

        Returns:
            feature_map: (H, W, D) float32 or float16 per-pixel features.
        """
        H, W = image_shape
        D = self.config.clip_feature_dim
        feature_map = np.zeros((H, W, D), dtype=np.float32)

        # Paint in reverse IoU order (lowest first, highest overwrites)
        for i in reversed(range(len(masks))):
            if i >= len(mask_features):
                continue
            seg = masks[i]["segmentation"]
            feature_map[seg] = mask_features[i]

        if self.config.save_float16:
            feature_map = feature_map.astype(np.float16)

        return feature_map

    # ─────────────────────────────────────────────────────────────────
    # Per-Frame Processing
    # ─────────────────────────────────────────────────────────────────

    def process_frame(self, frame_info: Dict) -> Dict:
        """Process a single frame through the full extraction pipeline.

        Args:
            frame_info: Dict with 'path', 'name', and 'time' keys.

        Returns:
            result: Dict with keys 'masks', 'labels', 'features',
                'label_map', 'feature_map', 'num_masks'.
        """
        img_path = frame_info["path"]
        frame_name = frame_info["name"]

        # Load image
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]

        # Handle RGBA (D-NeRF synthetic images have alpha channel)
        if image_np.shape[2] == 4:
            alpha = image_np[:, :, 3:4].astype(np.float32) / 255.0
            rgb = image_np[:, :, :3].astype(np.float32)
            bg = np.ones_like(rgb) * 255.0  # white background
            image_np = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            image_pil = Image.fromarray(image_np)

        # Step 1: Generate SAM masks
        masks = self._generate_masks(image_np)

        # Step 2: Extract CLIP features for each mask
        mask_features = self._extract_clip_features_for_masks(image_pil, masks)

        # Step 3: Assign semantic labels via text similarity
        labels = self._assign_labels(mask_features)

        # Step 4: Build per-pixel label map
        label_map = self._build_label_map(masks, labels, (H, W))

        # Step 5: Build dense per-pixel feature map
        feature_map = self._build_feature_map(masks, mask_features, (H, W))

        # ── Save outputs ─────────────────────────────────────────────

        # Save individual masks + metadata
        mask_data = {
            "num_masks": len(masks),
            "image_shape": [H, W],
            "frame_time": frame_info["time"],
        }
        mask_arrays = {}
        for i, m in enumerate(masks):
            mask_arrays[f"mask_{i}"] = m["segmentation"].astype(np.uint8)
            mask_data[f"mask_{i}_bbox"] = [int(v) for v in m["bbox"]]
            mask_data[f"mask_{i}_area"] = int(m["area"])
            mask_data[f"mask_{i}_iou"] = float(m["predicted_iou"])
            mask_data[f"mask_{i}_stability"] = float(m["stability_score"])
            mask_data[f"mask_{i}_label"] = int(labels[i]) if i < len(labels) else 0

        # Save masks as compressed npz
        masks_path = self.config.masks_dir / f"{frame_name}_masks.npz"
        np.savez_compressed(str(masks_path), **mask_arrays, metadata=json.dumps(mask_data))

        # Save label map as PNG (uint8, values are class IDs)
        labels_path = self.config.labels_dir / f"{frame_name}_labels.png"
        label_img = Image.fromarray(label_map, mode="L")
        label_img.save(str(labels_path))

        # Save dense CLIP feature map
        features_path = self.config.features_dir / f"{frame_name}_features.npy"
        np.save(str(features_path), feature_map)

        return {
            "frame_name": frame_name,
            "num_masks": len(masks),
            "label_map_shape": label_map.shape,
            "feature_map_shape": feature_map.shape,
            "unique_labels": np.unique(label_map).tolist(),
        }

    # ─────────────────────────────────────────────────────────────────
    # Main Pipeline
    # ─────────────────────────────────────────────────────────────────

    def run(self):
        """Execute the full semantic extraction pipeline.

        Steps:
            1. Load dataset frame metadata
            2. Initialize SAM 2 and CLIP models
            3. Create output directories
            4. Process all training (and optionally test) frames
            5. Save global metadata and text features
        """
        print("=" * 60)
        print("  Semantic Feature Extraction Pipeline (Phase 2)")
        print("=" * 60)
        print(f"  Dataset: {self.config.dataset_path}")
        print(f"  Output:  {self.config.output_dir}")
        print(f"  Device:  {self.config.device}")
        print()

        # Step 1: Load dataset
        print("[1/5] Loading dataset frames...")
        self.load_dataset_frames()

        # Step 2: Load models
        print("\n[2/5] Loading models...")
        self.load_sam2()
        self.load_clip()

        # Step 3: Create output directories
        print("\n[3/5] Creating output directories...")
        self.config.create_output_dirs()

        # Step 4: Process frames
        all_frames = [("train", self._train_frames)]
        if self.config.process_test_frames and self._test_frames:
            all_frames.append(("test", self._test_frames))

        total_masks = 0
        total_frames = 0

        for split_name, frames in all_frames:
            print(f"\n[4/5] Processing {split_name} frames ({len(frames)} frames)...")
            for frame_info in tqdm(frames, desc=f"  {split_name}", unit="frame"):
                try:
                    result = self.process_frame(frame_info)
                    total_masks += result["num_masks"]
                    total_frames += 1

                    if self.config.verbose:
                        tqdm.write(
                            f"    {result['frame_name']}: "
                            f"{result['num_masks']} masks, "
                            f"labels={result['unique_labels']}"
                        )
                except Exception as e:
                    warnings.warn(
                        f"Failed to process frame {frame_info['name']}: {e}"
                    )
                    continue

        # Step 5: Save global metadata
        print(f"\n[5/5] Saving global metadata...")
        self._save_metadata(total_frames, total_masks)

        # Save text features
        if self.text_features is not None:
            np.save(str(self.config.clip_text_features_path), self.text_features)
            print(f"  ✓ Text features saved: {self.config.clip_text_features_path}")

        print()
        print("=" * 60)
        print(f"  ✓ Extraction complete!")
        print(f"    Frames processed: {total_frames}")
        print(f"    Total masks generated: {total_masks}")
        print(f"    Avg masks per frame: {total_masks / max(total_frames, 1):.1f}")
        print(f"    Output directory: {self.config.output_dir}")
        print("=" * 60)

    def _save_metadata(self, total_frames: int, total_masks: int):
        """Save global pipeline metadata to metadata.json."""
        metadata = {
            "pipeline": "Semantic Feature Extraction (Phase 2)",
            "config": self.config.to_dict(),
            "statistics": {
                "total_frames_processed": total_frames,
                "total_masks_generated": total_masks,
                "avg_masks_per_frame": total_masks / max(total_frames, 1),
                "train_frames": len(self._train_frames),
                "test_frames": len(self._test_frames),
            },
            "output_format": {
                "masks": "NPZ with binary mask arrays + JSON metadata",
                "labels": "PNG uint8 label maps (0=unlabeled, 1..K=classes)",
                "features": f"NPY float{'16' if self.config.save_float16 else '32'} "
                            f"(H, W, {self.config.clip_feature_dim})",
                "clip_text_features": f"NPY float32 ({len(self.config.text_prompts)}, "
                                      f"{self.config.clip_feature_dim})",
            },
            "class_mapping": {
                str(i + 1): name
                for i, name in enumerate(self.config.text_prompts)
            },
        }

        with open(str(self.config.metadata_path), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Metadata saved: {self.config.metadata_path}")


# ─────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Extract semantic masks and CLIP features from D-NeRF frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction with default prompts
  python -m preprocess.extract_semantics \\
      --dataset_path data/dnerf/bouncingballs \\
      --sam2_checkpoint checkpoints/sam2_hiera_small.pt

  # Custom text prompts for a specific scene
  python -m preprocess.extract_semantics \\
      --dataset_path data/dnerf/standup \\
      --text_prompts "person" "floor" "background" \\
      --sam2_checkpoint checkpoints/sam2_hiera_small.pt

  # Lightweight / fast extraction
  python -m preprocess.extract_semantics \\
      --dataset_path data/dnerf/lego \\
      --sam_points_per_side 16 \\
      --max_masks_per_frame 32 \\
      --sam2_checkpoint checkpoints/sam2_hiera_small.pt
        """,
    )

    # Required
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to D-NeRF dataset root (contains transforms_train.json)",
    )

    # SAM 2
    parser.add_argument(
        "--sam2_checkpoint", type=str, default="sam2_hiera_small.pt",
        help="Path to SAM 2 checkpoint file",
    )
    parser.add_argument(
        "--sam2_model_cfg", type=str, default="sam2_hiera_s",
        help="SAM 2 model config name",
    )
    parser.add_argument(
        "--sam_points_per_side", type=int, default=32,
        help="Grid density for SAM automatic mask generation",
    )

    # CLIP
    parser.add_argument(
        "--clip_model_name", type=str, default="ViT-B-32",
        help="CLIP model variant (e.g., ViT-B-32, ViT-B-16, ViT-L-14)",
    )
    parser.add_argument(
        "--text_prompts", nargs="+", type=str,
        default=["background", "person", "ball", "floor", "object"],
        help="Text prompts for semantic labeling",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: <dataset_path>/semantics/)",
    )

    # Filtering
    parser.add_argument(
        "--max_masks_per_frame", type=int, default=64,
        help="Maximum masks to keep per frame",
    )

    # Grounding DINO
    parser.add_argument(
        "--use_grounding_dino", action="store_true",
        help="Use Grounding DINO for text-prompted box detection before SAM 2",
    )
    parser.add_argument(
        "--grounding_dino_config", type=str, default=None,
        help="Path to Grounding DINO config file",
    )
    parser.add_argument(
        "--grounding_dino_checkpoint", type=str, default=None,
        help="Path to Grounding DINO checkpoint file",
    )

    # Runtime
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for CLIP encoding",
    )
    parser.add_argument(
        "--no_test", action="store_true",
        help="Skip test frame processing",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print per-frame details",
    )

    return parser.parse_args()


def main():
    """CLI entry point for semantic extraction."""
    args = parse_args()

    config = SemanticConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        sam_points_per_side=args.sam_points_per_side,
        clip_model_name=args.clip_model_name,
        text_prompts=args.text_prompts,
        max_masks_per_frame=args.max_masks_per_frame,
        use_grounding_dino=args.use_grounding_dino,
        grounding_dino_config=args.grounding_dino_config,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        process_test_frames=not args.no_test,
        verbose=args.verbose,
    )

    extractor = SemanticExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()
