"""
Configuration for the Semantic Feature Extraction Pipeline.

Centralizes all configurable parameters for SAM 2 mask generation,
CLIP feature encoding, and output file management.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class SemanticConfig:
    """Configuration for semantic feature extraction.

    Attributes:
        dataset_path: Root directory of the D-NeRF scene
            (must contain transforms_train.json and train/ image folder).
        output_dir: Where to save extracted masks, labels, and features.
            Defaults to <dataset_path>/semantics/.
        
        sam2_checkpoint: Path to SAM 2 model checkpoint file.
        sam2_model_cfg: SAM 2 model architecture config identifier.
            Options: "sam2_hiera_tiny", "sam2_hiera_small",
                     "sam2_hiera_base_plus", "sam2_hiera_large".
        sam_points_per_side: Grid density for SAM auto mask generator
            (higher = more masks, slower). Default 32.
        sam_pred_iou_thresh: Minimum predicted IoU for a mask to be kept.
        sam_stability_score_thresh: Minimum stability score for mask filtering.
        
        clip_model_name: CLIP model variant for feature extraction.
            Examples: "ViT-B-32", "ViT-B-16", "ViT-L-14".
        clip_pretrained: Pretrained dataset for open_clip.
            Default "openai" for original OpenAI weights.
        clip_feature_dim: Dimensionality of CLIP feature vectors.
            512 for ViT-B/32, 768 for ViT-L/14.
        
        text_prompts: Text descriptions for open-vocabulary semantic labeling.
            Each prompt becomes a class; pixels are labeled by max cosine
            similarity between their CLIP feature and text embeddings.
            Include a "background" prompt for unlabeled regions.
        
        mask_threshold: Minimum mask area ratio to image area. Masks
            smaller than this fraction are discarded.
        min_mask_area: Minimum mask area in pixels (absolute).
        max_masks_per_frame: Maximum number of masks to retain per frame
            (sorted by predicted IoU descending).
        
        device: PyTorch device string ("cuda", "cuda:0", "cpu").
        batch_size: Number of frames to process in a single batch for
            CLIP encoding (GPU memory dependent).
        image_size: Expected input image resolution (D-NeRF default: 800).
        
        save_float16: If True, save CLIP features as float16 to halve
            disk usage (~800KB vs ~1.6MB per frame at 800x800).
        process_test_frames: If True, also extract semantics for test frames.
        verbose: If True, print detailed per-frame progress info.
    """

    # ── Dataset paths ────────────────────────────────────────────────
    dataset_path: str = ""
    output_dir: Optional[str] = None

    # ── SAM 2 configuration ──────────────────────────────────────────
    sam2_checkpoint: str = "sam2_hiera_small.pt"
    sam2_model_cfg: str = "sam2_hiera_s"
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.7
    sam_stability_score_thresh: float = 0.85

    # ── Grounding DINO configuration ─────────────────────────────────
    use_grounding_dino: bool = False
    grounding_dino_config: Optional[str] = None
    grounding_dino_checkpoint: Optional[str] = None
    grounding_box_threshold: float = 0.35
    grounding_text_threshold: float = 0.25

    # ── CLIP configuration ───────────────────────────────────────────
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_feature_dim: int = 512

    # ── Semantic labeling ────────────────────────────────────────────
    text_prompts: List[str] = field(default_factory=lambda: [
        "background",
        "person",
        "ball",
        "floor",
        "object",
    ])

    # ── Mask filtering ───────────────────────────────────────────────
    mask_threshold: float = 0.001  # min mask area as fraction of image area
    min_mask_area: int = 100       # min mask area in pixels
    max_masks_per_frame: int = 64  # max masks retained per frame

    # ── Runtime ──────────────────────────────────────────────────────
    device: str = "cuda"
    batch_size: int = 4
    image_size: int = 800

    # ── Output options ───────────────────────────────────────────────
    save_float16: bool = True
    process_test_frames: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Validate and set derived paths."""
        if self.dataset_path:
            self.dataset_path = str(Path(self.dataset_path).resolve())
        if self.output_dir is None and self.dataset_path:
            self.output_dir = str(Path(self.dataset_path) / "semantics")

    @property
    def masks_dir(self) -> Path:
        return Path(self.output_dir) / "masks"

    @property
    def labels_dir(self) -> Path:
        return Path(self.output_dir) / "labels"

    @property
    def features_dir(self) -> Path:
        return Path(self.output_dir) / "features"

    @property
    def metadata_path(self) -> Path:
        return Path(self.output_dir) / "metadata.json"

    @property
    def clip_text_features_path(self) -> Path:
        return Path(self.output_dir) / "clip_text_features.npy"

    def create_output_dirs(self):
        """Create all output subdirectories."""
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize config to a JSON-compatible dictionary."""
        return {
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "sam2_model_cfg": self.sam2_model_cfg,
            "sam2_checkpoint": self.sam2_checkpoint,
            "clip_model_name": self.clip_model_name,
            "clip_pretrained": self.clip_pretrained,
            "clip_feature_dim": self.clip_feature_dim,
            "text_prompts": self.text_prompts,
            "sam_points_per_side": self.sam_points_per_side,
            "sam_pred_iou_thresh": self.sam_pred_iou_thresh,
            "sam_stability_score_thresh": self.sam_stability_score_thresh,
            "mask_threshold": self.mask_threshold,
            "min_mask_area": self.min_mask_area,
            "max_masks_per_frame": self.max_masks_per_frame,
            "image_size": self.image_size,
            "save_float16": self.save_float16,
            "use_grounding_dino": self.use_grounding_dino,
            "grounding_box_threshold": self.grounding_box_threshold,
            "grounding_text_threshold": self.grounding_text_threshold,
        }
