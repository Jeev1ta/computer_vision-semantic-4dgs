"""
Semantic Feature Extraction Pipeline for 4D Gaussian Splatting.

This package provides tools to extract 2D semantic masks and CLIP feature
embeddings from D-NeRF training frames using SAM 2 and CLIP. The extracted
features serve as ground-truth supervisory signals for semantic lifting
into the 4D Gaussian representation (Phase 3).

Usage:
    python -m preprocess.extract_semantics --dataset_path data/dnerf/bouncingballs
"""

from preprocess.config import SemanticConfig
from preprocess.extract_semantics import SemanticExtractor
from preprocess.grounding_dino_sam import GroundingDINODetector, segment_with_grounding_dino_and_sam2

__all__ = ["SemanticConfig", "SemanticExtractor", "GroundingDINODetector", "segment_with_grounding_dino_and_sam2"]
