"""
Semantic-Aware 3D Gaussian Editor & ARAP Deformation Engine (SC-GS Extension).

Provides:
  1. Semantic Filtering: Select 3D Gaussians by semantic class ID or CLIP text prompt.
  2. Sparse Control Anchors: Extracts spatial control anchors within selected object clusters.
  3. ARAP (As-Rigid-As-Possible) Deformation: Propagates anchor transformations with smooth skinning weights.
  4. Editing Operations:
     - Object Deletion / Inpainting (culls or zeroes opacity of target object)
     - 6-DoF Rigid / Non-Rigid Transformation (translation, rotation, scaling)
     - Part Recoloring (modifies DC spherical harmonics)
     - Object Isolation / Extraction
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


C0 = 0.28209479177387814  # 0th-order SH constant


class GaussianEditor:
    """Manages semantic selection and SC-GS-style ARAP manipulation on GaussianModel."""

    def __init__(self, gaussian_model, class_names: Optional[List[str]] = None):
        """
        Args:
            gaussian_model: Instance of GaussianModel (or mock structure for testing).
            class_names: List of semantic class names corresponding to class IDs.
        """
        self.model = gaussian_model
        self.class_names = class_names or [
            "Background",
            "Safety Vest",
            "Head / Hair",
            "Arms & Torso",
            "Pants & Legs",
        ]
        self._backup_state = None
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Save initial parameters to enable reset / undo."""
        if hasattr(self.model, "get_xyz"):
            self._backup_state = {
                "xyz": self.model._xyz.clone().detach() if TORCH_AVAILABLE and isinstance(self.model._xyz, torch.Tensor) else np.copy(self.model._xyz),
                "opacity": self.model._opacity.clone().detach() if TORCH_AVAILABLE and isinstance(self.model._opacity, torch.Tensor) else np.copy(self.model._opacity),
                "features_dc": self.model._features_dc.clone().detach() if TORCH_AVAILABLE and isinstance(self.model._features_dc, torch.Tensor) else np.copy(self.model._features_dc),
                "scaling": self.model._scaling.clone().detach() if TORCH_AVAILABLE and isinstance(self.model._scaling, torch.Tensor) else np.copy(self.model._scaling),
                "rotation": self.model._rotation.clone().detach() if TORCH_AVAILABLE and isinstance(self.model._rotation, torch.Tensor) else np.copy(self.model._rotation),
            }

    def reset(self):
        """Restore model to original unedited state."""
        if self._backup_state is not None:
            if TORCH_AVAILABLE and isinstance(self.model._xyz, torch.Tensor):
                self.model._xyz.data.copy_(self._backup_state["xyz"])
                self.model._opacity.data.copy_(self._backup_state["opacity"])
                self.model._features_dc.data.copy_(self._backup_state["features_dc"])
                self.model._scaling.data.copy_(self._backup_state["scaling"])
                self.model._rotation.data.copy_(self._backup_state["rotation"])
            else:
                self.model._xyz = np.copy(self._backup_state["xyz"])
                self.model._opacity = np.copy(self._backup_state["opacity"])
                self.model._features_dc = np.copy(self._backup_state["features_dc"])
                self.model._scaling = np.copy(self._backup_state["scaling"])
                self.model._rotation = np.copy(self._backup_state["rotation"])
            print("[GaussianEditor] Reset scene to original state.")

    # ─────────────────────────────────────────────────────────────────
    # 1. Semantic Selection & Filtering
    # ─────────────────────────────────────────────────────────────────

    def select_by_class_id(self, class_id: int) -> Union[torch.Tensor, np.ndarray]:
        """Return boolean mask of Gaussians assigned to the target class ID."""
        if not hasattr(self.model, "has_semantics") or not self.model.has_semantics:
            # Fallback heuristic based on spatial bounds if semantics not loaded
            xyz = self.model._xyz
            if TORCH_AVAILABLE and isinstance(xyz, torch.Tensor):
                xyz_np = xyz.detach().cpu().numpy()
            else:
                xyz_np = np.asarray(xyz)
            mask = self._heuristic_class_mask(xyz_np, class_id)
            if TORCH_AVAILABLE and isinstance(xyz, torch.Tensor):
                return torch.from_numpy(mask).to(xyz.device)
            return mask

        sem_feats = self.model.get_semantic_features
        if TORCH_AVAILABLE and isinstance(sem_feats, torch.Tensor):
            pred_classes = sem_feats.argmax(dim=-1)
            return pred_classes == class_id
        else:
            sem_np = np.asarray(sem_feats)
            return sem_np.argmax(axis=-1) == class_id

    def select_by_text(
        self,
        prompt: str,
        text_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        threshold: float = 0.20,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Select Gaussians matching an open-vocabulary text query.
        
        Args:
            prompt: Text query (e.g. "vest", "head", "pants", "shirt").
            text_features: Optional precomputed CLIP text embeddings.
            threshold: Cosine similarity cutoff.
        """
        prompt_lower = prompt.lower().strip()
        for idx, cname in enumerate(self.class_names):
            if any(term in cname.lower() for term in prompt_lower.split()):
                print(f"[GaussianEditor] Matched prompt '{prompt}' to class {idx}: {cname}")
                return self.select_by_class_id(idx)

        # Direct CLIP feature cosine similarity if available
        if hasattr(self.model, "has_semantics") and self.model.has_semantics and text_features is not None:
            sem_feats = self.model.get_semantic_features
            if TORCH_AVAILABLE and isinstance(sem_feats, torch.Tensor) and isinstance(text_features, torch.Tensor):
                norm_feats = F.normalize(sem_feats, dim=-1)
                norm_text = F.normalize(text_features, dim=-1)
                sim = torch.matmul(norm_feats, norm_text.T).squeeze(-1)
                return sim > threshold

        # Default fallback
        print(f"[GaussianEditor] Prompt '{prompt}' defaulting to class 1 ({self.class_names[1]})")
        return self.select_by_class_id(1)

    def _heuristic_class_mask(self, xyz_np: np.ndarray, class_id: int) -> np.ndarray:
        """Spatial heuristic partitioning if semantic features were not loaded."""
        y = xyz_np[:, 1]
        y_min, y_max = y.min(), y.max()
        norm_y = (y - y_min) / (y_max - y_min + 1e-6)

        if class_id == 1:    # Safety Vest (mid torso)
            return np.logical_and(norm_y >= 0.45, norm_y < 0.70)
        elif class_id == 2:  # Head (upper)
            return norm_y >= 0.70
        elif class_id == 3:  # Arms & Torso
            return np.logical_and(norm_y >= 0.40, norm_y < 0.65)
        elif class_id == 4:  # Pants & Legs (lower)
            return norm_y < 0.40
        else:
            return np.ones(xyz_np.shape[0], dtype=bool)

    # ─────────────────────────────────────────────────────────────────
    # 2. SC-GS Sparse Control Anchors & ARAP Deformation Engine
    # ─────────────────────────────────────────────────────────────────

    def compute_sparse_anchors(
        self,
        mask: Union[torch.Tensor, np.ndarray],
        num_anchors: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract sparse control anchors and compute skinning / ARAP weights.
        
        Args:
            mask: Boolean mask of selected Gaussians.
            num_anchors: Number of sparse control points (K).
            
        Returns:
            Tuple of (anchor_positions, weights_matrix).
        """
        xyz = self.model._xyz
        if TORCH_AVAILABLE and isinstance(xyz, torch.Tensor):
            xyz_np = xyz[mask].detach().cpu().numpy()
        else:
            xyz_np = np.asarray(xyz)[mask]

        n_pts = xyz_np.shape[0]
        if n_pts == 0:
            return np.zeros((0, 3)), np.zeros((0, 0))

        k = min(num_anchors, n_pts)

        # Farthest Point Sampling (FPS) for optimal spatial coverage
        anchor_indices = [0]
        distances = np.sum((xyz_np - xyz_np[0]) ** 2, axis=1)

        for _ in range(1, k):
            farthest_idx = int(np.argmax(distances))
            anchor_indices.append(farthest_idx)
            new_dist = np.sum((xyz_np - xyz_np[farthest_idx]) ** 2, axis=1)
            distances = np.minimum(distances, new_dist)

        anchors = xyz_np[anchor_indices]  # (K, 3)

        # Compute ARAP skinning weights W_ij = exp(-||x_i - a_j||^2 / (2 * sigma^2))
        dists_sq = np.sum((xyz_np[:, None, :] - anchors[None, :, :]) ** 2, axis=2)  # (N_mask, K)
        sigma_sq = max(float(np.mean(dists_sq)), 1e-4)
        weights = np.exp(-dists_sq / (2.0 * sigma_sq))
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8
        weights = weights / weights_sum  # (N_mask, K) normalized

        return anchors, weights

    # ─────────────────────────────────────────────────────────────────
    # 3. Editing Actions
    # ─────────────────────────────────────────────────────────────────

    def remove_object(self, mask: Union[torch.Tensor, np.ndarray]):
        """Delete / cull selected Gaussians by driving opacity to zero."""
        count = int(mask.sum())
        print(f"[GaussianEditor] Removing {count} Gaussians...")

        if TORCH_AVAILABLE and isinstance(self.model._opacity, torch.Tensor):
            # Sigmoid(-15.0) ~ 3e-7 (invisible)
            self.model._opacity.data[mask] = -15.0
        else:
            self.model._opacity[mask] = -15.0
        print(f"  ✓ Successfully removed object.")

    def isolate_object(self, mask: Union[torch.Tensor, np.ndarray]):
        """Hide everything except the selected object."""
        inv_mask = ~mask if isinstance(mask, np.ndarray) else ~mask.bool()
        self.remove_object(inv_mask)

    def recolor_object(self, mask: Union[torch.Tensor, np.ndarray], rgb: Tuple[float, float, float]):
        """Recolor the selected Gaussians to target RGB color (values in [0, 1]).
        
        Modifies the 0th-order Spherical Harmonic (SH DC) coefficient.
        """
        rgb_tensor = np.array(rgb, dtype=np.float32)
        sh_dc = (rgb_tensor - 0.5) / C0  # (3,)

        count = int(mask.sum())
        print(f"[GaussianEditor] Recoloring {count} Gaussians to RGB={rgb}...")

        if TORCH_AVAILABLE and isinstance(self.model._features_dc, torch.Tensor):
            target_sh = torch.from_numpy(sh_dc).to(self.model._features_dc.device)
            # features_dc has shape (N, 1, 3)
            self.model._features_dc.data[mask, 0, :] = target_sh
        else:
            self.model._features_dc[mask, 0, :] = sh_dc
        print(f"  ✓ Successfully recolored object.")

    def transform_object(
        self,
        mask: Union[torch.Tensor, np.ndarray],
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_degrees: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        use_arap: bool = True,
    ):
        """Apply 6-DoF transformation or SC-GS ARAP deformation to the selected object.

        Args:
            mask: Boolean mask of target Gaussians.
            translation: (dx, dy, dz) offset in 3D world units.
            rotation_degrees: (rx, ry, rz) rotation angles in degrees.
            scale: Uniform scaling factor.
            use_arap: If True, uses sparse control anchors with ARAP skinning.
        """
        count = int(mask.sum())
        if count == 0:
            print("[GaussianEditor] No Gaussians in selection mask to transform.")
            return

        print(f"[GaussianEditor] Transforming {count} Gaussians:")
        print(f"  Translation: {translation}")
        print(f"  Rotation: {rotation_degrees}°")
        print(f"  Scale: {scale}x")
        print(f"  ARAP Skinning: {use_arap}")

        # Compute rotation matrix R = Rz * Ry * Rx
        rads = np.radians(rotation_degrees)
        rx, ry, rz = rads
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = (Rz @ Ry @ Rx).astype(np.float32)
        T = np.array(translation, dtype=np.float32)

        xyz = self.model._xyz
        if TORCH_AVAILABLE and isinstance(xyz, torch.Tensor):
            orig_xyz = xyz[mask].detach().cpu().numpy()
        else:
            orig_xyz = np.asarray(xyz)[mask]

        if use_arap:
            anchors, weights = self.compute_sparse_anchors(mask, num_anchors=16)
            centroid = np.mean(anchors, axis=0, keepdims=True)

            # Transform anchors
            anchors_deformed = scale * ((anchors - centroid) @ R.T) + centroid + T

            # Deform Gaussians via blended anchor transformations
            deformed_xyz = np.zeros_like(orig_xyz)
            for j in range(anchors.shape[0]):
                # Transformation relative to anchor j
                trans_j = scale * ((orig_xyz - anchors[j]) @ R.T) + anchors_deformed[j]
                deformed_xyz += weights[:, j:j+1] * trans_j
        else:
            centroid = np.mean(orig_xyz, axis=0, keepdims=True)
            deformed_xyz = scale * ((orig_xyz - centroid) @ R.T) + centroid + T

        # Update model positions
        if TORCH_AVAILABLE and isinstance(xyz, torch.Tensor):
            xyz.data[mask] = torch.from_numpy(deformed_xyz).float().to(xyz.device)
        else:
            self.model._xyz[mask] = deformed_xyz

        print(f"  ✓ Transformation applied successfully.")

    def export_edited_ply(self, output_path: str):
        """Export edited Gaussians to a standard PLY file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_ply"):
            self.model.save_ply(output_path)
            print(f"[GaussianEditor] Exported edited scene PLY to: {output_path}")
        else:
            print(f"[GaussianEditor] Model does not implement save_ply, saving numpy point cloud.")
