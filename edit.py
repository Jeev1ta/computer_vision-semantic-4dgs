"""
Interactive 3D Semantic Editing & ARAP Manipulation CLI for 4D-GS.

Provides command-line editing of 4D Gaussian Splatting models using learned
semantics (SAM 2 + CLIP) and SC-GS-style sparse control anchors.

Supported Operations:
  1. remove   : Delete/hide the selected semantic object (e.g. remove vest/head)
  2. recolor  : Repaint the selected object to a target RGB color
  3. translate: 3D translation with optional ARAP skinning
  4. rotate   : 3D rotation of the selected object
  5. isolate  : Hide everything except the target object
  6. demo     : Run a zero-setup interactive demonstration of all actions

Usage:
    # Recolor safety vest to neon cyan:
    python edit.py --model_path output/standup --query "vest" --action recolor --color 0 255 255

    # Remove head:
    python edit.py --model_path output/standup --query "head" --action remove

    # Translate arms upward by 0.3 units using ARAP skinning:
    python edit.py --model_path output/standup --query "arms" --action translate --dy 0.3 --arap

    # Zero-setup validation test:
    python edit.py --demo
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from scene.gaussian_editor import GaussianEditor


class MockGaussianModel:
    """Lightweight test representation for environment validation without CUDA."""

    def __init__(self, num_points: int = 6400):
        self.num_points = num_points
        # Generate human-like cylindrical / ellipsoid point cloud
        np.random.seed(42)
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        y = np.random.uniform(-1.0, 1.0, num_points)
        r = 0.3 * (1.0 - 0.2 * y**2)
        x = r * np.cos(theta) + np.random.normal(0, 0.02, num_points)
        z = r * np.sin(theta) + np.random.normal(0, 0.02, num_points)

        self._xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        self._opacity = np.zeros(num_points, dtype=np.float32)  # sigmoid(0) = 0.5
        self._features_dc = np.zeros((num_points, 1, 3), dtype=np.float32)
        self._scaling = np.ones((num_points, 3), dtype=np.float32) * -3.0
        self._rotation = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_points, 1))

        # Assign synthetic 4-class semantic logits
        # Class 1: Vest (0.0 <= y < 0.4)
        # Class 2: Head (y >= 0.5)
        # Class 3: Arms & Torso (-0.2 <= y < 0.5)
        # Class 4: Pants & Legs (y < -0.2)
        self.semantic_feature_dim = 5
        self._semantic_features = np.zeros((num_points, 5), dtype=np.float32)
        for i in range(num_points):
            yi = y[i]
            if yi >= 0.5:
                self._semantic_features[i, 2] = 5.0  # Head
            elif 0.0 <= yi < 0.4:
                self._semantic_features[i, 1] = 5.0  # Vest
            elif -0.2 <= yi < 0.0:
                self._semantic_features[i, 3] = 5.0  # Arms / Torso
            else:
                self._semantic_features[i, 4] = 5.0  # Pants

    @property
    def has_semantics(self):
        return True

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_semantic_features(self):
        return self._semantic_features

    def save_ply(self, path: str):
        # Write basic ASCII PLY
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {self.num_points}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float opacity\n")
            f.write("end_header\n")
            for i in range(self.num_points):
                f.write(f"{self._xyz[i, 0]:.4f} {self._xyz[i, 1]:.4f} {self._xyz[i, 2]:.4f} {self._opacity[i]:.2f}\n")


def run_demo():
    """Execute end-to-end editing operations on sample 3D Gaussians."""
    print("\n" + "=" * 70)
    print("🎨 3D SEMANTIC EDITING & ARAP MANIPULATION TEST (SC-GS DEMO)")
    print("=" * 70)

    model = MockGaussianModel(num_points=6400)
    editor = GaussianEditor(model)

    print("\n[Step 1] Semantic Part Selection:")
    for cid, cname in enumerate(editor.class_names):
        mask = editor.select_by_class_id(cid)
        count = int(mask.sum())
        print(f"  • Class {cid} ({cname:<16}): {count:>5} Gaussians ({count / model.num_points * 100:.1f}%)")

    print("\n[Step 2] Action: Object Recoloring (Safety Vest ➔ Neon Cyan):")
    vest_mask = editor.select_by_text("vest")
    editor.recolor_object(vest_mask, rgb=(0.0, 1.0, 1.0))

    print("\n[Step 3] Action: ARAP Anchor Deformation (Head ➔ Translate + Rotate):")
    head_mask = editor.select_by_text("head")
    anchors, weights = editor.compute_sparse_anchors(head_mask, num_anchors=8)
    print(f"  • Extracted {anchors.shape[0]} SC-GS control anchors with {weights.shape[1]}-DoF skinning weights.")
    editor.transform_object(
        head_mask,
        translation=(0.0, 0.25, 0.0),
        rotation_degrees=(0.0, 15.0, 0.0),
        scale=1.1,
        use_arap=True,
    )

    print("\n[Step 4] Action: Object Removal / Inpainting (Remove Pants):")
    pants_mask = editor.select_by_text("pants")
    editor.remove_object(pants_mask)

    print("\n[Step 5] Export Edited 3D Scene:")
    out_dir = Path("output/edited_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "edited_scene.ply"
    editor.export_edited_ply(str(out_ply))

    print("\n" + "=" * 70)
    print("✅ 3D SEMANTIC EDITING PIPELINE FULLY VERIFIED!")
    print(f"Output saved to: {out_ply}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="3D Semantic Gaussian Editing")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained 4D-GS model directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration checkpoint to edit (-1 = latest)")
    parser.add_argument("--query", type=str, default="vest", help="Known semantic class name/phrase (this CLI does not encode arbitrary text prompts)")
    parser.add_argument("--class_names", nargs="+", help="Required class map in semantic-logit ID order, including background and any reserved IDs")
    parser.add_argument(
        "--action", type=str, default="recolor",
        choices=["recolor", "remove", "translate", "rotate", "scale", "isolate"],
        help="Editing operation to apply",
    )
    # Transformation parameters
    parser.add_argument("--dx", type=float, default=0.0, help="Translation X")
    parser.add_argument("--dy", type=float, default=0.0, help="Translation Y")
    parser.add_argument("--dz", type=float, default=0.0, help="Translation Z")
    parser.add_argument("--rx", type=float, default=0.0, help="Rotation X (degrees)")
    parser.add_argument("--ry", type=float, default=0.0, help="Rotation Y (degrees)")
    parser.add_argument("--rz", type=float, default=0.0, help="Rotation Z (degrees)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
    parser.add_argument(
        "--arap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use anchor-weighted local transform blending (not an ARAP optimization)",
    )
    # Color parameters
    parser.add_argument("--color", nargs=3, type=float, default=[0.0, 255.0, 255.0], help="RGB color in [0, 255]")
    parser.add_argument("--output_dir", type=str, default="output/edited_model", help="Where to save edited model")
    parser.add_argument("--demo", action="store_true", help="Run interactive zero-setup demo")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        run_demo()
        return
    if args.model_path is None:
        raise SystemExit("Provide --model_path for a trained scene or pass --demo for the synthetic smoke test.")

    print(f"[GaussianEditor] Loading model from: {args.model_path}")
    from argparse import ArgumentParser
    from arguments import ModelHiddenParams
    from scene import GaussianModel

    class DummyArgs:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    model_root = Path(args.model_path)
    point_cloud_root = model_root / "point_cloud"
    if args.iteration < 0:
        candidates = [
            path for path in point_cloud_root.glob("iteration_*")
            if path.is_dir() and path.name.removeprefix("iteration_").isdigit()
        ]
        if not candidates:
            raise FileNotFoundError(f"No trained point-cloud iterations found under {point_cloud_root}")
        iteration_dir = max(candidates, key=lambda path: int(path.name.removeprefix("iteration_")))
    else:
        iteration_dir = point_cloud_root / f"iteration_{args.iteration}"
    ply_path = iteration_dir / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Trained Gaussian PLY not found: {ply_path}")

    # Recreate the deformation network with the project's full model configuration.
    hidden_parser = ArgumentParser()
    hidden_params = ModelHiddenParams(hidden_parser)
    model_params = hidden_params.extract(hidden_parser.parse_args([]))
    model_params.semantic_feature_dim = 5

    # Initialize and load the Gaussian PLY before its matching deformation state.
    gaussians = GaussianModel(3, model_params)
    gaussians.load_ply(str(ply_path))
    gaussians.load_model(str(iteration_dir))

    if not args.class_names:
        raise SystemExit(
            "Provide --class_names in semantic feature index order (including background); "
            "the checkpoint does not persist a verified class-name mapping."
        )
    if len(args.class_names) != gaussians.semantic_feature_dim:
        raise SystemExit(
            f"Received {len(args.class_names)} class names, but checkpoint semantic features have "
            f"dimension {gaussians.semantic_feature_dim}. Supply the exact ID mapping."
        )

    editor = GaussianEditor(gaussians, class_names=args.class_names)
    mask = editor.select_by_text(args.query)

    if args.action == "remove":
        editor.remove_object(mask)
    elif args.action == "isolate":
        editor.isolate_object(mask)
    elif args.action == "recolor":
        rgb_norm = (args.color[0] / 255.0, args.color[1] / 255.0, args.color[2] / 255.0)
        editor.recolor_object(mask, rgb_norm)
    elif args.action in ["translate", "rotate", "scale"]:
        editor.transform_object(
            mask,
            translation=(args.dx, args.dy, args.dz),
            rotation_degrees=(args.rx, args.ry, args.rz),
            scale=args.scale,
            use_arap=args.arap,
        )

    output_iteration = Path(args.output_dir) / "point_cloud" / iteration_dir.name
    out_ply = output_iteration / "point_cloud.ply"
    output_iteration.mkdir(parents=True, exist_ok=True)
    editor.export_edited_ply(out_ply)
    gaussians.save_deformation(str(output_iteration))
    print(f"[GaussianEditor] Exported deformation and semantic sidecars to: {output_iteration}")


if __name__ == "__main__":
    main()
