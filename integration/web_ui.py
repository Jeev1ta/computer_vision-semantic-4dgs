"""
web_ui.py
---------
The lightweight web interface from guide section 19, Option B:

    Object: [ chair v ]   [Show object] [Hide object] [Show all]

Reads the semantic label file produced by run_semantic_pipeline.py and lets
the user interactively select an object. Deliberately NOT a custom WebGL
3D engine (guide: "Do not spend weeks making a custom WebGL viewer") --
it just re-renders the top-down colored scatter (visualize.plot_topdown)
for whichever subset of Gaussians is currently selected.

Run with:
    python -m integration.web_ui --semantic_dir outputs/semantic/my_scene
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .visualize import label_to_color


def _load_semantic(semantic_dir: str):
    semantic_dir = Path(semantic_dir)
    with open(semantic_dir / "object_metadata.json") as f:
        meta = json.load(f)
    label_names = {obj["id"]: obj["label"] for obj in meta["objects"]}
    unknown_label = meta.get("unknown_label", -1)

    xyz_path = semantic_dir / "gaussian_xyz.npy"
    if xyz_path.exists():
        xyz = np.load(xyz_path)
        labels = np.load(semantic_dir / "gaussian_labels.npy")
        confidence = np.load(semantic_dir / "gaussian_confidence.npy")
    else:
        # single-image (2.5D) mode persists a colored PLY instead of npys
        from .ply_compat import PlyData
        from .visualize import _PALETTE, _UNKNOWN_COLOR

        ply = PlyData.read(str(semantic_dir / "gaussian_cloud.ply"))
        v = ply.elements[0]
        xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
        rgb = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], axis=1)
        palette = np.array([_UNKNOWN_COLOR] + [tuple(c) for c in _PALETTE], dtype=np.int64)
        d = ((rgb.astype(np.int64)[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        labels = (d.argmin(axis=1) - 1).astype(np.int64)
        obj_conf = {o["id"]: float(o.get("score", 1.0)) for o in meta.get("segmentation", [])}
        confidence = np.array([obj_conf.get(int(l), 1.0) for l in labels])

    return xyz, labels, confidence, label_names, unknown_label


def launch(semantic_dir: str, min_confidence: float = 0.0):
    import gradio as gr

    xyz, labels, confidence, label_names, unknown_label = _load_semantic(semantic_dir)
    choices = ["(show all)"] + [label_names[i] for i in sorted(label_names)]

    def render(selected_label: str, hide_mode: bool, conf_threshold: float):
        keep = confidence >= conf_threshold
        sel_xyz, sel_labels = xyz[keep], labels[keep]

        if selected_label != "(show all)":
            name_to_id = {v: k for k, v in label_names.items()}
            obj_id = name_to_id[selected_label]
            if hide_mode:
                subset = sel_labels != obj_id
            else:
                subset = sel_labels == obj_id
            sel_xyz, sel_labels = sel_xyz[subset], sel_labels[subset]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            colors = label_to_color(sel_labels, unknown_label).astype(np.float32) / 255.0
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(sel_xyz[:, 0], sel_xyz[:, 2], c=colors, s=1.5, linewidths=0)
            ax.set_title(f"{len(sel_xyz)} Gaussians shown")
            ax.set_aspect("equal", adjustable="datalim")
            return fig
        except ImportError:
            from PIL import Image, ImageDraw

            img_w, img_h = 600, 600
            img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            if len(sel_xyz) > 0:
                pts_a = sel_xyz[:, 0]
                pts_b = sel_xyz[:, 2]

                min_a, max_a = pts_a.min(), pts_a.max()
                min_b, max_b = pts_b.min(), pts_b.max()

                pad = 0.1
                range_a = max(max_a - min_a, 1e-5) * (1 + 2 * pad)
                range_b = max(max_b - min_b, 1e-5) * (1 + 2 * pad)
                mid_a = (min_a + max_a) / 2
                mid_b = (min_b + max_b) / 2

                scale_a = (img_w - 80) / range_a
                scale_b = (img_h - 80) / range_b
                scale = min(scale_a, scale_b)

                colors = label_to_color(sel_labels, unknown_label)

                for i in range(len(sel_xyz)):
                    px = int((pts_a[i] - mid_a) * scale + img_w / 2)
                    py = int((mid_b - pts_b[i]) * scale + img_h / 2)
                    r = 3
                    draw.ellipse([px - r, py - r, px + r, py + r], fill=tuple(colors[i]))

            draw.text((20, 20), f"{len(sel_xyz)} Gaussians shown", fill=(0, 0, 0))
            return img

    with gr.Blocks(title="Semantic 3DGS Viewer") as demo:
        gr.Markdown("## Semantic-Aware 3DGS — Object Viewer\n"
                     "Select an object to highlight or hide it (guide section 19, Option B).")
        with gr.Row():
            obj_dropdown = gr.Dropdown(choices=choices, value="(show all)", label="Object")
            hide_checkbox = gr.Checkbox(label="Hide selected object instead of showing only it")
            conf_slider = gr.Slider(0.0, 1.0, value=min_confidence, step=0.05,
                                     label="Minimum confidence")
        plot_output = gr.Plot()
        run_btn = gr.Button("Show object")

        run_btn.click(fn=render, inputs=[obj_dropdown, hide_checkbox, conf_slider], outputs=plot_output)
        demo.load(fn=render, inputs=[obj_dropdown, hide_checkbox, conf_slider], outputs=plot_output)

    demo.launch()



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Launch the semantic 3DGS object viewer web UI.")
    p.add_argument("--semantic_dir", required=True, help="e.g. outputs/semantic/my_scene")
    p.add_argument("--min_confidence", type=float, default=0.0)
    args = p.parse_args()
    launch(args.semantic_dir, args.min_confidence)
