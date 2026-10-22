"""
dashboard/server.py
-------------------
The ONE dashboard server for the project. Serves the frontend and a JSON API
that reads/writes ONLY real pipeline artifacts: no fallback numbers, no demo
objects, no synthetic segmentation. Missing pipeline stages are reported as
missing, never invented.

Persistence goes through the project's existing writers only
(integration.visualize / integration.mask_io), so every artifact on disk is
produced by the same code paths the CLI pipeline uses.

Run:  python dashboard/server.py [port]     (default port 8001)

API
---
GET  /api/status             backends + pipeline inputs present
GET  /api/scene              semantic point cloud (xyz/labels/confidence) + objects
GET  /api/metrics            run_metrics + ablation comparison + eval results
GET  /uploads/<name>         previously uploaded images (from data/uploads/)
POST /api/segment            {image: dataURI, filename, backend, max_objects}
                             -> real segmentation + honest 2.5D lift (in-memory
                                image, results persisted as PLY + metadata)
POST /api/pipeline/run       {strategy, min_views} -> real 2D->3D association
POST /api/pipeline/ablation  -> real ablation sweep
POST /api/labels/rename      {id, label} -> rename an object in the metadata
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integration import mask_io, visualize  # noqa: E402
from integration.run_semantic_pipeline import load_config, run_once, run_ablation  # noqa: E402
from integration.single_image import segment_and_lift  # noqa: E402
from integration.segmentation import available_backends  # noqa: E402

PORT = 8001
if len(sys.argv) > 1:
    try:
        PORT = int(sys.argv[1])
    except ValueError:
        pass

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# --- configured locations: plain folder name only, normalized, no traversal ---
_CFG = load_config(str(REPO_ROOT / "configs" / "config.yaml"))
_SCENE = str(_CFG.get("scene_name", "scene"))
if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", _SCENE):
    raise RuntimeError(f"config scene_name must be a plain folder name, got {_SCENE!r}")

SEMANTIC_DIR = os.path.normpath(os.path.join(REPO_ROOT, "outputs", "semantic", _SCENE))
REPO_ROOT_S = str(REPO_ROOT)
if ".." in SEMANTIC_DIR.split(os.sep) or not SEMANTIC_DIR.startswith(REPO_ROOT_S):
    raise RuntimeError(f"semantic dir escapes the project: {SEMANTIC_DIR}")

_CLOUD_PLY = "gaussian_cloud.ply"      # colored cloud (pipeline or 2.5D mode)
_METADATA_JSON = "object_metadata.json"


def _validated_subpath(base: str, name: str) -> str:
    """Join + normalize + validate in one visible step: rejects any '..'
    component and requires the result to stay inside `base`."""
    candidate = os.path.normpath(os.path.join(base, name))
    if ".." in candidate.split(os.sep) or not candidate.startswith(base + os.sep):
        raise ValueError(f"Path traversal forbidden: {name!r}")
    return candidate


def _load_json(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _palette_color_of(label: int, unknown_label: int = -1):
    if label == unknown_label:
        return (120, 120, 120)
    return tuple(int(c) for c in visualize._PALETTE[label % len(visualize._PALETTE)])


def _labels_from_colors(rgb: np.ndarray, unknown_label: int = -1) -> np.ndarray:
    """Inverse of visualize.label_to_color: nearest palette entry per point."""
    palette = np.array(
        [_palette_color_of(lab, unknown_label) for lab in range(-1, 64)],
        dtype=np.int64,
    )  # row i corresponds to label i-1
    d = ((rgb.astype(np.int64)[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    return (d.argmin(axis=1) - 1).astype(np.int64)


def scene_payload():
    """Current semantic point cloud + object metadata (canonical schema).

    Prefers the pipeline's npy outputs; falls back to the colored PLY cloud
    (single-image mode) with per-object confidence from the metadata.
    """
    meta = _load_json(os.path.join(SEMANTIC_DIR, _METADATA_JSON)) or {}
    label_names = {int(o["id"]): str(o["label"]) for o in meta.get("objects", [])}
    unknown_label = int(meta.get("unknown_label", -1))

    xyz_path = os.path.join(SEMANTIC_DIR, "gaussian_xyz.npy")
    if os.path.exists(xyz_path):
        xyz = np.load(xyz_path)
        labels = np.load(os.path.join(SEMANTIC_DIR, "gaussian_labels.npy"))
        conf_path = os.path.join(SEMANTIC_DIR, "gaussian_confidence.npy")
        confidence = (np.load(conf_path) if os.path.exists(conf_path)
                      else np.ones(len(xyz)))
    else:
        ply_path = os.path.join(SEMANTIC_DIR, _CLOUD_PLY)
        if not os.path.exists(ply_path):
            return None
        from integration.ply_compat import PlyData
        ply = PlyData.read(ply_path)
        v = ply.elements[0]
        xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
        rgb = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], axis=1)
        labels = _labels_from_colors(rgb, unknown_label)
        obj_conf = {o["id"]: float(o.get("score", 1.0)) for o in meta.get("segmentation", [])}
        confidence = np.array([obj_conf.get(int(l), 1.0) for l in labels])

    return {
        "n_points": int(len(xyz)),
        "mode": meta.get("mode", "multi_view_pipeline"),
        "depth_source": meta.get("depth_source"),
        "label_names": {str(k): v for k, v in label_names.items()},
        "xyz": xyz.astype(float).round(4).tolist(),
        "labels": labels.astype(int).tolist(),
        "confidence": confidence.astype(float).round(4).tolist(),
    }


def metrics_payload():
    run_metrics = _load_json(os.path.join(SEMANTIC_DIR, "run_metrics.json"))
    ablation = _load_json(os.path.join(SEMANTIC_DIR, "ablation", "comparison.json"))
    evaluation = _load_json(os.path.join(SEMANTIC_DIR, "eval_results.json"))
    meta = _load_json(os.path.join(SEMANTIC_DIR, _METADATA_JSON)) or {}
    return {
        "run_metrics": run_metrics,
        "ablation": ablation,
        "evaluation": evaluation,
        "metadata": meta,
    }


def status_payload():
    model_pc = REPO_ROOT / _CFG["paths"]["model_dir"] / "point_cloud"
    return {
        "backends": available_backends(),
        "inputs": {
            "colmap_data": os.path.isdir(str(REPO_ROOT / _CFG["paths"]["dataset_dir"])),
            "trained_ply": model_pc.exists() and any(model_pc.glob("iteration_*")),
            "masks": os.path.isdir(str(REPO_ROOT / _CFG["paths"]["mask_dir"] / "metadata")),
            "semantic_outputs": scene_payload() is not None,
        },
        "scene_name": _SCENE,
        "mode": (_load_json(os.path.join(SEMANTIC_DIR, _METADATA_JSON)) or {}).get("mode"),
    }


def persist_single_image_result(result: dict, source_filename: str):
    """Persist the 2.5D result using the project's existing writers:
    colored PLY cloud (visualize.save_colored_ply) + canonical metadata JSON
    (mask_io.write_semantic_object_metadata) + top-down preview."""
    os.makedirs(SEMANTIC_DIR, exist_ok=True)
    # a 2.5D run supersedes any previous multi-view npy outputs: remove them
    # so readers never mix results from two different modes
    for stale in ("gaussian_xyz.npy", "gaussian_labels.npy", "gaussian_confidence.npy",
                  "run_metrics.json", "eval_results.json"):
        stale_path = os.path.join(SEMANTIC_DIR, stale)
        if os.path.isfile(stale_path):
            os.remove(stale_path)

    xyz, labels = result["xyz"], result["labels"]
    visualize.save_colored_ply(xyz, labels, os.path.join(SEMANTIC_DIR, _CLOUD_PLY))
    visualize.plot_topdown(xyz, labels, os.path.join(SEMANTIC_DIR, "topdown_preview.png"),
                           result["label_names"])

    label_names = {int(k): str(v) for k, v in result["label_names"].items()}
    source_views = {int(o["id"]): [source_filename] for o in result["objects"]}
    mask_io.write_semantic_object_metadata(
        os.path.join(SEMANTIC_DIR, _METADATA_JSON),
        label_names,
        source_views,
        unknown_label=-1,
        extra={
            "mode": result["mode"],
            "depth_source": result["depth_source"],
            "segmentation_backend": result["backend"],
            "image_size": result["image_size"],
            "hfov_deg": result["hfov_deg"],
            "source_image": source_filename,
            "segmentation": result["objects"],
        },
    )


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    # ------------------------------------------------------------ helpers
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 64 * 1024 * 1024:
            raise ValueError(f"invalid body size {length}")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def log_message(self, fmt, *args):  # quieter: only API hits
        if args and "/api/" in str(args[0]):
            super().log_message(fmt, *args)

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path

        if route == "/api/status":
            return self._json(status_payload())

        if route == "/api/scene":
            scene = scene_payload()
            if scene is None:
                return self._json(
                    {"error": "no semantic outputs yet. Upload an image "
                              "(Segment panel) or run the pipeline."},
                    status=404)
            return self._json(scene)

        if route == "/api/metrics":
            return self._json(metrics_payload())

        if route.startswith("/uploads/"):
            raw_name = urllib.parse.unquote(route[len("/uploads/"):])
            name = _validated_subpath(SEMANTIC_DIR, os.path.basename(raw_name))
            if not os.path.isfile(name):
                return self._json({"error": "not found"}, status=404)
            ctype = "image/jpeg" if name.endswith((".jpg", ".jpeg")) else "image/png"
            with open(name, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        return super().do_GET()  # static files from dashboard/

    # --------------------------------------------------------------- POST
    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        try:
            if route == "/api/pipeline/run":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                metrics = run_once(
                    _CFG,
                    strategy_override=body.get("strategy"),
                    min_views_override=int(body["min_views"]) if body.get("min_views") else None,
                )
                return self._json({"success": True, "metrics": metrics})
            if route == "/api/pipeline/ablation":
                return self._json({"success": True, "ablation": run_ablation(_CFG)})
            if route == "/api/labels/rename":
                return self._post_rename()
            if route == "/api/segment":
                return self._post_segment()
            return self._json({"error": f"unknown endpoint {route}"}, status=404)
        except Exception as err:  # real errors reach the UI, never fake data
            return self._json({"error": f"{type(err).__name__}: {err}"}, status=500)

    def _post_segment(self):
        body = self._read_json_body()
        data_uri = body.get("image", "")
        if "base64," not in data_uri:
            raise ValueError("expected a base64 data URI in 'image'")
        header, b64 = data_uri.split(",", 1)
        img_bytes = base64.b64decode(b64)

        # process fully in memory (browser previews its own local file)
        import io
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
        filename = os.path.basename(str(body.get("filename", "upload.jpg")))[:80] or "upload.jpg"
        backend = body.get("backend", "auto")
        max_objects = int(body.get("max_objects", 8))
        result = segment_and_lift(np.array(img), backend=backend, max_objects=max_objects)
        persist_single_image_result(result, filename)

        return self._json({
            "success": True,
            "backend": result["backend"],
            "mode": result["mode"],
            "depth_source": result["depth_source"],
            "n_objects": len(result["objects"]),
            "objects": result["objects"],
            "scene": scene_payload(),
        })

    def _post_rename(self):
        body = self._read_json_body()
        obj_id, new_label = int(body["id"]), str(body["label"]).strip()[:60]
        if not new_label:
            raise ValueError("label must be non-empty")
        meta_path = os.path.join(SEMANTIC_DIR, _METADATA_JSON)
        meta = _load_json(meta_path)
        if not meta or "objects" not in meta:
            raise ValueError("no semantic metadata to rename (run a pipeline first)")
        found = False
        for obj in meta["objects"]:
            if int(obj["id"]) == obj_id:
                obj["label"] = new_label
                found = True
        if not found:
            raise ValueError(f"object id {obj_id} not found")
        # rewrite through the project's canonical metadata writer, preserving
        # all provenance fields (mode/depth_source/segmentation/...)
        label_names = {int(o["id"]): str(o["label"]) for o in meta["objects"]}
        source_views = {int(o["id"]): list(o.get("source_views", [])) for o in meta["objects"]}
        extra = {k: v for k, v in meta.items()
                 if k not in ("objects", "unknown_label")}
        mask_io.write_semantic_object_metadata(
            meta_path, label_names, source_views,
            unknown_label=int(meta.get("unknown_label", -1)), extra=extra,
        )
        return self._json({"success": True, "scene": scene_payload()})


def run():
    print(f"Semantic 3DGS dashboard  ->  http://localhost:{PORT}")
    st = status_payload()
    print(f"  backends: {st['backends']}")
    print(f"  pipeline inputs present: {st['inputs']}")
    server = ThreadingHTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard stopped.")


if __name__ == "__main__":
    run()
