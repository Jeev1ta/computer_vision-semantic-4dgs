"""
scripts/annotate_ground_truth.py
----------------------------------
Guide section 22: "You do not need a huge labeled dataset. Create a small
evaluation set for your own scene." This script gives you a minimal,
dependency-light way to hand-annotate ~10 held-out images with polygon
masks per object, saved in the exact layout eval_semantic.evaluate_scene()
expects:

    gt_masks/<image_name>/<object_id>.png   (binary 0/255 mask)

Usage:
    python scripts/annotate_ground_truth.py \
        --image data/my_scene/images/00031.jpg \
        --object_id 1 \
        --out_dir outputs/eval/gt_masks

Controls (OpenCV window):
    left click   - add a polygon vertex
    'z'          - undo last vertex
    'c'          - close/commit the current polygon (fills it into the mask)
    's'          - save the accumulated mask for this image+object and exit
    'q'          - quit without saving
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def annotate(image_path: str, object_id: int, out_dir: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    current_poly = []

    window = f"Annotate object {object_id} — click points, 'c' to close polygon, 's' to save, 'q' to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_poly.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = img.copy()
        overlay = display.copy()
        overlay[mask > 0] = (0, 255, 0)
        display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)

        for i, pt in enumerate(current_poly):
            cv2.circle(display, pt, 3, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display, current_poly[i - 1], pt, (0, 0, 255), 1)

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("z") and current_poly:
            current_poly.pop()
        elif key == ord("c") and len(current_poly) >= 3:
            pts = np.array([current_poly], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
            current_poly = []
        elif key == ord("s"):
            image_name = Path(image_path).name
            save_dir = Path(out_dir) / image_name
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{object_id}.png"
            cv2.imwrite(str(save_path), mask)
            print(f"Saved ground-truth mask -> {save_path}")
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hand-annotate a ground-truth mask for one object/image "
                                             "(guide section 22).")
    p.add_argument("--image", required=True)
    p.add_argument("--object_id", type=int, required=True)
    p.add_argument("--out_dir", default="outputs/eval/gt_masks")
    args = p.parse_args()
    annotate(args.image, args.object_id, args.out_dir)
