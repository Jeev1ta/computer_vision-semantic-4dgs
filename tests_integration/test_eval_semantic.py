import numpy as np

from integration.colmap_io import Camera
from integration.gaussian_io import GaussianCloud
from integration.eval_semantic import render_predicted_mask, iou_precision_recall


def _camera():
    return Camera(
        camera_id=1, image_id=1, name="cam0.jpg",
        width=100, height=100,
        fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        R=np.eye(3), t=np.zeros(3),
    )


def test_iou_precision_recall_perfect_overlap():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    scores = iou_precision_recall(mask, mask)
    assert scores["iou"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0


def test_iou_precision_recall_no_overlap():
    a = np.zeros((10, 10), dtype=bool)
    a[0:2, 0:2] = True
    b = np.zeros((10, 10), dtype=bool)
    b[8:10, 8:10] = True
    scores = iou_precision_recall(a, b)
    assert scores["iou"] == 0.0
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0


def test_render_predicted_mask_labels_only_selected_object():
    cloud = GaussianCloud(
        xyz=np.array([[0.0, 0.0, 2.0], [0.4, 0.0, 2.0]]),
        opacity=np.array([0.9, 0.9]),
        scale=None, rotation=None, n=2,
    )
    labels = np.array([0, 1])
    cam = _camera()

    pred0 = render_predicted_mask(cloud, labels, object_id=0, camera=cam, dilate_px=2)
    pred1 = render_predicted_mask(cloud, labels, object_id=1, camera=cam, dilate_px=2)

    assert pred0.any()
    assert pred1.any()
    # the two objects project to different image regions
    assert not np.array_equal(pred0, pred1)


def test_render_predicted_mask_empty_for_unused_label():
    cloud = GaussianCloud(
        xyz=np.array([[0.0, 0.0, 2.0]]),
        opacity=np.array([0.9]),
        scale=None, rotation=None, n=1,
    )
    labels = np.array([0])
    cam = _camera()
    pred = render_predicted_mask(cloud, labels, object_id=99, camera=cam)
    assert not pred.any()
