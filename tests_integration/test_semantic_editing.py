import numpy as np
import pytest
from PIL import Image

from eval_semantics import compute_iou, evaluate_folder
from scene.gaussian_editor import GaussianEditor


class _NumpySemanticModel:
    def __init__(self, semantic_features):
        self._xyz = np.zeros((len(semantic_features), 3), dtype=np.float32)
        self._opacity = np.zeros((len(semantic_features), 1), dtype=np.float32)
        self._features_dc = np.zeros((len(semantic_features), 1, 3), dtype=np.float32)
        self._scaling = np.zeros((len(semantic_features), 3), dtype=np.float32)
        self._rotation = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(semantic_features), 1))
        self._semantic_features = np.asarray(semantic_features, dtype=np.float32)

    @property
    def has_semantics(self):
        return self._semantic_features.size > 0

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_semantic_features(self):
        return self._semantic_features


def test_select_by_known_class_uses_logits():
    model = _NumpySemanticModel([[0, 4, 0, 0, 0], [0, 0, 0, 4, 0]])
    mask = GaussianEditor(model).select_by_text("safety vest")
    np.testing.assert_array_equal(mask, [True, False])


def test_unknown_text_query_fails_instead_of_selecting_arbitrary_class():
    model = _NumpySemanticModel([[0, 4, 0, 0, 0]])
    with pytest.raises(ValueError, match="does not match a known class"):
        GaussianEditor(model).select_by_text("a bicycle")


def test_semantic_selection_rejects_models_without_semantics():
    model = _NumpySemanticModel(np.empty((1, 0), dtype=np.float32))
    with pytest.raises(RuntimeError, match="no trained semantic features"):
        GaussianEditor(model).select_by_class_id(1)


def test_semantic_selection_rejects_feature_class_count_mismatch():
    model = _NumpySemanticModel([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="one logit per class"):
        GaussianEditor(model).select_by_class_id(1)


def test_iou_skips_classes_absent_from_both_prediction_and_ground_truth():
    labels = np.zeros((2, 2), dtype=np.uint8)
    miou, per_class = compute_iou(labels, labels, num_classes=3)
    assert miou == 0.0
    assert per_class == {}


def test_evaluation_pairs_matching_filenames_and_reports_measured_metrics(tmp_path):
    model_dir = tmp_path / "model"
    gt_dir = tmp_path / "gt" / "labels"
    (model_dir / "renders_semantics").mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    labels = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    Image.fromarray(labels).save(model_dir / "renders_semantics" / "frame_01.png")
    Image.fromarray(labels).save(gt_dir / "frame_01.png")
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(gt_dir / "unmatched.png")

    report = evaluate_folder(str(model_dir), str(tmp_path / "gt"))
    assert report["paired_frames"] == ["frame_01.png"]
    assert report["metrics"]["semantic_miou"] == 100.0
    assert "fps" not in report["metrics"]


def test_evaluation_does_not_return_placeholder_report_when_inputs_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Semantic render directory not found"):
        evaluate_folder(str(tmp_path / "missing_model"), str(tmp_path / "missing_gt"))