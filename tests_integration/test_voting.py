import numpy as np

from integration.voting import (
    majority_vote,
    majority_vote_batch,
    confidence_weighted_vote_batch,
    single_view_labels,
    compute_projection_weight,
    UNKNOWN_LABEL,
)


def test_majority_vote_scalar_matches_guide_example():
    # guide section 17 worked example: chair=4, background=1 -> chair, 0.80
    votes = [0, 0, 0, 1, 0]  # 0 = chair, 1 = background
    label, conf = majority_vote(votes)
    assert label == 0
    assert conf == 0.8


def test_majority_vote_scalar_all_unknown():
    label, conf = majority_vote([UNKNOWN_LABEL, UNKNOWN_LABEL])
    assert label == UNKNOWN_LABEL
    assert conf == 0.0


def test_majority_vote_batch_matches_scalar_reference():
    votes = np.array([
        [0, 0, 0, 1, 0],
        [UNKNOWN_LABEL, UNKNOWN_LABEL, 1, 1, UNKNOWN_LABEL],
        [UNKNOWN_LABEL, UNKNOWN_LABEL, UNKNOWN_LABEL, UNKNOWN_LABEL, UNKNOWN_LABEL],
    ])
    labels, conf, n_sup = majority_vote_batch(votes, min_views=1)

    for i, row in enumerate(votes):
        exp_label, exp_conf = majority_vote(row.tolist())
        assert labels[i] == exp_label
        assert np.isclose(conf[i], exp_conf)


def test_majority_vote_batch_respects_min_views():
    # Only 1 supporting view but min_views=2 -> should stay unknown.
    votes = np.array([[0, UNKNOWN_LABEL, UNKNOWN_LABEL]])
    labels, conf, n_sup = majority_vote_batch(votes, min_views=2)
    assert labels[0] == UNKNOWN_LABEL
    assert n_sup[0] == 1


def test_single_view_labels():
    votes = np.array([
        [0, 1, 1],
        [UNKNOWN_LABEL, 1, 1],
    ])
    labels, conf, n_sup, _ = single_view_labels(votes, view_index=0)
    assert labels.tolist() == [0, UNKNOWN_LABEL]
    assert conf.tolist() == [1.0, 0.0]


def test_single_view_labels_auto_picks_most_informative_view():
    votes = np.array([
        [UNKNOWN_LABEL, 0, 0],
        [UNKNOWN_LABEL, 0, 1],
    ])
    labels, conf, n_sup, chosen = single_view_labels(votes)  # no explicit index
    assert chosen == 1  # column 1 has the most votes -> most informative view
    assert labels.tolist() == [0, 0]


def test_confidence_weighted_vote_overrides_majority():
    # 2 votes for label 0 with low weight, 1 vote for label 1 with high weight
    # -> weighted result should prefer label 1 even though it has fewer votes.
    votes = np.array([[0, 0, 1]])
    weights = np.array([[0.1, 0.1, 10.0]])
    labels, conf, n_sup = confidence_weighted_vote_batch(votes, weights, min_views=1)
    assert labels[0] == 1
    assert conf[0] > 0.9


def test_confidence_weighted_vote_matches_majority_with_equal_weights():
    votes = np.array([[0, 0, 0, 1, 0]])
    weights = np.ones_like(votes, dtype=np.float64)
    labels_w, conf_w, _ = confidence_weighted_vote_batch(votes, weights, min_views=1)
    labels_m, conf_m, _ = majority_vote_batch(votes, min_views=1)
    assert labels_w[0] == labels_m[0]
    assert np.isclose(conf_w[0], conf_m[0])


def test_compute_projection_weight_closer_is_higher():
    dist = np.array([1.0, 10.0])
    w = compute_projection_weight(scale=None, distance_to_camera=dist)
    assert w[0] > w[1]  # closer Gaussian should get a higher weight


def test_compute_projection_weight_with_scale_and_confidence():
    scale = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    dist = np.array([2.0, 2.0])
    conf = np.array([0.9, 0.1])
    w = compute_projection_weight(scale=scale, distance_to_camera=dist, mask_confidence=conf)
    assert w[0] > w[1]
