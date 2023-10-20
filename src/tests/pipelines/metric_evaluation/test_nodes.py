import pytest

from klass.pipelines.metric_evaluation.nodes import (
    f1_score,
    macro,
    micro,
    precision_score,
    recall_score,
)


@pytest.fixture
def tp():
    return 2100


@pytest.fixture
def fp():
    return 3000


@pytest.fixture
def tn():
    return 60000


@pytest.fixture
def fn():
    return 3500


def test_precision_score(tn, fp, fn, tp):
    assert precision_score(tn, fp, fn, tp) == float(tp / (tp + fp))


def test_recall_score(tn, fp, fn, tp):
    assert recall_score(tn, fp, fn, tp) == float(tp / (tp + fn))


def test_f1_score(tn, fp, fn, tp):
    prec = float(tp / (tp + fp))
    recall = float(tp / (tp + fn))
    assert f1_score(tn, fp, fn, tp) == (2 * prec * recall) / (prec + recall)


def test_macro(tn, fp, fn, tp):
    prec = float(tp / (tp + fp))
    rec = float(tp / (tp + fn))
    f1 = (2 * prec * rec) / (prec + rec)

    prec_2 = float(tn / (tn + fn))
    rec_2 = float(tn / (tn + fp))
    f1_2 = (2 * prec_2 * rec_2) / (prec_2 + rec_2)

    assert macro(tn, fp, fn, tp) == (
        round((prec + prec_2) / 2, 4),
        round((rec + rec_2) / 2, 4),
        round((f1 + f1_2) / 2, 4),
    )


def test_micro(tn, fp, fn, tp):
    prec = float((tp + tn) / (tp + tn + fp + fn))
    rec = float((tp + tn) / (tp + tn + fn + fp))
    f1 = 2 * (prec * rec) / (prec + rec)

    assert micro(tn, fp, fn, tp) == (round(prec, 4), round(rec, 4), round(f1, 4))
