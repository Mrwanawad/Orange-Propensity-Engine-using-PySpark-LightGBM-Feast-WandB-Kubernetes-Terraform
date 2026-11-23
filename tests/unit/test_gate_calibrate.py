import numpy as np

from ope.evaluation.metrics import ece
from ope.models import gate
from ope.models.calibrate import Calibrator, choose


def test_gate_passes_good_model():
    r = gate.check([0.75, 0.76, 0.74, 0.75, 0.76], baseline_auc=0.70, calibration_ece=0.01)
    assert r["passed"]


def test_gate_fails_each_check():
    assert not gate.check([0.705] * 5, 0.70, 0.01)["checks"]["beats_baseline"]
    assert not gate.check([0.75] * 5, 0.70, 0.05)["checks"]["calibrated"]
    assert not gate.check([0.80, 0.80, 0.80, 0.80, 0.70], 0.60, 0.01)["checks"]["stable_folds"]
    champ = {"cv_auc": 0.80, "cv_se": 0.002}
    assert not gate.check([0.75] * 5, 0.70, 0.01, champ)["checks"]["not_worse_than_champion"]


def test_calibration_fixes_overconfident_scores():
    rng = np.random.default_rng(0)
    true_p = rng.beta(1, 12, 20_000)
    y = (rng.uniform(size=true_p.size) < true_p).astype(int)
    raw = np.clip(true_p * 3, 0, 1)  # overconfident by a factor of 3
    cal, report = choose(raw, y)
    assert report[report["chosen"]]["ece"] < report["raw"]["ece"]
    assert ece(y, cal(raw)) < 0.02


def test_calibrators_are_monotone():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=p.size) < p**2).astype(int)
    grid = np.linspace(0.01, 0.99, 50)
    for m in ("isotonic", "platt"):
        out = Calibrator(m).fit(p, y)(grid)
        assert np.all(np.diff(out) >= -1e-12)
