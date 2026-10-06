"""Tests for the head-pose calibration: data, evaluation and the pose series."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eye_tracker.gaze.calibration import (
    POSE_ID_STRIDE,
    CalibrationReport,
    CalibrationSample,
    evaluate,
    make_plan,
    place_groups,
    pose_of,
)
from eye_tracker.gaze.store import CalibrationData, load_calibration, save_calibration
from eye_tracker.types import layout_signature
from gaze_synth import TWO, calibration_samples, random_points, synth_features

GAZE_FEATURES = (0, 1, 6, 7)


def pose_samples(
    pose: int,
    head_offset: tuple[float, float, float],
    rng: np.random.Generator,
    per_point: int = 20,
) -> list[CalibrationSample]:
    """Samples of one pose as the pose series records them (5 dots per monitor)."""
    out: list[CalibrationSample] = []
    for t in make_plan(TWO, 5, first_id=pose * POSE_ID_STRIDE):
        target = np.tile((t.x, t.y), (per_point, 1))
        feats = synth_features(target, rng, 1.0, head_offset=head_offset)
        out += [CalibrationSample(f, t.x, t.y, t.monitor_index, t.point_id) for f in feats]
    return out


def still(samples: list[CalibrationSample], keep: float = 0.15) -> list[CalibrationSample]:
    """The samples of a user who held the head still while calibrating."""
    head = [2, 3, 4, 5]  # roll, tx, ty, tz
    feats = np.array([s.features for s in samples])
    centre = np.median(feats[:, head], axis=0)
    for s in samples:
        s.features[head] = centre + keep * (s.features[head] - centre)
    return samples


def test_plan_first_id_offsets_the_point_ids() -> None:
    plan = make_plan(TWO, 5, first_id=2 * POSE_ID_STRIDE)
    assert [t.point_id for t in plan] == [2000 + i for i in range(10)]
    assert [t.point_id for t in make_plan(TWO, 5)] == list(range(10))


def test_the_same_place_gets_one_group_in_every_plan() -> None:
    rng = np.random.default_rng(1)
    base = calibration_samples(TWO, rng, per_point=2)  # 9 dots per monitor
    poses = pose_samples(1, (5.0, 0, 0), rng, 2) + pose_samples(2, (-5.0, 0, 0), rng, 2)
    groups = place_groups([*base, *poses])
    places = {(s.monitor_index, round(s.x), round(s.y)) for s in [*base, *poses]}
    assert np.unique(groups).shape[0] == len(places) == 18  # not 18 + 10 + 10
    # a mirrored plan variant differs in the last float digits only
    a, b = base[0], CalibrationSample(base[0].features, base[0].x + 1e-9, base[0].y, 0, 5)
    assert place_groups([a, b]).tolist() == [0, 0]


def test_evaluate_reports_the_error_per_pose() -> None:
    rng = np.random.default_rng(2)
    samples = calibration_samples(TWO, rng, 1.0, per_point=15)
    samples += pose_samples(1, (6.0, 0, 0), rng, 15) + pose_samples(2, (-6.0, 0, 0), rng, 15)
    _, report = evaluate(samples, TWO, nonlinear=GAZE_FEATURES)
    assert report.n_points == 18  # places, not (pose, dot) pairs
    assert set(report.per_pose_error_px) == {0, 1, 2}
    assert set(report.unseen_pose_error_px) == {0, 1, 2}
    assert all(math.isfinite(v) and v > 0 for v in report.per_pose_error_px.values())


def test_evaluate_without_poses_reports_pose_zero_only() -> None:
    samples = calibration_samples(TWO, np.random.default_rng(3), 1.0, per_point=15)
    _, report = evaluate(samples, TWO, nonlinear=GAZE_FEATURES)
    assert set(report.per_pose_error_px) == {0}
    assert report.per_pose_error_px[0] == report.median_error_px
    assert report.unseen_pose_error_px == {}


def test_report_pose_fields_round_trip_and_old_reports_load() -> None:
    samples = calibration_samples(TWO, np.random.default_rng(4), 1.0, per_point=15)
    samples += pose_samples(1, (6.0, 0, 0), np.random.default_rng(5), 15)
    _, report = evaluate(samples, TWO, nonlinear=GAZE_FEATURES)
    data = report.to_dict()
    assert set(data["per_pose_error_px"]) == {"0", "1"}
    assert CalibrationReport.from_dict(data).per_pose_error_px == report.per_pose_error_px
    for key in ("per_pose_error_px", "unseen_pose_error_px"):
        del data[key]
    old = CalibrationReport.from_dict(data)
    assert old.per_pose_error_px == {}
    assert old.unseen_pose_error_px == {}


def test_poses_keep_the_model_accurate_when_the_head_moves() -> None:
    rng = np.random.default_rng(6)
    base = still(calibration_samples(TWO, rng, 1.0, per_point=15))
    shifts = [(7.0, 0, 0), (-7.0, 0, 0), (0, 5.0, 0), (0, 0, 8.0), (0, 0, -8.0)]
    poses = [s for i, off in enumerate(shifts, 1) for s in pose_samples(i, off, rng, 15)]
    plain, _ = evaluate(base, TWO, nonlinear=GAZE_FEATURES)
    trained, _ = evaluate([*base, *poses], TWO, nonlinear=GAZE_FEATURES)
    points = random_points(TWO, rng, 300)
    truth_err = {}
    for name, model in (("plain", plain), ("trained", trained)):
        errors = []
        for off in shifts:
            X = synth_features(points, rng, 1.0, head_offset=off)
            errors.append(np.median(np.hypot(*(model.predict(X) - points).T)))
        truth_err[name] = float(np.mean(errors))
    assert truth_err["trained"] < truth_err["plain"]
    # the range the head pause judges by now covers the poses
    X = synth_features(points, rng, 1.0, head_offset=shifts[0])
    assert float(trained.extrapolation(X)[:, 3].mean()) < float(plain.extrapolation(X)[:, 3].mean())


def test_pose_samples_survive_the_calibration_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    samples = calibration_samples(TWO, rng, 1.0, per_point=5) + pose_samples(2, (5.0, 0, 0), rng, 5)
    model, report = evaluate(samples, TWO, nonlinear=GAZE_FEATURES)
    data = CalibrationData(
        backend="facemesh",
        feature_version="v",
        layout_signature=layout_signature(TWO),
        monitors=list(TWO),
        samples=samples,
        implicit_samples=[],
        model=model,
        report=report.to_dict(),
    )
    path = tmp_path / "calibration.json"
    save_calibration(path, data)
    loaded = load_calibration(path)
    assert loaded is not None
    assert sorted({pose_of(s.point_id) for s in loaded.samples}) == [0, 2]
    assert CalibrationReport.from_dict(loaded.report).per_pose_error_px == pytest.approx(
        report.per_pose_error_px, rel=1e-3
    )
