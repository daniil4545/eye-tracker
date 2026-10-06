"""Tests for the moving-dot stage that follows the calibration dots."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eye_tracker.gaze.calibration import (
    POSE_ID_STRIDE,
    PURSUIT_LAG_S,
    PURSUIT_POSE,
    CalibrationCollector,
    CalibrationReport,
    CalibrationSample,
    balance_pursuit_weights,
    evaluate,
    make_plan,
    make_pursuit,
    place_groups,
    pose_of,
)
from eye_tracker.gaze.store import CalibrationData, load_calibration, save_calibration
from eye_tracker.types import Observation, layout_signature
from gaze_synth import TWO, calibration_samples, synth_features

GAZE_FEATURES = (0, 1, 6, 7)


def obs() -> Observation:
    return Observation(timestamp=0.0, face_count=1, features=np.arange(8.0), quality=1.0)


def collector(pursuit_s: float = 10.0) -> CalibrationCollector:
    return CalibrationCollector(
        make_plan(TWO)[:1],
        settle_s=0.8,
        collect_s=1.0,
        min_samples=1,
        pursuit=make_pursuit(TWO),
        pursuit_s=pursuit_s,
    )


def finish_the_dots(c: CalibrationCollector) -> float:
    """Runs the single dot; returns the time the moving dot appears."""
    c.start(0.0)
    c.update(0.0)
    c.update(0.8)
    assert c.add(obs())
    assert c.update(1.8) == ["target"]
    return 1.8


def test_the_moving_dot_follows_the_dots_on_every_monitor() -> None:
    c = collector()
    t = finish_the_dots(c)
    tracks = make_pursuit(TWO)
    assert c.in_pursuit
    assert (c.phase, c.pursuit_track, c.pursuit_tracks) == ("settle", 0, 2)
    start = c.current
    assert start is not None
    assert (start.monitor_index, start.nx, start.ny) == (tracks[0].monitor.index, 0.5, 0.5)
    assert not c.add(obs()), "nothing is recorded while the dot waits"

    c.update(t + 0.8)
    assert c.phase == "pursuit"
    assert c.update(t + 0.8 + 10.0) == ["target"]
    assert c.pursuit_track == 1
    current = c.current
    assert current is not None
    assert current.monitor_index == tracks[1].monitor.index
    c.update(t + 0.8 + 10.0 + 0.8)
    assert c.update(t + 2 * (0.8 + 10.0)) == ["finished"]
    assert (c.phase, c.current, c.in_pursuit, c.progress) == ("done", None, False, 1.0)


def test_a_frame_is_labelled_where_the_dot_was_a_moment_ago() -> None:
    c = collector()
    t = finish_the_dots(c)
    c.update(t + 0.8)
    assert not c.add(obs()), "no labels while the eyes catch the dot"
    c.update(t + 0.8 + 5.0)
    assert c.add(obs())
    sample = c.samples[-1]
    seen = make_pursuit(TWO)[0].target((5.0 - PURSUIT_LAG_S) / 10.0, 0)
    assert (sample.x, sample.y) == pytest.approx((seen.x, seen.y))
    assert sample.monitor_index == seen.monitor_index
    assert sample.point_id == PURSUIT_POSE * POSE_ID_STRIDE + 2  # third 2 s segment
    assert pose_of(sample.point_id) == PURSUIT_POSE


def test_progress_counts_the_moving_dot_by_time() -> None:
    c = collector()
    t = finish_the_dots(c)
    total = 1.8 + 2 * 10.8
    assert c.progress == pytest.approx(1.8 / total)
    c.update(t + 0.8)
    c.update(t + 0.8 + 5.0)
    assert c.progress == pytest.approx((1.8 + 5.8) / total)


def test_without_pursuit_time_there_is_no_moving_dot() -> None:
    c = collector(pursuit_s=0.0)
    c.start(0.0)
    c.update(0.0)
    c.update(0.8)
    c.add(obs())
    assert c.update(1.8) == ["finished"]
    assert c.pursuit_tracks == 0


def test_a_moving_dot_segment_is_one_cross_validation_group() -> None:
    feats = np.zeros(8)
    segment = PURSUIT_POSE * POSE_ID_STRIDE + 3
    samples = [
        CalibrationSample(feats, 100.0, 100.0, 0, segment),
        CalibrationSample(feats, 180.0, 140.0, 0, segment),
        CalibrationSample(feats, 260.0, 180.0, 0, segment + 1),
        CalibrationSample(feats, 100.0, 100.0, 0, 4),  # an ordinary dot at the same place
    ]
    groups = place_groups(samples)
    assert groups[0] == groups[1]
    assert len(set(groups.tolist())) == 3


def test_the_moving_dot_does_not_count_as_dots() -> None:
    rng = np.random.default_rng(3)
    dots = calibration_samples(TWO, rng, 1.0, per_point=10)
    moving = [
        CalibrationSample(s.features, s.x, s.y, s.monitor_index, PURSUIT_POSE * POSE_ID_STRIDE + i)
        for i, s in enumerate(calibration_samples(TWO, rng, 1.0, per_point=1))
    ]
    _, report = evaluate([*dots, *moving], TWO)
    _, alone = evaluate(dots, TWO)
    assert report.n_points == alone.n_points
    assert PURSUIT_POSE in report.per_pose_error_px


def pose_samples(
    pose: int,
    head_offset: tuple[float, float, float],
    rng: np.random.Generator,
    per_point: int = 20,
) -> list[CalibrationSample]:
    """Samples of ``pose`` (ids ``pose * POSE_ID_STRIDE + k``, 5 dots per monitor)."""
    out: list[CalibrationSample] = []
    for t in make_plan(TWO, 5):
        target = np.tile((t.x, t.y), (per_point, 1))
        feats = synth_features(target, rng, 1.0, head_offset=head_offset)
        out += [
            CalibrationSample(f, t.x, t.y, t.monitor_index, pose * POSE_ID_STRIDE + t.point_id)
            for f in feats
        ]
    return out


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


def test_the_size_gate_error_comes_from_the_usual_pose() -> None:
    rng = np.random.default_rng(6)
    base = calibration_samples(TWO, rng, 1.0, per_point=15)
    # a head position the model cannot follow must not widen the gate; the tiny
    # weight keeps the fit itself as it was without it
    noisy = []
    for t in make_plan(TWO, 5):
        feats = synth_features(np.tile((t.x, t.y), (15, 1)), rng, 8.0, head_offset=(6.0, 0, 0))
        noisy += [
            CalibrationSample(
                f, t.x, t.y, t.monitor_index, POSE_ID_STRIDE + t.point_id, weight=1e-4
            )
            for f in feats
        ]
    _, alone = evaluate(base, TWO, nonlinear=GAZE_FEATURES)
    _, both = evaluate([*base, *noisy], TWO, nonlinear=GAZE_FEATURES)
    for monitor, (x, y) in both.per_monitor_error_px.items():
        ax, ay = alone.per_monitor_error_px[monitor]
        assert x < 1.5 * ax
        assert y < 1.5 * ay


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


def test_moving_dot_samples_weigh_as_much_as_the_ordinary_dots() -> None:
    features = np.zeros(2)
    samples = [
        CalibrationSample(features, 0.0, 0.0, 0, 1, 1.0),
        CalibrationSample(features, 0.0, 0.0, 0, 2, 1.0),
        *[CalibrationSample(features, 0.0, 0.0, 0, POSE_ID_STRIDE + i) for i in range(8)],
        CalibrationSample(features, 0.0, 0.0, 0, -1, 0.5),  # learned from the mouse
    ]
    balanced = balance_pursuit_weights(samples)
    assert sum(s.weight for s in balanced if pose_of(s.point_id) > 0) == pytest.approx(2.0)
    assert [s.weight for s in balanced if pose_of(s.point_id) <= 0] == [1.0, 1.0, 0.5]
    assert balance_pursuit_weights(samples[:2]) == samples[:2]  # no moving dot: unchanged


def test_old_calibrations_with_head_pose_samples_still_work(tmp_path: Path) -> None:
    # Saved before the head-pose series was removed: dots with ids 1000-6999.
    rng = np.random.default_rng(9)
    samples = calibration_samples(TWO, rng, 1.0, per_point=10)
    samples += [s for pose in range(1, 7) for s in pose_samples(pose, (4.0, 0, 0), rng, 3)]
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
    assert max(s.point_id for s in loaded.samples) >= 6 * POSE_ID_STRIDE
    again, refit = evaluate(loaded.samples, TWO, nonlinear=GAZE_FEATURES)  # ordinary samples
    assert refit.n_points == report.n_points
    assert math.isfinite(refit.median_error_px)
    assert np.all(np.isfinite(again.predict(np.array([s.features for s in loaded.samples[:5]]))))
    assert len(balance_pursuit_weights(loaded.samples)) == len(loaded.samples)
