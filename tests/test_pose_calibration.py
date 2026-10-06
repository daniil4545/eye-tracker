"""Tests for the head-pose calibration: data, evaluation and the pose series."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from eye_tracker.gaze.calibration import (
    EVENT_FINISHED,
    EVENT_POSE,
    EVENT_POSE_MORE,
    EVENT_POSE_SKIPPED,
    PHASE_DONE,
    PHASE_WAIT,
    POSE_ID_STRIDE,
    POSES,
    CalibrationReport,
    CalibrationSample,
    PoseSeries,
    PoseSpec,
    evaluate,
    make_plan,
    place_groups,
    pose_of,
    pose_regression,
)
from eye_tracker.gaze.store import CalibrationData, load_calibration, save_calibration
from eye_tracker.types import Observation, layout_signature
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


# ------------------------------------------------------------------ PoseSeries
HEAD = {"roll": 2, "tx": 3, "ty": 4, "tz": 5}
FRAME = (640, 480)


def make_series(**kwargs: object) -> tuple[PoseSeries, np.ndarray]:
    base = still(calibration_samples(TWO, np.random.default_rng(9), 1.0, per_point=8))
    series = PoseSeries(TWO, base, HEAD, settle_s=0.2, collect_s=0.3, min_samples=3, **kwargs)  # type: ignore[arg-type]
    return series, np.median([s.features for s in base], axis=0)


Behaviour = Callable[[PoseSpec, bool], float | None]


def drive(
    series: PoseSeries, neutral: np.ndarray, behaviour: Behaviour, *, seconds: float = 900.0
) -> list[str]:
    """Run the series on a fake clock; ``behaviour(pose, waiting)`` is the shift of
    the pose's feature the user holds (``None``: no face)."""
    now, events = 0.0, []
    series.start(now)
    while series.phase != PHASE_DONE:
        events += series.update(now)
        spec = series.pose
        shift = behaviour(spec, series.phase == PHASE_WAIT) if spec is not None else None
        if shift is not None and spec is not None:
            features = neutral.copy()
            features[HEAD[spec.feature]] += shift
            series.add(
                Observation(
                    timestamp=now, face_count=1, features=features, quality=1.0, frame_size=FRAME
                )
            )
        now += 0.05
        assert now < seconds, "the series never finished"
    return events


def follows(spec: PoseSpec, waiting: bool) -> float:
    """A user who does what the prompt says (the second of a pair goes the other way)."""
    sign = -1.0 if spec.opposite_of else 1.0
    return sign * 2.0 * spec.min_shift


def test_every_pose_is_taken_when_the_user_follows_the_prompts() -> None:
    series, neutral = make_series()
    events = drive(series, neutral, follows)
    assert events.count(EVENT_POSE) == len(POSES)
    assert events[-1] == EVENT_FINISHED
    assert series.accepted == [1, 2, 3, 4, 5, 6]
    assert {pose_of(s.point_id) for s in series.samples} == {1, 2, 3, 4, 5, 6}
    assert series.frame_size == FRAME
    assert len(series.plan) == len(POSES) * 10


def test_a_pose_that_is_not_reached_is_asked_twice_then_skipped() -> None:
    series, neutral = make_series()
    events = drive(series, neutral, lambda s, w: 0.0 if s.name == "left" else follows(s, w))
    assert events.count(EVENT_POSE_MORE) == 1
    assert events.count(EVENT_POSE_SKIPPED) == 1
    assert series.accepted == [2, 3, 4, 5, 6]
    assert series.skipped_poses == ["left"]


def test_the_second_pose_of_a_pair_must_go_the_other_way() -> None:
    series, neutral = make_series()
    drive(series, neutral, lambda s, w: 4.0 if s.feature == "tx" else follows(s, w))
    assert series.accepted == [1, 3, 4, 5, 6]  # "right" went the same way as "left"
    assert series.skipped_poses == ["right"]


def test_a_pose_given_up_during_the_dots_is_retried_once() -> None:
    series, neutral = make_series()
    tries: list[bool] = []

    def behaviour(spec: PoseSpec, waiting: bool) -> float:
        if spec.name != "lower":
            return follows(spec, waiting)
        tries.append(waiting)
        return 4.0 if waiting else 0.0  # reached, then back to the neutral position

    events = drive(series, neutral, behaviour)
    assert events.count(EVENT_POSE) == len(POSES) + 1
    assert series.skipped_poses == ["lower"]
    assert 5 not in series.accepted
    assert all(pose_of(s.point_id) != 5 for s in series.samples)


def test_skip_pose_drops_the_current_pose() -> None:
    series, _neutral = make_series()
    series.start(0.0)
    series.update(0.0)
    assert series.pose is not None
    assert series.pose.name == "left"
    assert EVENT_POSE in series.skip_pose(1.0)
    assert series.pose is not None
    assert series.pose.name == "right"
    assert series.skipped_poses == ["left"]


def test_frames_far_from_the_pose_are_dropped() -> None:
    series, neutral = make_series()
    glitch = {"n": 0}

    def behaviour(spec: PoseSpec, waiting: bool) -> float:
        glitch["n"] += 1
        if not waiting and glitch["n"] % 7 == 0:
            return follows(spec, waiting) + 40.0  # a tracking glitch
        return follows(spec, waiting)

    drive(series, neutral, behaviour)
    shifts = [abs(s.features[HEAD["tx"]] - neutral[HEAD["tx"]]) for s in series.samples[:30]]
    assert max(shifts) < 10.0


def test_a_series_needs_base_samples() -> None:
    with pytest.raises(ValueError, match="base"):
        PoseSeries(TWO, [], HEAD)


def _report_with(median: float, base_error: float | None, **kw: object) -> CalibrationReport:
    return CalibrationReport(
        monitor_accuracy=0.99,
        mean_error_px=median,
        median_error_px=median,
        per_monitor_accuracy={0: 1.0},
        n_samples=100,
        n_points=18,
        alpha=1.0,
        grade="excellent",
        per_pose_error_px={} if base_error is None else {0: base_error},
        **kw,  # type: ignore[arg-type]
    )


def test_a_worse_baseline_is_a_regression() -> None:
    old = _report_with(100.0, None).to_dict()  # saved before per-pose errors existed
    assert pose_regression(old, _report_with(0, 119.0)) is None
    assert pose_regression(old, _report_with(0, 121.0)) == (100.0, 121.0)
    again = _report_with(250.0, 100.0).to_dict()  # median over all poses, baseline 100
    assert pose_regression(again, _report_with(0, 110.0)) is None
    assert pose_regression(again, _report_with(0, 130.0)) == (100.0, 130.0)
    assert pose_regression({"grade": "bad"}, _report_with(0, 500.0)) is None
    assert pose_regression(old, _report_with(0, math.nan)) is None
