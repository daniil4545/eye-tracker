"""Tests for the moving-dot stage that follows the calibration dots."""

from __future__ import annotations

import numpy as np
import pytest

from eye_tracker.gaze.calibration import (
    POSE_ID_STRIDE,
    PURSUIT_LAG_S,
    PURSUIT_POSE,
    CalibrationCollector,
    CalibrationSample,
    evaluate,
    make_plan,
    make_pursuit,
    place_groups,
    pose_of,
)
from eye_tracker.types import Observation
from gaze_synth import TWO, calibration_samples


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
