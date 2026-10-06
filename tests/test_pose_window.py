"""The calibration window in head-pose mode (a fake controller, a fake clock)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from eye_tracker.gaze.calibration import POSES, PoseSpec, evaluate, pose_of
from eye_tracker.gaze.store import CalibrationData
from eye_tracker.types import Observation, layout_signature
from eye_tracker.ui import calibration_window as cw
from eye_tracker.ui.calibration_window import CalibrationWindow
from gaze_synth import calibration_samples, synth_features
from test_pose_calibration import GAZE_FEATURES, HEAD, still
from test_ui_dialogs import BACKEND, LEFT, RIGHT, FakeClock, FakeController, _dispose

pytestmark = pytest.mark.usefixtures("qapp")

MONITORS = [LEFT, RIGHT]
FRAME = (640, 480)

Move = Callable[[PoseSpec, bool], float | None]


def make_profile(
    *, base_error: float | None = None, frame_size: tuple[int, int] = FRAME
) -> CalibrationData:
    rng = np.random.default_rng(21)
    samples = still(calibration_samples(MONITORS, rng, 1.0, per_point=15))
    model, report = evaluate(samples, MONITORS, nonlinear=GAZE_FEATURES)
    report_dict = report.to_dict()
    if base_error is not None:
        report_dict["per_pose_error_px"] = {"0": base_error}
    implicit = dataclasses.replace(samples[0], point_id=-1, weight=0.5)
    return CalibrationData(
        backend=BACKEND[0],
        feature_version=BACKEND[1],
        layout_signature=layout_signature(MONITORS),
        monitors=list(MONITORS),
        samples=samples,
        implicit_samples=[implicit],
        model=model,
        report=report_dict,
        camera="0",
        frame_size=frame_size,
    )


def follow(spec: PoseSpec, waiting: bool) -> float:
    return (-2.0 if spec.opposite_of else 2.0) * spec.min_shift


@pytest.fixture
def poses() -> Iterator[tuple[CalibrationWindow, FakeController, FakeClock]]:
    controller = FakeController()
    controller.profile = make_profile()
    controller.head_indices = HEAD
    clock = FakeClock()
    window = CalibrationWindow(
        controller,
        clock=clock,
        poses=True,
        settle_s=0.2,
        collect_s=0.3,
        min_samples=3,
        max_retries=1,
        fit_in_thread=False,
    )
    yield window, controller, clock
    window.cancel()
    _dispose(window)


def run(
    window: CalibrationWindow,
    controller: FakeController,
    clock: FakeClock,
    move: Move,
    *,
    step: float = 0.05,
    frame_size: tuple[int, int] = FRAME,
    seconds: float = 400.0,
) -> None:
    """Feed one observation per tick until the window leaves the running state."""
    rng = np.random.default_rng(22)
    for _ in range(int(seconds / step)):
        if window.state != cw.STATE_RUNNING:
            return
        spec = window.current_pose
        shift = move(spec, window.current_target is None) if spec is not None else None
        if shift is None:
            obs = Observation(timestamp=clock(), face_count=0)
        else:
            target = window.current_target
            point = np.array([[target.x, target.y]]) if target else np.array([[1920.0, 540.0]])
            features = synth_features(point, rng, 1.0)[0]
            features[HEAD[spec.feature]] += shift  # type: ignore[union-attr]
            obs = Observation(
                timestamp=clock(),
                face_count=1,
                features=features,
                quality=1.0,
                frame_size=frame_size,
            )
        controller.observation.emit(obs)
        clock.advance(step)
        window.tick()
    raise AssertionError("the pose run did not finish")


def begin(window: CalibrationWindow) -> None:
    assert window.start()
    QTest.keyClick(window.surfaces()[0], Qt.Key.Key_Space)
    assert window.state == cw.STATE_RUNNING


def test_without_a_usable_calibration_the_poses_do_not_start() -> None:
    controller = FakeController()
    window = CalibrationWindow(controller, poses=True)
    finished: list[bool] = []
    window.finished.connect(finished.append)
    assert not window.start()
    assert finished == [False]
    assert controller.begun == 0
    assert controller.notifications
    assert "alibrat" in controller.notifications[-1][1]
    _dispose(window)


def test_the_intro_names_the_poses_and_the_time(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, _controller, _clock = poses
    assert window.start()
    texts = " ".join(label.text() for label in window.surfaces()[0].findChildren(QLabel))
    assert f"{len(POSES)} head positions" in texts
    assert "seconds" in texts


def feed(
    window: CalibrationWindow, controller: FakeController, clock: FakeClock, shift: float
) -> None:
    """One second of frames with the head ``shift`` away from the usual position."""
    pose = window.current_pose
    assert pose is not None
    rng = np.random.default_rng(23)
    for _ in range(20):
        features = synth_features(np.array([[1920.0, 540.0]]), rng, 1.0)[0]
        features[HEAD[pose.feature]] += shift
        controller.observation.emit(
            Observation(timestamp=clock(), face_count=1, features=features, quality=1.0)
        )
        clock.advance(0.05)
        window.tick()


def test_the_dots_start_only_after_the_head_has_moved(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = poses
    begin(window)
    assert window.hint() == POSES[0].prompt
    feed(window, controller, clock, 0.0)
    assert window.current_target is None  # the head has not moved
    feed(window, controller, clock, 2 * POSES[0].min_shift)
    assert window.current_target is not None
    assert window.hint() != POSES[0].prompt


def test_all_positions_are_added_to_the_profile(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = poses
    profile = controller.profile
    assert profile is not None
    begin(window)
    run(window, controller, clock, follow)
    assert window.state == cw.STATE_FITTING
    window.tick()
    assert window.state == cw.STATE_RESULT
    report = window.report
    assert report is not None
    assert set(report.per_pose_error_px) == set(range(len(POSES) + 1))
    summary = window.surfaces()[0].card.labels["poses"].text()  # type: ignore[attr-defined]
    assert "Usual position" in summary
    assert "Left" in summary
    assert window.save()
    data = controller.finished[-1]
    assert data is not None
    assert {pose_of(s.point_id) for s in data.samples} == set(range(len(POSES) + 1))
    assert data.implicit_samples == profile.implicit_samples
    assert data.key == profile.key
    assert (data.camera, data.frame_size) == (profile.camera, profile.frame_size)
    assert data.model.is_fitted


def test_a_run_that_took_two_poses_keeps_the_others(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = poses
    profile = controller.profile
    assert profile is not None
    first = dataclasses.replace(profile)
    begin(window)
    run(window, controller, clock, follow)
    window.tick()
    assert window.save()
    saved = controller.finished[-1]
    assert saved is not None
    # second run: the user only manages "left" and "tilt"
    controller.profile = saved
    window2 = CalibrationWindow(
        controller,
        clock=clock,
        poses=True,
        settle_s=0.2,
        collect_s=0.3,
        min_samples=3,
        fit_in_thread=False,
    )
    begin(window2)
    run(
        window2, controller, clock, lambda s, w: follow(s, w) if s.name in ("left", "tilt") else 0.0
    )
    window2.tick()
    assert window2.state == cw.STATE_RESULT
    assert window2.save()
    data = controller.finished[-1]
    assert data is not None
    assert {pose_of(s.point_id) for s in data.samples} == set(range(len(POSES) + 1))
    assert first.samples  # the base samples were never touched
    _dispose(window2)


def test_fewer_than_two_positions_is_an_error_and_keeps_the_profile(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = poses
    begin(window)
    run(window, controller, clock, lambda s, w: 0.0, step=0.25, seconds=1000.0)
    assert window.state == cw.STATE_ERROR
    assert "positions" in window.error
    assert controller.finished == []


def test_a_worse_usual_position_keeps_the_old_calibration(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = poses
    controller.profile = make_profile(base_error=1.0)  # nothing can match that
    begin(window)
    run(window, controller, clock, follow)
    window.tick()
    assert window.state == cw.STATE_ERROR
    assert "not changed" in window.error
    assert "1 px" in window.error
    window.cancel()
    assert controller.finished == [None]


def test_another_camera_is_refused(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = poses
    begin(window)
    run(window, controller, clock, follow, frame_size=(1280, 480))
    window.tick()
    assert window.state == cw.STATE_ERROR
    assert "camera" in window.error.lower()
    assert controller.finished == []


def test_no_face_for_a_while_skips_the_position(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, clock = poses
    begin(window)
    for _ in range(int((cw.POSE_NO_FACE_S + 2) / 0.25)):
        controller.observation.emit(Observation(timestamp=clock(), face_count=0))
        clock.advance(0.25)
        window.tick()
    pose = window.current_pose
    assert pose is not None
    assert pose.name != "left"  # given up
    assert window.state == cw.STATE_RUNNING


def test_escape_during_the_poses_changes_nothing(
    poses: tuple[CalibrationWindow, FakeController, FakeClock],
) -> None:
    window, controller, _clock = poses
    begin(window)
    QTest.keyClick(window.surfaces()[0], Qt.Key.Key_Escape)
    assert window.state == cw.STATE_CLOSED
    assert controller.finished == [None]
