"""Calibration: where to show targets, how to collect samples, how good the result is.

The flow driven by the calibration window is::

    plan = make_plan(monitors)                       # dots to show, in order
    collector = CalibrationCollector(plan)
    collector.start(now)
    ...every UI tick:   events = collector.update(now)
    ...every frame:     collector.add(observation)
    model, report = evaluate(collector.samples, monitors, nonlinear=gaze_indices)

``gaze_indices`` are the backend's gaze-direction features
(``model.gaze_feature_indices(backend.feature_names, backend.gaze_features)``).
Everything here is plain Python/numpy and driven by an injected clock, so the
whole procedure is unit-testable without a camera or a screen.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from ..types import Monitor, Observation, nearest_monitor, virtual_bounds
from .model import DEFAULT_ALPHAS, SUPPORTED_DEGREES, GazeModel, lopo_predictions, select_model

log = logging.getLogger(__name__)

# Collector phases.
PHASE_IDLE = "idle"
PHASE_SETTLE = "settle"
PHASE_COLLECT = "collect"
PHASE_DONE = "done"

# Events returned by CalibrationCollector.update().
EVENT_TARGET = "target"
EVENT_RETRY = "retry"
EVENT_FINISHED = "finished"

#: Minimum monitor accuracy for each grade, best first.
GRADE_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("excellent", 0.97),
    ("good", 0.90),
    ("fair", 0.75),
)

#: Head-pose calibration: the dots of pose ``n`` (1, 2, ...) have the ids
#: ``n * POSE_ID_STRIDE + k``; the ordinary calibration is pose 0, learned samples
#: have negative ids.
POSE_ID_STRIDE = 1000

MIN_SAMPLES = 10
MIN_POINTS = 3

#: Percentile of the absolute held-out error per axis that is reported as a
#: monitor's gaze error (:attr:`CalibrationReport.per_monitor_error_px`). The 75th
#: covers most looks without being dominated by the few dots the face was half
#: lost on.
AXIS_ERROR_PERCENTILE = 75.0

# Timer ticks are sums of floats (11.8 + 0.8 = 12.599999…); without a little
# slack a phase could end one tick late.
_TIME_EPS = 1e-6


# ------------------------------------------------------------------------ plan
@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """One dot of the calibration sequence.

    ``nx``/``ny`` are relative to the monitor (0..1), ``x``/``y`` are global
    coordinates of the same point.
    """

    point_id: int
    monitor_index: int
    nx: float
    ny: float
    x: float
    y: float


def make_plan(
    monitors: Sequence[Monitor],
    points_per_monitor: int = 9,
    margin: float = 0.1,
    first_id: int = 0,
) -> list[CalibrationTarget]:
    """Calibration targets for every monitor, in the order they should be shown.

    ``points_per_monitor``: 1 (centre), 5 (centre + 4 corners) or a square number
    ``k*k`` (a ``k x k`` grid spanning ``margin .. 1 - margin``). Monitors are
    visited left to right (then top to bottom). Within a monitor the grid is
    walked as a serpentine; the orientation of that walk is chosen so that it
    starts at the corner nearest to where the previous monitor ended, which keeps
    eye and head travel short. ``point_id`` counts up from ``first_id``.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"margin must be in [0, 0.5), got {margin!r}")
    base = _monitor_path(points_per_monitor, margin)
    variants = [
        base,
        [(1.0 - nx, ny) for nx, ny in base],
        [(nx, 1.0 - ny) for nx, ny in base],
        [(1.0 - nx, 1.0 - ny) for nx, ny in base],
    ]

    plan: list[CalibrationTarget] = []
    last: tuple[float, float] | None = None
    for monitor in sorted(monitors, key=lambda m: (m.rect.x, m.rect.y, m.index)):
        rect = monitor.rect
        path = base
        if last is not None:
            distances = [math.dist(rect.denormalize(*v[0]), last) for v in variants]
            # index() returns the first of equally good variants, so the canonical
            # top-left start wins ties.
            path = variants[distances.index(min(distances))]
        for nx, ny in path:
            x, y = rect.denormalize(nx, ny)
            plan.append(CalibrationTarget(first_id + len(plan), monitor.index, nx, ny, x, y))
        last = (plan[-1].x, plan[-1].y)
    return plan


def _monitor_path(points: int, margin: float) -> list[tuple[float, float]]:
    lo, hi = margin, 1.0 - margin
    if points == 1:
        return [(0.5, 0.5)]
    if points == 5:
        # Centre first (easiest to find), then the corners as a loop.
        return [(0.5, 0.5), (lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    k = math.isqrt(points) if points > 0 else 0
    if k < 2 or k * k != points:
        raise ValueError(f"points_per_monitor must be 1, 5 or a square number >= 4, got {points!r}")
    coords = np.linspace(lo, hi, k).tolist()
    path: list[tuple[float, float]] = []
    for row, ny in enumerate(coords):
        xs = coords if row % 2 == 0 else coords[::-1]
        path.extend((nx, ny) for nx in xs)
    return path


# --------------------------------------------------------------------- samples
@dataclass(eq=False)
class CalibrationSample:
    """A feature vector labelled with the global point the user was looking at.

    ``point_id`` groups samples of one calibration dot (implicit samples learned
    from mouse use carry negative ids). ``weight`` scales the sample's influence
    on the fit.
    """

    features: np.ndarray
    x: float
    y: float
    monitor_index: int
    point_id: int
    weight: float = 1.0


def place_groups(samples: Iterable[CalibrationSample]) -> np.ndarray:
    """One group id per sample: the place looked at, to the pixel.

    The same place in another pose (another ``point_id``) is the same group, so
    cross-validation never trains on a dot it is asked to predict. Mirrored plan
    variants differ in the last float digits, hence the rounding.
    """
    places: dict[tuple[int, int, int], int] = {}
    return np.array(
        [
            places.setdefault((s.monitor_index, round(float(s.x)), round(float(s.y))), len(places))
            for s in samples
        ],
        dtype=np.int64,
    )


def pose_of(point_id: int) -> int:
    """Pose number of a calibration sample: 0 for the ordinary calibration."""
    return point_id // POSE_ID_STRIDE if point_id >= 0 else -1


def samples_to_arrays(
    samples: Iterable[CalibrationSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack samples into ``X (n, d)``, targets ``Y (n, 2)`` and ``weights (n,)``."""
    items = list(samples)
    if not items:
        raise ValueError("no samples")
    X = np.vstack([np.asarray(s.features, dtype=np.float64).reshape(1, -1) for s in items])
    Y = np.array([(s.x, s.y) for s in items], dtype=np.float64)
    W = np.array([s.weight for s in items], dtype=np.float64)
    return X, Y, W


# ------------------------------------------------------------------- collector
class CalibrationCollector:
    """Time-based state machine that shows each target and gathers its samples.

    Per target: ``settle`` (the eyes travel to the new dot; nothing is recorded)
    then ``collect`` (usable observations are recorded). A target that gathered
    fewer than ``min_samples`` when its collect window ends gets up to
    ``max_retries`` extra collect windows (samples so far are kept) and is then
    skipped, dropping its samples.

    Every phase starts at the ``now`` of the :meth:`update` call that entered it,
    so a stalled UI thread never eats into the next target's time. Progress
    properties describe the state as of the last :meth:`start`/:meth:`update`.
    """

    def __init__(
        self,
        plan: Sequence[CalibrationTarget],
        settle_s: float = 0.8,
        collect_s: float = 1.0,
        min_samples: int = 5,
        max_retries: int = 1,
    ) -> None:
        if settle_s < 0 or collect_s <= 0:
            raise ValueError("settle_s must be >= 0 and collect_s > 0")
        if min_samples < 1 or max_retries < 0:
            raise ValueError("min_samples must be >= 1 and max_retries >= 0")
        self._plan: tuple[CalibrationTarget, ...] = tuple(plan)
        self.settle_s = float(settle_s)
        self.collect_s = float(collect_s)
        self.min_samples = int(min_samples)
        self.max_retries = int(max_retries)
        self._reset()

    def _reset(self) -> None:
        self._phase = PHASE_IDLE
        self._index = 0
        self._phase_start = 0.0
        self._now = 0.0
        self._retries = 0
        self._pending: list[CalibrationSample] = []
        self._samples: list[CalibrationSample] = []
        self._skipped: list[int] = []
        self._events: list[str] = []
        self._n_features: int | None = None
        self._frame_sizes: Counter[tuple[int, int]] = Counter()

    # ------------------------------------------------------------- control
    def start(self, now: float) -> None:
        """(Re)start from the first target. The next :meth:`update` reports ``"target"``
        (or ``"finished"`` for an empty plan)."""
        self._reset()
        self._now = now
        if not self._plan:
            self._phase = PHASE_DONE
            self._events.append(EVENT_FINISHED)
            return
        self._enter(PHASE_SETTLE, now)
        self._events.append(EVENT_TARGET)

    def update(self, now: float) -> list[str]:
        """Advance the timers; returns the events since the last call, in order:
        ``"target"`` (a new target is current), ``"retry"`` (the current target gets
        another collect window) and ``"finished"``."""
        if self._phase in (PHASE_SETTLE, PHASE_COLLECT):
            self._now = max(self._now, now)
            # Loop so zero-length phases (settle_s == 0) pass in a single call.
            while self._step(self._now):
                pass
        events, self._events = self._events, []
        return events

    def add(self, obs: Observation) -> bool:
        """Offer an observation; returns True if it was recorded for the current target.

        Only usable observations that were really analysed are accepted: motion-gate
        copies (``skipped``) may repeat a frame from before the target appeared.
        """
        target = self.current
        if self._phase != PHASE_COLLECT or target is None:
            return False
        if not obs.usable or obs.skipped or obs.features is None:
            return False
        features = np.array(obs.features, dtype=np.float64).reshape(-1)
        if features.size == 0 or not np.all(np.isfinite(features)):
            return False
        if self._n_features is None:
            self._n_features = features.size
        elif features.size != self._n_features:
            log.warning(
                "Ignoring observation with %d features (expected %d)",
                features.size,
                self._n_features,
            )
            return False
        self._pending.append(
            CalibrationSample(features, target.x, target.y, target.monitor_index, target.point_id)
        )
        width, height = obs.frame_size
        if width > 0 and height > 0:
            self._frame_sizes[(int(width), int(height))] += 1
        return True

    # ------------------------------------------------------------- properties
    @property
    def plan(self) -> tuple[CalibrationTarget, ...]:
        return self._plan

    @property
    def phase(self) -> str:
        """``"idle"``, ``"settle"``, ``"collect"`` or ``"done"``."""
        return self._phase

    @property
    def current(self) -> CalibrationTarget | None:
        """Target being shown, or ``None`` before start and after the last target."""
        if self._phase in (PHASE_SETTLE, PHASE_COLLECT):
            return self._plan[self._index]
        return None

    @property
    def current_index(self) -> int:
        """Position of the current target in the plan."""
        return self._index

    @property
    def phase_progress(self) -> float:
        """0..1 through the current phase (a retry restarts the collect phase)."""
        elapsed = self._now - self._phase_start
        if self._phase == PHASE_SETTLE:
            return 1.0 if self.settle_s == 0 else _clamp01(elapsed / self.settle_s)
        if self._phase == PHASE_COLLECT:
            return _clamp01(elapsed / self.collect_s)
        return 1.0 if self._phase == PHASE_DONE else 0.0

    @property
    def point_progress(self) -> float:
        """0..1 through the current target, settle and collect time combined."""
        total = self.settle_s + self.collect_s
        if self._phase == PHASE_SETTLE:
            return self.phase_progress * self.settle_s / total
        if self._phase == PHASE_COLLECT:
            return (self.settle_s + self.phase_progress * self.collect_s) / total
        return 1.0 if self._phase == PHASE_DONE else 0.0

    @property
    def progress(self) -> float:
        """0..1 through the whole plan."""
        if self._phase == PHASE_DONE:
            return 1.0
        if self._phase == PHASE_IDLE or not self._plan:
            return 0.0
        return (self._index + self.point_progress) / len(self._plan)

    @property
    def current_sample_count(self) -> int:
        """Samples recorded so far for the current target."""
        return len(self._pending)

    @property
    def samples(self) -> list[CalibrationSample]:
        """Samples of all completed (not skipped) targets."""
        return list(self._samples)

    @property
    def skipped_points(self) -> list[int]:
        """Point ids that were skipped for lack of usable observations."""
        return list(self._skipped)

    @property
    def frame_size(self) -> tuple[int, int]:
        """Camera frame size ``(w, h)`` of the recorded observations, ``(0, 0)`` if unknown.

        The most common one if the camera changed its size mid-calibration. Store
        it in ``CalibrationData.frame_size`` so a later change of the camera's
        aspect ratio is noticed.
        """
        if not self._frame_sizes:
            return (0, 0)
        return self._frame_sizes.most_common(1)[0][0]

    # ------------------------------------------------------------- internals
    def _enter(self, phase: str, now: float) -> None:
        self._phase = phase
        self._phase_start = now

    def _step(self, now: float) -> bool:
        """Perform at most one transition; True if one happened."""
        elapsed = now - self._phase_start + _TIME_EPS
        if self._phase == PHASE_SETTLE:
            if elapsed < self.settle_s:
                return False
            self._enter(PHASE_COLLECT, now)
            return True
        if self._phase != PHASE_COLLECT or elapsed < self.collect_s:
            return False

        target = self._plan[self._index]
        if len(self._pending) >= self.min_samples:
            self._samples.extend(self._pending)
        elif self._retries < self.max_retries:
            self._retries += 1
            self._enter(PHASE_COLLECT, now)
            self._events.append(EVENT_RETRY)
            log.debug(
                "Calibration point %d: %d/%d samples, retrying",
                target.point_id,
                len(self._pending),
                self.min_samples,
            )
            return True
        else:
            self._skipped.append(target.point_id)
            log.info(
                "Calibration point %d skipped (%d usable samples)",
                target.point_id,
                len(self._pending),
            )

        self._pending = []
        self._retries = 0
        self._index += 1
        if self._index >= len(self._plan):
            self._enter(PHASE_DONE, now)
            self._events.append(EVENT_FINISHED)
            return False
        self._enter(PHASE_SETTLE, now)
        self._events.append(EVENT_TARGET)
        return True


# ------------------------------------------------------------------- head poses
EVENT_POSE = "pose"  # a new pose is asked for
EVENT_POSE_MORE = "pose_more"  # not reached in time: asked again
EVENT_POSE_SKIPPED = "pose_skipped"  # given up

PHASE_WAIT = "wait"  # waiting for the head to take the pose

#: Seconds the head gets to reach a pose, per ask (a pose is asked for twice).
POSE_WAIT_S = 15.0
#: Seconds the head must stay in the pose before the dots start.
POSE_HOLD_S = 0.5
#: Frames whose median tells where the head is.
POSE_FRAMES = 10
#: Dots per monitor in a pose: the centre and the corners.
POSE_POINTS = 5
#: A pose needs this share of its dots to have samples ...
POSE_MIN_DOT_SHARE = 0.6
#: ... and its samples must hold this share of the asked shift.
POSE_KEEP_SHARE = 0.6
#: Fewer poses than this are no calibration.
POSE_MIN_ACCEPTED = 2
#: Frames this far from the pose's median head position (cm, degrees) are glitches.
POSE_OUTLIER_CM = 6.0
POSE_OUTLIER_DEG = 10.0
#: New baseline error may grow by this factor before the old calibration is kept.
POSE_REGRESSION = 1.2
#: Polynomial degree tried for a calibration with head poses, every feature expanded:
#: the iris x head terms hold the usual pose and the shifted ones together (issue #2).
POSE_DEGREE = 2


@dataclass(frozen=True, slots=True)
class PoseSpec:
    """A head position to calibrate in: the feature that moves and by how much.

    The sign of the shift is not checked, because mirroring depends on the camera;
    a pose with ``opposite_of`` must only go the other way than that pose did.
    """

    name: str
    feature: str
    min_shift: float  # cm for tx, ty, tz; degrees for roll
    prompt: str
    opposite_of: str = ""


POSES: tuple[PoseSpec, ...] = (
    PoseSpec("left", "tx", 3.0, "Move your head to the left"),
    PoseSpec("right", "tx", 3.0, "Move your head to the right", "left"),
    PoseSpec("closer", "tz", 4.0, "Lean in, closer to the screen"),
    PoseSpec("farther", "tz", 4.0, "Lean back, away from the screen", "closer"),
    PoseSpec("lower", "ty", 3.0, "Lower your head"),
    PoseSpec("tilt", "roll", 5.0, "Tilt your head a little to one side"),
)
HEAD_FEATURES = ("roll", "tx", "ty", "tz")


class PoseSeries:
    """The calibration dots in several head poses, one pose after the other.

    For each pose it waits until the head has moved (median of the last frames
    differs from the head of the ordinary calibration by ``min_shift``), then runs
    a :class:`CalibrationCollector` over :data:`POSE_POINTS` dots per monitor. A
    pose whose samples do not hold the shift or that lost too many dots is asked
    once more. It offers the collector's interface (``update``, ``add``,
    ``current``, ``progress``, ...) so the window drives it the same way.
    """

    def __init__(
        self,
        monitors: Sequence[Monitor],
        base_samples: Sequence[CalibrationSample],
        head_indices: Mapping[str, int],
        *,
        poses: Sequence[PoseSpec] = POSES,
        settle_s: float = 0.8,
        collect_s: float = 1.0,
        min_samples: int = 5,
        max_retries: int = 1,
        margin: float = 0.1,
    ) -> None:
        base = [s for s in base_samples if 0 <= s.point_id < POSE_ID_STRIDE]
        if not base:
            raise ValueError("a pose series needs base samples")
        if not all(name in head_indices for name in HEAD_FEATURES):
            raise ValueError("a pose series needs the head features")
        self._head = {name: int(head_indices[name]) for name in HEAD_FEATURES}
        feats = np.array([s.features for s in base])
        self._neutral = {n: float(np.median(feats[:, i])) for n, i in self._head.items()}
        self._poses = tuple(poses)
        self._plans = [
            make_plan(monitors, POSE_POINTS, margin, first_id=(n + 1) * POSE_ID_STRIDE)
            for n in range(len(self._poses))
        ]
        self._timing = (settle_s, collect_s, min_samples, max_retries)
        self._reset()

    def _reset(self) -> None:
        self._index = 0
        self._phase = PHASE_IDLE
        self._now = 0.0
        self._asked = 0
        self._tries = 0
        self._wait_start = 0.0
        self._hold_since: float | None = None
        self._recent: deque[np.ndarray] = deque(maxlen=POSE_FRAMES)
        self._collector: CalibrationCollector | None = None
        self._samples: list[CalibrationSample] = []
        self._shifts: dict[str, float] = {}
        self._skipped_poses: list[str] = []
        self._skipped_points: list[int] = []
        self._frame_sizes: Counter[tuple[int, int]] = Counter()
        self._events: list[str] = []

    # ------------------------------------------------------------- control
    def start(self, now: float) -> None:
        self._reset()
        self._now = now
        self._enter_wait(now)

    def update(self, now: float) -> list[str]:
        """Advance the clock; returns the events since the last call (the collector's
        ``target``, ``retry`` and the pose events above, ``finished`` at the end)."""
        self._now = max(self._now, now)
        now = self._now
        if self._phase == PHASE_WAIT:
            self._step_wait(now)
        elif self._collector is not None:
            for event in self._collector.update(now):
                if event == EVENT_FINISHED:
                    self._finish_pose(now)
                else:
                    self._events.append(event)
        events, self._events = self._events, []
        return events

    def add(self, obs: Observation) -> bool:
        """Offer an observation: moves the wait forward or records a dot sample."""
        if self._phase == PHASE_WAIT:
            self._watch(obs)
            return False
        return self._collector is not None and self._collector.add(obs)

    def skip_pose(self, now: float) -> list[str]:
        """Give the current pose up (the face was lost for too long)."""
        if self._phase in (PHASE_IDLE, PHASE_DONE):
            return []
        self._now = max(self._now, now)
        self._drop_pose(self._now)
        events, self._events = self._events, []
        return events

    # ------------------------------------------------------------- properties
    @property
    def pose(self) -> PoseSpec | None:
        return self._poses[self._index] if self._index < len(self._poses) else None

    @property
    def pose_number(self) -> int:
        """1-based number of the current pose."""
        return self._index + 1

    @property
    def asked_again(self) -> bool:
        return self._phase == PHASE_WAIT and self._asked > 1

    @property
    def accepted(self) -> list[int]:
        """Numbers (1-based) of the poses taken."""
        return sorted({pose_of(s.point_id) for s in self._samples})

    @property
    def skipped_poses(self) -> list[str]:
        return list(self._skipped_poses)

    @property
    def plan(self) -> tuple[CalibrationTarget, ...]:
        return tuple(t for plan in self._plans for t in plan)

    @property
    def phase(self) -> str:
        if self._phase == PHASE_COLLECT and self._collector is not None:
            return self._collector.phase
        return self._phase

    @property
    def current(self) -> CalibrationTarget | None:
        return self._collector.current if self._collector is not None else None

    @property
    def current_index(self) -> int:
        inner = self._collector.current_index if self._collector is not None else 0
        return min(self._index, len(self._plans) - 1) * len(self._plans[0]) + inner

    @property
    def phase_progress(self) -> float:
        return self._collector.phase_progress if self._collector is not None else 0.0

    @property
    def progress(self) -> float:
        if self._phase == PHASE_DONE:
            return 1.0
        inner = self._collector.progress if self._collector is not None else 0.0
        return (self._index + inner) / len(self._poses)

    @property
    def samples(self) -> list[CalibrationSample]:
        """Samples of the poses taken, glitch frames removed."""
        return list(self._samples)

    @property
    def skipped_points(self) -> list[int]:
        return list(self._skipped_points)

    @property
    def frame_size(self) -> tuple[int, int]:
        if not self._frame_sizes:
            return (0, 0)
        return self._frame_sizes.most_common(1)[0][0]

    # ------------------------------------------------------------- internals
    def _enter_wait(self, now: float) -> None:
        self._phase = PHASE_WAIT
        self._collector = None
        self._asked = 1
        self._wait_start = now
        self._hold_since = None
        self._recent.clear()
        self._events.append(EVENT_POSE)

    def _step_wait(self, now: float) -> None:
        if self._hold_since is not None and now - self._hold_since >= POSE_HOLD_S:
            self._begin_dots(now)
        elif now - self._wait_start >= POSE_WAIT_S:
            if self._asked < 2:
                self._asked += 1
                self._wait_start = now
                self._events.append(EVENT_POSE_MORE)
            else:
                self._drop_pose(now)

    def _watch(self, obs: Observation) -> None:
        if not obs.usable or obs.skipped or obs.features is None:
            return
        features = np.asarray(obs.features, dtype=np.float64).reshape(-1)
        if features.size <= max(self._head.values()) or not np.all(np.isfinite(features)):
            return
        self._recent.append(features)
        pose = self._poses[self._index]
        if len(self._recent) < POSE_FRAMES or not self._moved(pose, self._median_shift(pose)):
            self._hold_since = None
        elif self._hold_since is None:
            self._hold_since = self._now

    def _median_shift(self, pose: PoseSpec, frames: Sequence[np.ndarray] | None = None) -> float:
        rows = list(self._recent) if frames is None else frames
        column = self._head[pose.feature]
        return float(np.median([row[column] for row in rows])) - self._neutral[pose.feature]

    def _moved(self, pose: PoseSpec, shift: float, share: float = 1.0) -> bool:
        if abs(shift) < share * pose.min_shift:
            return False
        partner = self._shifts.get(pose.opposite_of)
        return partner is None or shift * partner < 0

    def _begin_dots(self, now: float) -> None:
        settle_s, collect_s, min_samples, max_retries = self._timing
        self._phase = PHASE_COLLECT
        self._collector = CalibrationCollector(
            self._plans[self._index],
            settle_s=settle_s,
            collect_s=collect_s,
            min_samples=min_samples,
            max_retries=max_retries,
        )
        self._collector.start(now)
        self._events.extend(self._collector.update(now))

    def _finish_pose(self, now: float) -> None:
        collector = self._collector
        pose = self._poses[self._index]
        assert collector is not None
        self._skipped_points.extend(collector.skipped_points)
        if collector.frame_size != (0, 0):
            self._frame_sizes[collector.frame_size] += 1
        plan = self._plans[self._index]
        kept = self._without_glitches(collector.samples)
        enough = len(plan) - len(collector.skipped_points) >= POSE_MIN_DOT_SHARE * len(plan)
        shift = self._median_shift(pose, [s.features for s in kept]) if kept else 0.0
        if enough and kept and self._moved(pose, shift, POSE_KEEP_SHARE):
            self._samples.extend(kept)
            self._shifts[pose.name] = shift
            self._next_pose(now)
        elif self._tries < 1:
            self._tries += 1
            self._enter_wait(now)
        else:
            self._drop_pose(now)

    def _drop_pose(self, now: float) -> None:
        self._skipped_poses.append(self._poses[self._index].name)
        self._events.append(EVENT_POSE_SKIPPED)
        self._next_pose(now)

    def _next_pose(self, now: float) -> None:
        self._index += 1
        self._tries = 0
        if self._index >= len(self._poses):
            self._phase = PHASE_DONE
            self._collector = None
            self._events.append(EVENT_FINISHED)
        else:
            self._enter_wait(now)

    def _without_glitches(self, samples: Sequence[CalibrationSample]) -> list[CalibrationSample]:
        if not samples:
            return []
        columns = list(self._head.values())
        limits = np.array(
            [POSE_OUTLIER_DEG if name == "roll" else POSE_OUTLIER_CM for name in self._head]
        )
        head = np.array([s.features[columns] for s in samples])
        keep = np.all(np.abs(head - np.median(head, axis=0)) <= limits, axis=1)
        return [s for s, ok in zip(samples, keep, strict=True) if ok]


def merge_poses(
    stored: Sequence[CalibrationSample], taken: Sequence[CalibrationSample]
) -> list[CalibrationSample]:
    """``stored`` with the poses of ``taken`` replaced; the poses it lacks stay."""
    replaced = {pose_of(s.point_id) for s in taken}
    return [s for s in stored if pose_of(s.point_id) not in replaced] + list(taken)


def balance_pose_weights(samples: Sequence[CalibrationSample]) -> list[CalibrationSample]:
    """Pose samples reweighted so that together they weigh as much as the ordinary
    dots: without it the many pose samples outweigh the usual pose."""
    base = sum(s.weight for s in samples if pose_of(s.point_id) == 0)
    count = sum(1 for s in samples if pose_of(s.point_id) > 0)
    if base <= 0 or count == 0:
        return list(samples)
    weight = base / count
    return [replace(s, weight=weight) if pose_of(s.point_id) > 0 else s for s in samples]


def evaluate_poses(
    samples: Sequence[CalibrationSample],
    monitors: Sequence[Monitor],
    old_report: dict[str, Any],
    *,
    nonlinear: Sequence[int] | None = None,
) -> tuple[GazeModel, CalibrationReport]:
    """:func:`evaluate` for a calibration with head poses, fitted two ways: the usual
    model, and :data:`POSE_DEGREE` over every feature (the better one on real data,
    issue #2). Keeps the one that passes :func:`pose_regression` with the lower mean
    error of the head positions; the second way when neither passes."""
    candidates = [
        evaluate(samples, monitors, POSE_DEGREE),
        evaluate(samples, monitors, nonlinear=nonlinear),
    ]
    passing = [c for c in candidates if pose_regression(old_report, c[1]) is None]
    if not passing:
        return candidates[0]
    return min(passing, key=lambda c: _mean_pose_error(c[1]))


def _mean_pose_error(report: CalibrationReport) -> float:
    errors = [v for k, v in report.per_pose_error_px.items() if k > 0]
    return float(np.mean(errors)) if errors else math.inf


def pose_regression(
    old_report: dict[str, Any], report: CalibrationReport
) -> tuple[float, float] | None:
    """``(old, new)`` when the head positions make the calibration worse, else ``None``.

    Worse means the ordinary dots (pose 0) lost more than :data:`POSE_REGRESSION`
    (old ``per_pose_error_px[0]``, or ``median_error_px`` for reports without it),
    or the mean error of the head positions lost as much against an old report with them.
    """
    try:
        old = CalibrationReport.from_dict(old_report)
    except ValueError:
        return None
    before = old.per_pose_error_px.get(0, old.median_error_px)
    after = report.per_pose_error_px.get(0, math.nan)
    if not (math.isfinite(before) and math.isfinite(after)) or before <= 0:
        return None
    if after > POSE_REGRESSION * before:
        return before, after
    old_poses = [v for k, v in old.per_pose_error_px.items() if k > 0]
    new_poses = [v for k, v in report.per_pose_error_px.items() if k > 0]
    if old_poses and new_poses:
        before, after = float(np.mean(old_poses)), float(np.mean(new_poses))
        if after > POSE_REGRESSION * before:
            return before, after
    return None


# ------------------------------------------------------------------ evaluation
@dataclass
class CalibrationReport:
    """Quality of a calibration, estimated by leave-one-point-out cross-validation."""

    #: Fraction of samples whose held-out prediction is nearest to the right monitor.
    monitor_accuracy: float
    mean_error_px: float
    median_error_px: float
    per_monitor_accuracy: dict[int, float]
    n_samples: int
    n_points: int
    alpha: float
    grade: str
    #: Polynomial degree chosen for the model (see ``gaze.model``).
    degree: int = 2
    #: Monitors without a single surviving calibration dot (usually the face was
    #: not found while looking at them). They appear in ``per_monitor_accuracy``
    #: with 0.0 and cap the grade at "fair".
    uncovered_monitors: list[int] = field(default_factory=list)
    #: Gaze error per monitor and axis, ``{monitor: (x_px, y_px)}``: the
    #: :data:`AXIS_ERROR_PERCENTILE` th percentile of the absolute held-out error
    #: of that monitor's samples. Split panes are only told apart where they are
    #: large compared with it. Empty in reports saved before it existed (see
    #: :func:`eye_tracker.gaze.store.axis_error`) and for uncovered monitors.
    per_monitor_error_px: dict[int, tuple[float, float]] = field(default_factory=dict)
    #: Median held-out error in pixels per pose (0 is the ordinary calibration); the
    #: dot is held out, the pose is not. Empty in reports saved before it existed.
    per_pose_error_px: dict[int, float] = field(default_factory=dict)
    #: Median error per pose when the whole pose is held out: what to expect in a
    #: head position that was not calibrated. Empty with fewer than two poses.
    unseen_pose_error_px: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        """One human-readable line, e.g. ``"Excellent — 99% monitor accuracy, 180 samples"``."""
        # Floor, so "100%" is only ever shown for a perfect result.
        pct = math.floor(self.monitor_accuracy * 100.0 + 1e-9)
        text = f"{self.grade.capitalize()} — {pct}% monitor accuracy, {self.n_samples} samples"
        if math.isfinite(self.mean_error_px):
            text += f", mean error {self.mean_error_px:.0f} px"
        if self.uncovered_monitors:
            n = len(self.uncovered_monitors)
            text += f"; {n} screen{'s' if n > 1 else ''} not calibrated"
        return text

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (monitor keys become strings, non-finite numbers ``None``)."""
        out = asdict(self)
        out["per_monitor_accuracy"] = {str(k): v for k, v in self.per_monitor_accuracy.items()}
        out["per_monitor_error_px"] = {
            str(k): [_json_number(float(v)) for v in xy]
            for k, xy in self.per_monitor_error_px.items()
        }
        for key in ("per_pose_error_px", "unseen_pose_error_px"):
            out[key] = {str(k): _json_number(float(v)) for k, v in getattr(self, key).items()}
        return {k: _json_number(v) for k, v in out.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationReport:
        """Inverse of :meth:`to_dict`. Raises ``ValueError`` on malformed input."""
        try:
            per_monitor = {
                int(k): float(v) for k, v in dict(data.get("per_monitor_accuracy") or {}).items()
            }
            return cls(
                monitor_accuracy=_float(data["monitor_accuracy"]),
                mean_error_px=_float(data["mean_error_px"]),
                median_error_px=_float(data["median_error_px"]),
                per_monitor_accuracy=per_monitor,
                n_samples=int(data["n_samples"]),
                n_points=int(data["n_points"]),
                alpha=_float(data["alpha"]),
                grade=str(data["grade"]),
                degree=int(data.get("degree", 2)),
                uncovered_monitors=[int(i) for i in data.get("uncovered_monitors") or []],
                per_monitor_error_px=axis_errors_from_dict(data.get("per_monitor_error_px")),
                per_pose_error_px=pose_errors_from_dict(data.get("per_pose_error_px")),
                unseen_pose_error_px=pose_errors_from_dict(data.get("unseen_pose_error_px")),
            )
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise ValueError(f"invalid calibration report: {exc}") from exc


def pose_errors_from_dict(value: Any) -> dict[int, float]:
    """``per_pose_error_px`` as stored (``{"0": 120.0}``); damaged entries are dropped."""
    if not isinstance(value, dict):
        return {}
    out: dict[int, float] = {}
    for key, error in value.items():
        try:
            pose, px = int(key), float(error)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(px) and px >= 0.0:
            out[pose] = px
    return dict(sorted(out.items()))


def axis_errors_from_dict(value: Any) -> dict[int, tuple[float, float]]:
    """``per_monitor_error_px`` as stored (``{"0": [x, y]}``); absent or null gives ``{}``.

    Entries that are not two finite, non-negative numbers are dropped: a damaged
    entry only means "unknown", which callers estimate otherwise. A value that is
    not an object at all raises ``ValueError``, like the rest of a malformed report.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("per_monitor_error_px must be an object")
    out: dict[int, tuple[float, float]] = {}
    for key, pair in value.items():
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            continue
        try:
            ex, ey = float(pair[0]), float(pair[1])
            index = int(key)
        except (TypeError, ValueError, OverflowError):
            continue
        if all(math.isfinite(v) and v >= 0.0 for v in (ex, ey)):
            out[index] = (ex, ey)
    return dict(sorted(out.items()))


def per_axis_errors(
    predictions: np.ndarray, targets: np.ndarray, monitor_indices: Sequence[int]
) -> dict[int, tuple[float, float]]:
    """``{monitor: (x_px, y_px)}``: the :data:`AXIS_ERROR_PERCENTILE` th percentile of
    the absolute error on each axis over each monitor's samples.

    ``predictions`` and ``targets`` are ``(n, 2)`` arrays (held-out predictions and
    the points looked at); ``monitor_indices`` gives each row's monitor. Rows with
    a non-finite prediction are ignored; a monitor without a finite row is left out.
    """
    preds = np.asarray(predictions, dtype=np.float64).reshape(-1, 2)
    truth = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    owners = np.asarray([int(i) for i in monitor_indices], dtype=np.int64)
    if not preds.shape[0] == truth.shape[0] == owners.shape[0]:
        raise ValueError("predictions, targets and monitor indices differ in length")
    err = np.abs(preds - truth)
    finite = np.isfinite(err).all(axis=1)
    out: dict[int, tuple[float, float]] = {}
    for idx in sorted(set(owners.tolist())):
        rows = err[(owners == idx) & finite]
        if rows.shape[0] == 0:
            continue
        ex, ey = np.percentile(rows, AXIS_ERROR_PERCENTILE, axis=0)
        out[int(idx)] = (float(ex), float(ey))
    return out


def grade_for(monitor_accuracy: float) -> str:
    """``"excellent"`` (>= 0.97), ``"good"`` (>= 0.9), ``"fair"`` (>= 0.75) or ``"poor"``."""
    for name, threshold in GRADE_THRESHOLDS:
        if monitor_accuracy >= threshold:
            return name
    return "poor"


def evaluate(
    samples: Sequence[CalibrationSample],
    monitors: Sequence[Monitor],
    degree: int | None = None,
    *,
    nonlinear: Sequence[int] | None = None,
) -> tuple[GazeModel, CalibrationReport]:
    """Fit the final model and estimate how well it will work.

    Hyper-parameters are chosen by leave-one-point-out cross-validation (every
    fold hides all samples of one calibration dot): ``alpha`` always, and the
    polynomial degree too unless ``degree`` is given. The report is computed from
    the held-out predictions of the chosen settings, so it reflects accuracy on
    gaze points the model has not seen. The returned model is fitted on all samples.

    ``nonlinear`` should be the backend's gaze-direction feature indices (see
    ``gaze.model.gaze_feature_indices``): only they get polynomial terms, which
    keeps the model accurate when the user's posture changes. ``None`` expands
    every feature (the behaviour before backends declared them).

    A monitor without any calibration dot (possible with three or more monitors)
    is reported with 0 % accuracy in ``per_monitor_accuracy``, listed in
    ``uncovered_monitors`` and caps the grade at "fair".

    Raises ``ValueError("not enough calibration data")`` with fewer than 10
    samples, fewer than 3 points, or points on fewer than two monitors (one on a
    single-monitor desk).
    """
    monitor_list = list(monitors)
    if not monitor_list:
        raise ValueError("no monitors")
    known = {m.index for m in monitor_list}
    usable = [s for s in samples if s.monitor_index in known]
    if len(usable) < len(samples):
        log.warning("Ignoring %d samples of unknown monitors", len(samples) - len(usable))

    point_ids = place_groups(usable)
    covered = {s.monitor_index for s in usable}
    if (
        len(usable) < MIN_SAMPLES
        or np.unique(point_ids).shape[0] < MIN_POINTS
        or len(covered) < min(2, len(monitor_list))
    ):
        raise ValueError("not enough calibration data")

    X, Y, W = samples_to_arrays(usable)
    bounds = virtual_bounds(monitor_list)
    degrees = SUPPORTED_DEGREES if degree is None else (degree,)
    selection = select_model(
        X,
        Y,
        point_ids,
        alphas=DEFAULT_ALPHAS,
        degrees=degrees,
        bounds=bounds,
        weights=W,
        nonlinear=nonlinear,
    )
    preds = selection.predictions
    if preds is None:  # every candidate diverged; still report honest held-out errors
        preds = lopo_predictions(
            X,
            Y,
            point_ids,
            selection.alpha,
            degree=selection.degree,
            bounds=bounds,
            weights=W,
            nonlinear=nonlinear,
        )

    truth = np.array([s.monitor_index for s in usable])
    predicted = [_predicted_monitor(monitor_list, p) for p in preds]
    correct = np.array([p == t for p, t in zip(predicted, truth.tolist(), strict=True)])
    errors = np.hypot(preds[:, 0] - Y[:, 0], preds[:, 1] - Y[:, 1])
    finite = errors[np.isfinite(errors)]
    poses = np.array([pose_of(s.point_id) for s in usable])
    per_pose = _median_by_pose(errors, poses)
    unseen: dict[int, float] = {}
    if np.unique(poses).shape[0] >= 2:
        held_out = lopo_predictions(
            X,
            Y,
            poses,
            selection.alpha,
            degree=selection.degree,
            bounds=bounds,
            weights=W,
            nonlinear=nonlinear,
        )
        unseen = _median_by_pose(np.hypot(*(held_out - Y).T), poses)

    # A monitor without dots was never calibrated: the model only extrapolates
    # there, and gaze at it is often where the face is lost. An "Excellent"
    # grade computed over the other monitors would hide that.
    uncovered = sorted(known - covered)
    per_monitor = {int(idx): float(np.mean(correct[truth == idx])) for idx in sorted(covered)}
    per_monitor.update({int(idx): 0.0 for idx in uncovered})
    grade = grade_for(float(np.mean(correct)))
    if uncovered and grade in ("excellent", "good"):
        grade = "fair"
    if uncovered:
        log.warning("Calibration has no points on monitor(s) %s", uncovered)

    report = CalibrationReport(
        monitor_accuracy=float(np.mean(correct)),
        mean_error_px=float(np.mean(finite)) if finite.size else math.nan,
        median_error_px=float(np.median(finite)) if finite.size else math.nan,
        per_monitor_accuracy=dict(sorted(per_monitor.items())),
        n_samples=len(usable),
        n_points=int(np.unique(point_ids).shape[0]),
        alpha=selection.alpha,
        grade=grade,
        degree=selection.degree,
        uncovered_monitors=[int(i) for i in uncovered],
        per_monitor_error_px=per_axis_errors(preds, Y, truth.tolist()),
        per_pose_error_px=per_pose,
        unseen_pose_error_px=unseen,
    )
    model = GazeModel(degree=selection.degree, alpha=selection.alpha, nonlinear=nonlinear).fit(
        X, Y, W, bounds, regions=[m.rect for m in monitor_list]
    )
    log.info(
        "Calibration evaluated: %s (degree %d, alpha %g)",
        report.summary(),
        report.degree,
        report.alpha,
    )
    return model, report


def _median_by_pose(errors: np.ndarray, poses: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for pose in np.unique(poses):
        rows = errors[(poses == pose) & np.isfinite(errors)]
        if rows.size:
            out[int(pose)] = float(np.median(rows))
    return out


def _predicted_monitor(monitors: Sequence[Monitor], point: np.ndarray) -> int | None:
    px, py = float(point[0]), float(point[1])
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    return nearest_monitor(monitors, px, py)[0].index


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _float(value: Any) -> float:
    return math.nan if value is None else float(value)


def _json_number(value: Any) -> Any:
    """JSON has no NaN/Infinity; represent them as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
