"""Window pieces and snapshots for the window-focus mode (pure logic, no platform)."""

from __future__ import annotations

from eye_tracker.panes.decider import PaneConfig, eligible_panes, pane_qualifies
from eye_tracker.panes.windows import visible_pieces, window_snapshot
from eye_tracker.types import Rect, WindowInfo, WindowRef

MONITOR = Rect(0, 0, 1920, 1080)


def win(number: int, x: int, y: int, w: int, h: int, pid: int = 100) -> WindowInfo:
    return WindowInfo(number=number, pid=pid, rect=Rect(x, y, w, h))


def pieces_by_number(windows: list[WindowInfo], monitor: Rect = MONITOR) -> dict[int, Rect]:
    return {w.number: rect for w, rect in visible_pieces(windows, monitor)}


def test_visible_piece_of_covered_window() -> None:
    front = win(2, 0, 0, 1000, 1080)
    back = win(1, 0, 0, 1920, 1080)
    pieces = pieces_by_number([front, back])
    assert pieces[2] == Rect(0, 0, 1000, 1080)
    assert pieces[1] == Rect(1000, 0, 920, 1080)  # the largest uncovered remainder


def test_single_fullscreen_window_keeps_whole_monitor() -> None:
    assert pieces_by_number([win(1, 0, 0, 1920, 1080)]) == {1: MONITOR}


def test_pieces_do_not_overlap_and_noise_is_dropped() -> None:
    windows = [
        win(3, 500, 300, 400, 400),
        win(4, 10, 10, 30, 30),  # smaller than 64 px: noise, covers nothing
        win(2, 0, 0, 800, 600),
        win(1, 0, 0, 1920, 1080),
    ]
    pieces = pieces_by_number(windows)
    assert 4 not in pieces
    rects = list(pieces.values())
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert a.right <= b.x or b.right <= a.x or a.bottom <= b.y or b.bottom <= a.y
    assert pieces[1].w >= 64
    assert pieces[1].h >= 64


def test_fully_covered_window_has_no_piece_and_clipped_to_monitor() -> None:
    windows = [win(2, -100, -100, 2200, 1300), win(1, 200, 200, 300, 300)]
    pieces = pieces_by_number(windows)
    assert pieces == {2: MONITOR}


def test_sliver_piece_is_noise() -> None:
    windows = [win(2, 0, 0, 1900, 1080), win(1, 0, 0, 1920, 1080)]
    assert pieces_by_number(windows) == {2: Rect(0, 0, 1900, 1080)}  # 20 px strip dropped


def test_focused_matched_by_pid_and_frame() -> None:
    windows = [win(2, 0, 0, 900, 1080, pid=7), win(1, 900, 0, 1020, 1080, pid=8)]
    fg = WindowRef(handle=(8, None), pid=8, rect=Rect(901, 1, 1019, 1079))  # within 2 px
    snap = window_snapshot(windows, MONITOR, 1, fg, 5.0)
    assert snap is not None
    assert snap.window_handle == 1
    assert snap.taken_at == 5.0
    assert snap.focused is not None
    assert snap.focused.id == 1
    assert {p.id for p in snap.panes} == {1, 2}
    other = WindowRef(handle=(8, None), pid=8, rect=Rect(300, 300, 100, 100))
    snap = window_snapshot(windows, MONITOR, 1, other, 5.0)
    assert snap is not None
    assert snap.focused is None  # frame differs: another monitor


def test_snapshot_without_foreground_or_list() -> None:
    assert window_snapshot(None, MONITOR, 0, None, 1.0) is None
    snap = window_snapshot([win(1, 0, 0, 800, 600)], MONITOR, 0, None, 1.0)
    assert snap is not None
    assert snap.focused is None


def _quarters(monitor: Rect) -> tuple[WindowInfo, WindowInfo]:
    half_w, half_h = monitor.w // 2, monitor.h // 2
    top = WindowInfo(1, 1, Rect(monitor.x, monitor.y, half_w, half_h))
    below = WindowInfo(2, 1, Rect(monitor.x, monitor.y + half_h, half_w, half_h))
    return top, below


def test_mi_quarter_qualifies_at_268() -> None:
    cfg = PaneConfig(precision=2.5, min_pane_px=240.0)
    mi = Rect(0, 0, 3440, 1440)
    a, b = _quarters(mi)  # 1720x720, stacked: y extent 720 >= 2.5 * 268 = 670
    snap = window_snapshot([a, b], mi, 0, WindowRef((1, None), 1, a.rect), 0.0)
    assert snap is not None
    assert eligible_panes(snap, (268.0, 268.0), cfg) == 1


def test_macbook_quarter_stacked_needs_2_0() -> None:
    mac = Rect(0, 0, 1710, 1112)
    a, b = _quarters(mac)  # 855x556, stacked: 556 < 670
    snap = window_snapshot([a, b], mac, 0, WindowRef((1, None), 1, a.rect), 0.0)
    assert snap is not None
    assert eligible_panes(snap, (268.0, 268.0), PaneConfig(precision=2.5)) == 0
    assert eligible_panes(snap, (268.0, 268.0), PaneConfig(precision=2.0)) == 1  # 536
    pane = snap.pane(2)
    assert pane is not None
    assert snap.focused is not None
    # per-axis sigma (P75): y 223 gives 557.5 > 556 at 2.5, x 342 does not matter for stacking
    assert not pane_qualifies(pane, snap.focused, (342.0, 223.0), PaneConfig(precision=2.5))
