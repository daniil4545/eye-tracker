"""Windows as panes: the part of each window the user can see, for window-focus mode.

The window list gives frames front to back. Each window keeps the largest
rectangle of its frame that no window in front covers, so the pieces never
overlap and a gaze point belongs to at most one window. The pieces form a
:class:`PaneSnapshot` for an ordinary :class:`PaneDecider`: the window is the
``Pane`` (``id`` is the window number, ``window_handle`` the monitor index).
Only ids and frames are used, never titles.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import Rect, WindowInfo, WindowRef
from .types import Pane, PaneSnapshot

#: Windows and pieces smaller than this on either side are noise (popups, slivers).
MIN_PIECE_PX = 64

#: Frames of the focused window and the window list agree within this many pixels.
_FRAME_TOLERANCE = 2

PROVIDER = "windows"


def _clip(rect: Rect, bounds: Rect) -> Rect | None:
    x0, y0 = max(rect.x, bounds.x), max(rect.y, bounds.y)
    x1, y1 = min(rect.right, bounds.right), min(rect.bottom, bounds.bottom)
    if x1 <= x0 or y1 <= y0:
        return None
    return Rect(x0, y0, x1 - x0, y1 - y0)


def _subtract(rect: Rect, cover: Rect) -> list[Rect]:
    """``rect`` minus ``cover`` as up to four disjoint rectangles."""
    inner = _clip(rect, cover)
    if inner is None:
        return [rect]
    parts = [
        Rect(rect.x, rect.y, rect.w, inner.y - rect.y),  # above
        Rect(rect.x, inner.bottom, rect.w, rect.bottom - inner.bottom),  # below
        Rect(rect.x, inner.y, inner.x - rect.x, inner.h),  # left
        Rect(inner.right, inner.y, rect.right - inner.right, inner.h),  # right
    ]
    return [p for p in parts if p.w > 0 and p.h > 0]


def visible_pieces(windows: Sequence[WindowInfo], monitor: Rect) -> list[tuple[WindowInfo, Rect]]:
    """The largest visible piece of every window on ``monitor`` (``windows`` front to back).

    A window or piece under :data:`MIN_PIECE_PX` on a side is noise: it neither
    becomes a target nor covers what is behind it.
    """
    covers: list[Rect] = []
    result: list[tuple[WindowInfo, Rect]] = []
    for window in windows:
        frame = _clip(window.rect, monitor)
        if frame is None or frame.w < MIN_PIECE_PX or frame.h < MIN_PIECE_PX:
            continue
        remains = [frame]
        for cover in covers:
            remains = [part for r in remains for part in _subtract(r, cover)]
            if not remains:
                break
        covers.append(frame)
        best = max(remains, key=lambda r: r.w * r.h, default=None)
        if best is not None and best.w >= MIN_PIECE_PX and best.h >= MIN_PIECE_PX:
            result.append((window, best))
    return result


def _frames_match(a: Rect, b: Rect) -> bool:
    return (
        abs(a.x - b.x) <= _FRAME_TOLERANCE
        and abs(a.y - b.y) <= _FRAME_TOLERANCE
        and abs(a.w - b.w) <= _FRAME_TOLERANCE
        and abs(a.h - b.h) <= _FRAME_TOLERANCE
    )


def window_snapshot(
    windows: Sequence[WindowInfo] | None,
    monitor: Rect,
    index: int,
    foreground: WindowRef | None,
    now: float,
) -> PaneSnapshot | None:
    """Snapshot of the windows of monitor ``index``; ``None`` when the list is unavailable.

    The focused window is the front-most one with the foreground window's pid and
    frame; with none (focus is on another monitor) no pane is focused and the
    decider stays silent.
    """
    if windows is None:
        return None
    pieces = visible_pieces(windows, monitor)
    focused_number: int | None = None
    if foreground is not None and foreground.pid is not None and foreground.rect is not None:
        for window, _piece in pieces:
            if window.pid == foreground.pid and _frames_match(window.rect, foreground.rect):
                focused_number = window.number
                break
    panes = tuple(
        Pane(window.number, piece, window.number == focused_number, PROVIDER)
        for window, piece in pieces
    )
    return PaneSnapshot(window_handle=index, panes=panes, taken_at=now)
