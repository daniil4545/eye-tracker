# Window focus (experimental, macOS)

Eye Tracker moves keyboard focus between monitors. With window focus on, it also moves focus
between the windows on the monitor you are already on: look at a window for a moment and it
gets the keyboard focus and comes forward. This fork adds it; it is off by default.

Turn it on under **Settings → Window focus (experimental)**, or with **Follow windows** in the tray
menu. It needs macOS, **Accessibility** permission and the monitor switching to be on. A
calibration is needed too: without one, window focus stays silent. In the settings the option is
marked "not supported on this system" elsewhere.

## How it decides

The window you look at is chosen with the same cautious rules as [split-pane focus](panes.md#how-it-decides),
measured against how accurate your calibration is.

1. **Visible parts.** Each window counts only by the largest rectangle of it that no window in
   front covers, so the pieces never overlap. Windows (or pieces) under 64 px on a side are
   ignored. Only ordinary windows are listed, and only their number, owner and frame are read,
   never titles.
2. **Same monitor only.** Window focus works on the monitor the cursor is on, once the monitor
   switch has settled (`windows.after_monitor_switch_ms`, 1.5 s after a switch).
3. **Size gate.** A window takes part only if its visible part is at least `windows.precision`
   (2.5) times your gaze error wide (windows side by side) or tall (stacked windows), and never
   under `windows.min_window_px` (240 px). The gaze error is the one the calibration measured on
   the dots, per monitor. Smaller windows are left alone.
4. **Past the border.** The gaze must be `windows.hysteresis` (half the gaze error) past the border
   between two windows.
5. **A steady look.** You must look at the other window for `windows.dwell_ms` (0.5 s).
6. **Not while you work.** No switch for `windows.typing_grace_ms` (3 s) after you type or switch
   windows yourself, for `windows.cooldown_ms` (1 s) after the previous window switch, or for the
   mouse grace of the switching settings after you use the mouse. Typing while looking at
   another window counts as reading it: it gets the focus only `windows.reading_grace_ms` (8 s)
   after your last keystroke.

Before the window gets the focus, Eye Tracker checks again that it is still where you look. The
cursor does not move.

## Pause while the head is off range

Window focus pauses while your head is farther from the range you calibrated in than
`windows.pause_off_range` (0.25) allows. The distance is the largest of the roll, side, up and
down, and distance offsets, as a fraction of the calibrated range. Window focus resumes when you
move back. Set it to 0 to turn the pause off. With the **lite** backend (no head position) the
pause never applies. A window is much smaller than a monitor, so a head that has left the
calibrated range makes the gaze estimate too unreliable to pick one.

## Only the target window comes forward

On macOS, bringing an app forward normally raises all its windows. Eye Tracker raises the target
window and asks the system to bring the app forward with its front window only, as a click does,
so your other windows of that app stay behind.

## Improving accuracy

The [moving-dot stage](calibration.md#the-moving-dot) of the calibration gives window focus
more samples, with head movement. Recalibrate if windows are picked wrongly, and see
[troubleshooting](troubleshooting.md#window-focus).
