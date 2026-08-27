"""Tests for robust drawer-knob detection fusion."""

import pytest

from teamd_tidyup_pkg.states.drawer_open import DrawerOpenTask
from teamd_tidyup_pkg.states.drawer_open import Knob


class _Logger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


def _task_with_detections(detections):
    """Build only the collaborators needed by ``_detect_knobs``."""
    task = DrawerOpenTask.__new__(DrawerOpenTask)
    values = {
        'same_knob_radius': 0.12,
        'knob_search_tilts': [-0.35],
        'knob_detection_samples': 3,
        'knob_min_detection_votes': 2,
        'knob_detection_sample_interval': 0.0,
    }
    task._value = values.__getitem__
    task.logger = _Logger()
    task._look = lambda _tilt: None
    task._spin = lambda _seconds: None
    task._base_pose = lambda: (0.0, 0.0, 0.0)
    task._to_odom = lambda knob: _anchor(knob)
    detection_iterator = iter(detections)
    task._detect_once = lambda: next(detection_iterator)
    return task


def _anchor(knob):
    """Use an identity base-to-odom conversion for the focused unit test."""
    knob.odom = (knob.x, knob.y, knob.z)
    return knob


def test_detection_fusion_uses_median_and_rejects_single_frame_candidate():
    """Keep the stable knob, even when a false candidate appears once."""
    task = _task_with_detections([
        [Knob(0.70, 0.08, 0.30, 0.3), Knob(0.40, -0.35, 0.60, 0.9)],
        [Knob(0.71, 0.10, 0.31, 0.4)],
        [Knob(0.69, 0.09, 0.29, 0.5)],
    ])

    knobs = task._detect_knobs()

    assert len(knobs) == 1
    assert knobs[0].x == pytest.approx(0.70)
    assert knobs[0].y == pytest.approx(0.09)
    assert knobs[0].z == pytest.approx(0.30)
    assert knobs[0].score == pytest.approx(0.5)
