"""Tests for PCA-box-aligned object grasp orientations."""

import math

import numpy as np
import tf_transformations as tft
from geometry_msgs.msg import Quaternion

from teamd_tidyup_pkg.states.recog import _box_aligned_grasp_orientation


def _message(values):
    return Quaternion(x=values[0], y=values[1], z=values[2], w=values[3])


def _rotation(message):
    values = [message.x, message.y, message.z, message.w]
    return tft.quaternion_matrix(values)[:3, :3]


def test_top_grasp_closes_along_short_edge():
    box_yaw = math.radians(35.0)
    box_rotation = tft.euler_matrix(0.0, 0.0, box_yaw)[:3, :3]
    orientation, _ = _box_aligned_grasp_orientation(
        _message(tft.quaternion_from_euler(0.0, 0.0, box_yaw)),
        side_grasp=False,
        object_xy=(1.0, 0.0),
    )
    hand_rotation = _rotation(orientation)

    np.testing.assert_allclose(
        np.abs(hand_rotation[:, 1]),
        np.abs(box_rotation[:, 1]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        hand_rotation[:, 2],
        [0.0, 0.0, -1.0],
        atol=1e-7,
    )


def test_side_grasp_approaches_along_long_edge():
    box_yaw = math.radians(35.0)
    box_rotation = tft.euler_matrix(0.0, 0.0, box_yaw)[:3, :3]
    long_axis = box_rotation[:, 0]
    orientation, approach_axis = _box_aligned_grasp_orientation(
        _message(tft.quaternion_from_euler(0.0, 0.0, box_yaw)),
        side_grasp=True,
        object_xy=long_axis[:2],
    )
    hand_rotation = _rotation(orientation)

    np.testing.assert_allclose(approach_axis, long_axis, atol=1e-7)
    np.testing.assert_allclose(
        np.abs(hand_rotation[:, 1]),
        np.abs(box_rotation[:, 1]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        hand_rotation[:, 2],
        long_axis,
        atol=1e-7,
    )


def test_side_grasp_selects_robot_facing_end():
    orientation, approach_axis = _box_aligned_grasp_orientation(
        _message(tft.quaternion_from_euler(0.0, 0.0, 0.0)),
        side_grasp=True,
        object_xy=(-1.0, 0.0),
    )

    np.testing.assert_allclose(approach_axis, [-1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(
        _rotation(orientation)[:, 2],
        [-1.0, 0.0, 0.0],
        atol=1e-7,
    )
