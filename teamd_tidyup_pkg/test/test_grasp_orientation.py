"""グリッパが物体の短辺側の面を挟むことを確かめるテスト。"""

import math

import numpy as np
import tf_transformations as tft
from geometry_msgs.msg import Quaternion

from teamd_tidyup_pkg.states.grasp import _hand_yaw
from teamd_tidyup_pkg.states.grasp import _wrist_roll_goal
from teamd_tidyup_pkg.states.recog import _horizontal_long_axis
from teamd_tidyup_pkg.states.recog import _horizontal_short_axis
from teamd_tidyup_pkg.states.recog import _side_grasp_orientation

# 上把持で IK に渡す姿勢。手のひらが真下を向きます。
PALM_DOWN = tft.quaternion_from_euler(math.pi, 0.0, 0.0)
WRIST_ROLL_LIMITS = (-1.92, 3.67)


def _message(values):
    return Quaternion(x=values[0], y=values[1], z=values[2], w=values[3])


def _rotation(message):
    values = [message.x, message.y, message.z, message.w]
    return tft.quaternion_matrix(values)[:3, :3]


def _closing_axis_after_grasp(box_yaw):
    """recog が決めた向きで grasp が手首を回したあとの手先姿勢を返す。

    wrist_roll は手のひらが真下のとき base_link の -Z まわりに効くので、
    関節角を d 増やすとハンドは鉛直まわりに -d 回ります。
    """
    box_orientation = _message(tft.quaternion_from_euler(0.0, 0.0, box_yaw))
    long_axis = _horizontal_long_axis(box_orientation)
    # _hand_yaw は「指の間を通り抜ける向き」なので、90度足すと
    # グリッパの閉じ方向 (ローカル Y) が長辺方向になります。
    target_hand_yaw = math.atan2(long_axis[1], long_axis[0]) + math.pi / 2.0

    palm_down = _message(PALM_DOWN)
    goal = _wrist_roll_goal(
        current_roll=0.0,
        current_hand_yaw=_hand_yaw(palm_down),
        target_hand_yaw=target_hand_yaw,
        limits=WRIST_ROLL_LIMITS,
    )
    hand = tft.euler_matrix(0.0, 0.0, -goal)[:3, :3] @ _rotation(palm_down)
    return hand, long_axis


def test_long_axis_follows_box_x_axis():
    box_yaw = math.radians(35.0)
    long_axis = _horizontal_long_axis(
        _message(tft.quaternion_from_euler(0.0, 0.0, box_yaw)),
    )
    np.testing.assert_allclose(
        long_axis,
        [math.cos(box_yaw), math.sin(box_yaw), 0.0],
        atol=1e-7,
    )


def test_top_grasp_closes_along_long_edge():
    """どの向きに置かれていても、指は長辺方向に閉じて短辺側を挟む。"""
    for degrees in (0.0, 35.0, 90.0, 120.0, -47.0):
        box_yaw = math.radians(degrees)
        hand, long_axis = _closing_axis_after_grasp(box_yaw)

        # 手のひらは真下を向いたままです。
        np.testing.assert_allclose(hand[:, 2], [0.0, 0.0, -1.0], atol=1e-7)
        # 指が閉じる向き (ローカル Y) は長辺と一致します。
        assert abs(abs(float(hand[:, 1] @ long_axis)) - 1.0) < 1e-7, degrees


def test_side_grasp_approaches_from_short_edge_and_closes_along_long_edge():
    """横把持は短辺方向から接近し、長辺方向に閉じる。"""
    box_yaw = math.radians(35.0)
    box_orientation = _message(
        tft.quaternion_from_euler(0.0, 0.0, box_yaw),
    )
    long_axis = _horizontal_long_axis(box_orientation)
    short_axis = _horizontal_short_axis(box_orientation)
    orientation, approach_axis = _side_grasp_orientation(short_axis[:2])
    hand = _rotation(orientation)

    assert abs(abs(float(approach_axis @ short_axis)) - 1.0) < 1e-7
    assert abs(abs(float(hand[:, 1] @ long_axis)) - 1.0) < 1e-7
