#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把持している物体を所定位置へ配置するステート."""

import time

from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State

from carrobo_manipulation_pkg.hsrif import HSRInterfaces


# ---------------------------------------------------------------------------
# 配置場所でハンドを開く前に遷移する関節姿勢です。
# /joint_states で取得した値のうち、whole_body で指令する7関節を使います。
PLACE_JOINT_POSITIONS = {
    'arm_lift_joint': 0.013891496695578098,
    'arm_flex_joint': -0.4696487784385681,
    'arm_roll_joint': 0.004460785537958145,
    'wrist_flex_joint': -1.12323796749115,
    'wrist_roll_joint': -0.32181528210639954,
    'head_pan_joint': -0.08123230189085007,
    'head_tilt_joint': -0.09592577069997787,
}

# 配置姿勢になった後、ハンドを開く前に前進する距離です。
PLACE_FORWARD_DISTANCE = 0.20  # [m]

# ハンドを開いた後、そのまま静止して待つ時間です。
# すぐ動くと物体が指から離れきらず、一緒に持っていってしまいます。
RELEASE_HOLD_TIME = 5.0  # [s]
# ---------------------------------------------------------------------------


class PlaceState(State):
    """指定姿勢へ遷移し、前進してからハンドを開く."""

    def __init__(self, node: Node, hsrif: HSRInterfaces):
        """ステートを初期化する."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.hsrif = hsrif

    def execute(self, blackboard: Blackboard) -> str:
        """配置姿勢へ遷移し、ハンドを開いてから静止して物体を離す."""
        self.node.get_logger().info('Executing state Place')

        try:
            self.hsrif.whole_body.move_to_joint_positions(
                PLACE_JOINT_POSITIONS,
                sync=True,
            )
            self.hsrif.gripper.command(1.0, sync=True)

            # 開いた直後に動くと、物体が指に引っかかったまま運ばれます。
            # 開いた姿勢のまま静止して、確実に落としてから次へ進みます。
            self.node.get_logger().info(
                f'ハンドを開いたまま {RELEASE_HOLD_TIME:.0f} 秒待ちます。'
            )
            time.sleep(RELEASE_HOLD_TIME)
        # hsrb_interface の動作例外をステート失敗へ変換します。
        except Exception as error:
            self.node.get_logger().error(f'配置動作に失敗しました: {error}')
            return 'failed'

        self.node.get_logger().info('物体の配置が完了しました。')
        return 'succeeded'
