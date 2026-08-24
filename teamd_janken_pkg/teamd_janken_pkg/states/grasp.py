#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blackboard に保存された姿勢で物体を把持するステート."""

from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State

from carrobo_manipulation_pkg.hsrif import HSRInterfaces


class GraspState(State):
    """既存の carrobo_manipulation_pkg と同じ手順で把持する."""

    def __init__(self, node: Node, hsrif: HSRInterfaces):
        """ステートを初期化する."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.hsrif = hsrif

    def execute(self, blackboard: Blackboard) -> str:
        """把持前姿勢へ移動し、物体を掴んで走行姿勢へ戻る."""
        self.node.get_logger().info('Executing state Grasp')

        if 'grasp_pose' not in blackboard or 'grasp_approach' not in blackboard:
            self.node.get_logger().error('Blackboard に把持姿勢がありません。')
            return 'failed'

        try:
            self.hsrif.gripper.command(1.0)
            self.hsrif.whole_body.move_end_effector_pose(
                blackboard.grasp_pose,
                'base_link',
                sync=True,
            )
            self.hsrif.whole_body.move_end_effector_by_line(
                (0, 0, 1),
                blackboard.grasp_approach,
                sync=True,
            )
            self.hsrif.gripper.command(-0.1, 1.0)
            self.hsrif.whole_body.move_to_go(sync=True)
        # hsrb_interface の動作例外をステート失敗へ変換します。
        except Exception as error:
            self.node.get_logger().error(f'把持動作に失敗しました: {error}')
            return 'failed'

        self.node.get_logger().info(f'{blackboard.target_name} を把持しました。')
        return 'succeeded'
