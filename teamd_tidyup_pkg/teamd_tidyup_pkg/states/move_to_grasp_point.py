#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物体を把持する場所まで移動するステート."""

from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State



# 把持場所の map 座標
GOAL_X = 6.9
GOAL_Y = 0.56
GOAL_YAW = 0.0

# 0.0 は到着するまで待ち続けます。必要なら秒数を指定してください。
NAVIGATION_TIMEOUT = 0.0


class Move2GraspPointState(State):
    """把持場所へのナビゲーションを実行するステート."""

    def __init__(self, node: Node, nav: NavModule):
        """ステートを初期化する."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.nav = nav

    def execute(self, blackboard: Blackboard) -> str:
        """設定された把持場所へ移動する."""
        self.node.get_logger().info('Executing state Move2GraspPoint')

        if GOAL_X is None or GOAL_Y is None or GOAL_YAW is None:
            self.node.get_logger().error(
                '把持場所が未設定です。move_to_grasp_point.py の '
                'GOAL_X / GOAL_Y / GOAL_YAW を設定してください。'
            )
            return 'failed'

        goal = Pose2D(x=float(GOAL_X), y=float(GOAL_Y), theta=float(GOAL_YAW))
        self.node.get_logger().info(
            f'Navigation goal: x={goal.x:.2f}, y={goal.y:.2f}, '
            f'yaw={goal.theta:.2f}'
        )

        if self.nav.nav_goal(goal=goal, timeout=NAVIGATION_TIMEOUT):
            return 'succeeded'

        status = self.nav.nav_status
        self.node.get_logger().error(
            f'把持場所への移動に失敗しました: {status.message}'
        )
        return 'failed'
