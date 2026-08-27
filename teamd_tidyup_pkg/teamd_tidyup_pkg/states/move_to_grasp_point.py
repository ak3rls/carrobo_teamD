#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物体を把持する場所まで移動するステート."""

from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State


# 把持場所の map 座標
GRASP_POINT_GOALS = {
    'roomA':{'x': 6.66, 'y':0.481, 'yaw': 0.0},  
    'roomB': {'x': 6.8, 'y':  5.2, 'yaw': -1.060},
}

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
        if 'current_room' not in blackboard:
            self.node.get_logger().error(
                'Blackboardにcurrent_roomが設定されていません。'
            )
            return 'failed'

        current_room = blackboard.current_room

        if current_room not in GRASP_POINT_GOALS:
            self.node.get_logger().error(
                f'{current_room} の把持場所が設定されていません。'
            )
            return 'failed'

        grasp_goal = GRASP_POINT_GOALS[current_room]
        if grasp_goal['x'] is None or grasp_goal['y'] is None or grasp_goal['yaw'] is None:
            self.node.get_logger().error(
                f'{current_room} の把持場所座標が未設定です。'
            )
            return 'failed'

        goal = Pose2D(
            x=float(grasp_goal['x']),
            y=float(grasp_goal['y']),
            theta=float(grasp_goal['yaw']),
        )

        self.node.get_logger().info(
            f'{current_room} grasp goal: '
            f'x={goal.x:.3f}, y={goal.y:.3f}, '
            f'yaw={goal.theta:.3f}'
        )

        if self.nav.nav_goal(goal=goal, timeout=NAVIGATION_TIMEOUT):
            return 'succeeded'

        status = self.nav.nav_status
        self.node.get_logger().error(
            f'把持場所への移動に失敗しました: {status.message}'
        )
        return 'failed'
