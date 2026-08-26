#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""現在いる部屋のBox前まで移動するステート."""

from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State


# Box手前の配置作業姿勢
BOX_GOALS = {
    'roomA': {'x': 7.351,'y':-1.579,'yaw':-1.665},
    'roomB': {'x': 7.323,'y': 5.046,'yaw': -0.053},
}

# 0.0 は到着するまで待ち続けます。必要なら秒数を指定してください。
NAVIGATION_TIMEOUT = 0.0


class Move2BoxState(State):
    """current_roomのBox前へ移動するステート."""

    def __init__(self, node: Node, nav: NavModule):
        """ステートを初期化する."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.nav = nav

    def execute(self, blackboard: Blackboard) -> str:
        """現在いる部屋のBox前へ移動する."""
        self.node.get_logger().info('Executing state Move2Box')

    
        if 'current_room' not in blackboard:
            self.node.get_logger().error(
                'Blackboardにcurrent_roomが設定されていません。'
            )
            return 'failed'

        current_room = blackboard.current_room

        # roomFには片付け先のBoxがない
        # roomAまたはroomB以外では失敗
        if current_room not in BOX_GOALS:
            self.node.get_logger().error(
                f'{current_room} のBox移動目標がありません。'
            )
            return 'failed'

        box_goal = BOX_GOALS[current_room]

        # 座標の設定漏れ確認
        if box_goal['x'] is None or box_goal['y'] is None or box_goal['yaw'] is None:
            self.node.get_logger().error(f'{current_room} のBox座標が未設定です。')
            return 'failed'

        goal = Pose2D(x=float(box_goal['x']), y=float(box_goal['y']), theta=float(box_goal['yaw']))

        self.node.get_logger().info(
            f'{current_room} Box goal: '
            f'x={goal.x:.2f}, '
            f'y={goal.y:.2f}, '
            f'yaw={goal.theta:.2f}'
        )

        succeeded = self.nav.nav_goal(
            goal=goal,
            timeout=NAVIGATION_TIMEOUT,
        )

        if not succeeded:
            status = self.nav.nav_status
            self.node.get_logger().error(
                f'{current_room} のBox前への移動に失敗しました: '
                f'{status.message}'
            )
            return 'failed'

        self.node.get_logger().info(
            f'{current_room} のBox前へ到着しました。'
        )

        return 'succeeded'



