#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部屋間を移動するステート."""

from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State



ROOM_GOALS = {
    'roomA':{'x': 6.960, 'y':-0.950, 'yaw': 0.977},   #RViz
    'roomB':{'x': 6.848, 'y': 4.283, 'yaw':-1.060}
}

# 0.0 は到着するまで待ち続けます。必要なら秒数を指定してください。
NAVIGATION_TIMEOUT = 0.0



class Move2RoomState(State):
    """現在の部屋から次の部屋へのナビゲーションを実行するステート."""
    #frontroom > roomA > roomB

    def __init__(self, node: Node, nav: NavModule, source_room: str, target_room: str):
        """ステートを初期化する."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.nav = nav
        self.source_room = source_room
        self.target_room = target_room

    def execute(self, blackboard: Blackboard) -> str:
        """設定された部屋へ移動する."""
        self.node.get_logger().info(f'Executing state Move2Room {self.source_room}>{self.target_room}')

        # if 'current_room' not in blackboard:
        #     self.node.get_logger().error(
        #         'Blackboardにcurrent_roomが設定されていません。'
        #     )
        #     return 'failed

        if blackboard.current_room != self.source_room:
            self.node.get_logger().error(
                f'現在の部屋が一致しません: '
                f'expected={self.source_room}, '
                f'actual={blackboard.current_room}'
            )
            return 'failed'

        if self.target_room not in ROOM_GOALS:
            self.node.get_logger().error(
                f'{self.target_room} の移動目標がありません。'
            )
            return 'failed'

        room_goal = ROOM_GOALS[self.target_room]

        if room_goal['x'] is None or room_goal['y'] is None or room_goal['yaw'] is None:
            self.node.get_logger().error(f'{self.target_room} の座標が未設定です。')
            return 'failed'

        goal = Pose2D(x=float(room_goal['x']), y=float(room_goal['y']), theta=float(room_goal['yaw']))

        self.node.get_logger().info(
            f'Room {self.target_room} goal: '
            f'x={goal.x:.2f}, y={goal.y:.2f}, '
            f'yaw={goal.theta:.2f}'
        )

        succeeded = self.nav.nav_goal(
            goal=goal,
            timeout=NAVIGATION_TIMEOUT,
        )

        if not succeeded:
            status = self.nav.nav_status
            self.node.get_logger().error(
                f'部屋間移動に失敗しました: {status.message}'
            )
            return 'failed'

        blackboard.current_room = self.target_room

        return 'succeeded'


