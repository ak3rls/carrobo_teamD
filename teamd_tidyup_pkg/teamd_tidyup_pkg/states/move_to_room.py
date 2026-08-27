#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部屋間を移動するステート."""

from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State


# 各部屋への経路です。手前から順に通過する目標を並べます。
# 最後の要素がその部屋の到達目標で、それより前は経由点です。
# 物が散らかっている場所を避けたいときは、安全な側を通る経由点を足します。
ROOM_GOALS = {
    # 'roomA': [{'x': 6.960, 'y': -0.950, 'yaw': 0.977}],   #RViz
    'roomA': [
        # {'x': 3.40, 'y': -1.40, 'yaw': 0.54},
        # {'x': 3.80, 'y': -0.79, 'yaw': -0.10},
        {'x': 6.66, 'y': 0.481, 'yaw': 0.0},
    ],
    'roomB': [
        # 経由点 1: TODO 座標を入れてください。
        {'x': 5.1, 'y': 2.89, 'yaw': 2.94},
        # 経由点 2
        {'x': 5.25, 'y': 4.6, 'yaw': 0.977},
        # 到達目標
        # {'x': 6.848, 'y': 4.283, 'yaw': -1.060},
        {'x': 6.8, 'y': 5.2, 'yaw': -1.060},
    ],
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
        """設定された部屋へ、経由点を順に通って移動する."""
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

        waypoints = ROOM_GOALS[self.target_room]
        if not waypoints:
            self.node.get_logger().error(
                f'{self.target_room} の経路が空です。'
            )
            return 'failed'

        # 途中で止まらないよう、動き出す前に全部の座標を確かめます。
        for step, waypoint in enumerate(waypoints, start=1):
            if (
                waypoint['x'] is None
                or waypoint['y'] is None
                or waypoint['yaw'] is None
            ):
                self.node.get_logger().error(
                    f'{self.target_room} の {step} 番目の座標が未設定です。'
                )
                return 'failed'

        total = len(waypoints)
        for step, waypoint in enumerate(waypoints, start=1):
            goal = Pose2D(
                x=float(waypoint['x']),
                y=float(waypoint['y']),
                theta=float(waypoint['yaw']),
            )
            # 最後の目標だけが部屋の到達点で、それ以外は通過点です。
            kind = '目標' if step == total else '経由点'
            self.node.get_logger().info(
                f'Room {self.target_room} {kind} ({step}/{total}): '
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
                    f'部屋間移動の{kind} ({step}/{total}) に'
                    f'失敗しました: {status.message}'
                )
                return 'failed'

        blackboard.current_room = self.target_room

        return 'succeeded'
