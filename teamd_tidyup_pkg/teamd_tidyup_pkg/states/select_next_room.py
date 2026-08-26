#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""認識対象がなくなった後の遷移先を決定するステート."""

from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State


class SelectNextRoomState(State):
    """現在の部屋に応じて次の部屋への移動またはタスク終了を選択する."""

    def __init__(self, node: Node):
        """ステートを初期化する."""
        super().__init__(outcomes=['move_to_room_b', 'finished', 'failed'])
        self.node = node

    def execute(self, blackboard: Blackboard) -> str:
        """roomA の後は roomB へ進み、roomB の後はタスクを終了する."""
        if 'current_room' not in blackboard:
            self.node.get_logger().error(
                'Blackboardにcurrent_roomが設定されていません。'
            )
            return 'failed'

        if blackboard.current_room == 'roomA':
            return 'move_to_room_b'
        if blackboard.current_room == 'roomB':
            return 'finished'

        self.node.get_logger().error(
            f'未対応の部屋です: {blackboard.current_room}'
        )
        return 'failed'
