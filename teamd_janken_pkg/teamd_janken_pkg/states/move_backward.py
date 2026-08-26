#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ロボットを1 m後退させるステート."""

from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State


class MoveForwardState(State):
    """ロボットを1 m後退させるステート."""

    def __init__(self, node: Node, nav: NavModule):
        """ステートを初期化する."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.nav = nav

    def execute(self, blackboard: Blackboard) -> str:
        """現在向いている方向から1 m後ろへ下がる."""
        self.node.get_logger().info('1 m後退します。')
        
        succeeded = self.nav.go_rel(x = -1.0, timeout = 30.0)

        if succeeded:
            self.node.get_logger().info('1 m後退しました。')
            return 'succeeded'

        self.node.get_logger().error('1 m後退できませんでした。')
        return 'failed'