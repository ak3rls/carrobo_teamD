"""ロボットが出す手を決定して表示させるステート."""

import random

from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State

from teamd_janken_interfaces.srv import ShowHand

from .service_utils import call_service


class ShowRobotHandState(State):
    """ロボットの手を決め、表示ノードへ要求する."""

    HANDS = ('rock', 'scissors', 'paper')

    def __init__(self, node: Node, service_name: str = '/janken/show_robot_hand',):
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.service_name = service_name
        self.client = node.create_client(ShowHand, service_name,)


    def execute(self, blackboard: Blackboard) -> str:
        """ロボットの手をランダムに決めて表示する."""
        robot_hand = random.choice(self.HANDS)
        blackboard.robot_hand = robot_hand

        self.node.get_logger().info('Executing state RecognizeHumanHandState')

        request = ShowHand.Request()
        request.hand = robot_hand

        response = call_service(
            self.node,
            self.client,
            request,
            self.service_name,
        )

        if response is None:
            return 'failed'

        if not response.success:
            self.node.get_logger().error(
                f'ロボットの手を表示できませんでした: '
                f'{response.message}'
            )
            return 'failed'

        return 'succeeded'