"""ロボット視点のじゃんけんの勝敗を判定するステート."""

from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State
import random
from teamd_janken_pkg.nodes.client_node import JankenClient



class JankenState(State):
    """Blackboard上のロボットの手と人間の手を比較して勝敗を返すステート."""

    VALID_HANDS = {'rock', 'scissors', 'paper'}

    # ロボットが勝つパターン(ロボットの手, 人間の手)
    ROBOT_WIN_PAIRS = {
        ('rock', 'scissors'),
        ('scissors', 'paper'),
        ('paper', 'rock'),
    }

    def __init__(self, node: Node):
        super().__init__(outcomes=['win', 'lose', 'draw'])
        self.node = node
        self.client = JankenClient()


    def execute(self, blackboard: Blackboard) -> str:
        """ロボット視点の勝敗を返す."""
        self.node.get_logger().info('Executing state JankenState')

        robot_hand_rundom = random.choice(['rock','scissors','paper'])
        
        
        robot_hand = self.client.send_robot_request(robot_hand_rundom)
        human_hand = self.client.send_hand_request(1)



        # if robot_hand not in self.VALID_HANDS:
        #     self.node.get_logger().error(
        #         f'ロボットの手が不正です: {robot_hand}'
        #     )
        #     return 'failed'

        # if human_hand not in self.VALID_HANDS:
        #     self.node.get_logger().error(
        #         f'人間の手が不正です: {human_hand}'
        #     )
        #     return 'failed'



        if robot_hand_rundom == human_hand:
            result = 'draw'
        elif (robot_hand_rundom, human_hand,) in self.ROBOT_WIN_PAIRS:
            result = 'win'
        else:
            result = 'lose'

        return result