"""ロボット視点のじゃんけんの勝敗を判定するステート."""

from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State
from teamd_janken_interfaces.srv import hand_recog


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
        super().__init__(outcomes=['win', 'lose', 'failed'])
        self.node = node

        self.cli = self.create_client(hand_recog, "hand_recog_jg")
        while not self.cli.wait_for_service(timeout_sec = 1.0):
            self.get_logger().info("サーバーがひらくのを待ってます")
        self.req = hand_recog.Request()

    def send_request_hand(self, kara: int):
        self.req. = int(kara)
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def execute(self, blackboard: Blackboard) -> str:
        """ロボット視点の勝敗を返す."""
        self.node.get_logger().info('Executing state JankenState')
        
        
        robot_hand = getattr(blackboard, 'robot_hand', None)
        human_hand = getattr(blackboard, 'human_hand', None)



        if robot_hand not in self.VALID_HANDS:
            self.node.get_logger().error(
                f'ロボットの手が不正です: {robot_hand}'
            )
            return 'failed'

        if human_hand not in self.VALID_HANDS:
            self.node.get_logger().error(
                f'人間の手が不正です: {human_hand}'
            )
            return 'failed'



        if robot_hand == human_hand:
            result = 'draw'
        elif (robot_hand, human_hand,) in self.ROBOT_WIN_PAIRS:
            result = 'win'
        else:
            result = 'lose'

        blackboard.janken_result = result

        # self.node.get_logger().info(
        #     f'勝敗判定: robot={robot_hand}, '
        #     f'human={human_hand}, result={result}'
        # )

        return result