"""手認識クラスを呼び出して、人間の手を取得するステート."""

from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State


class RecognizeHumanHandState(State):
    """手認識クラスから人間の手を取得するステート."""

    # 数字と文字列の両方に対応します。
    HAND_ALIASES = {
        0: 'rock', 1: 'scissors', 2: 'paper',

        '0': 'rock', '1': 'scissors', '2': 'paper',

        'rock': 'rock', 'グー': 'rock', 'ぐー': 'rock',
        'scissors': 'scissors', 'scissor': 'scissors', 'チョキ': 'scissors', 'ちょき': 'scissors',
        'paper': 'paper', 'パー': 'paper', 'ぱー': 'paper',
    }


    def __init__(self, node: Node, hand_recognizer):
        super().__init__(outcomes=['succeeded', 'failed'])

        self.node = node
        self.hand_recognizer = hand_recognizer

    def _normalize_hand(self, value):
        """数字または文字列を共通表現へ変換する."""
        if isinstance(value, str):
            value = value.strip().lower()

        return self.HAND_ALIASES.get(value)



    def execute(self, blackboard: Blackboard) -> str:
        """認識クラスを呼び出し、結果をBlackboardへ保存する."""
        self.node.get_logger().info('Executing state RecognizeHumanHandState')
        try:
            raw_result = self.hand_recognizer.recognize()
        except Exception as error:
            self.node.get_logger().error(f'手認識処理で例外が発生しました: {error}')
            return 'failed'

        self.node.get_logger().info(f'手認識クラスからの戻り値: {raw_result}')

        human_hand = self._normalize_hand(raw_result)

        if human_hand is None:
            self.node.get_logger().error(f'認識結果が不正です: {raw_result}')
            return 'failed'

        blackboard.human_hand = human_hand

        # self.node.get_logger().info(f'人間の手: {human_hand}')

        return 'succeeded'