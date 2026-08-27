#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ターミナルでゲームを続けるか確認するステート."""

import unicodedata

import rclpy
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State


class ContinueState(State):
    """ターミナル入力からゲームを続けるか判定するステート."""

    YES_INPUTS = {
        'yes',
        'y',
        'はい',
        'もう一度',
        'もう一回',
        '続ける',
        'やる',
    }

    NO_INPUTS = {
        'no',
        'n',
        'いいえ',
        '終わり',
        '終了',
        'やめる',
        'q',
        'quit',
    }

    def __init__(self, node: Node, max_attempts: int = 3): # max_attemps 試行回数
        super().__init__(outcomes=['yes', 'no', 'failed'])
        self.node = node
        self.max_attempts = max_attempts

    @staticmethod
    def _normalize_text(text: str) -> str:
        """入力文字列の空白や表記揺れを整理する."""
        text = unicodedata.normalize('NFKC', text)
        return text.strip().lower()

    def execute(self, blackboard: Blackboard) -> str:
        """ターミナルから回答を取得する."""
        self.node.get_logger().info('Executing state ContinueState')
        self.node.get_logger().info(
            'もう一度じゃんけんをするか確認します。'
        )

        for attempt in range(1, self.max_attempts + 1):
            if self.is_canceled() or not rclpy.ok():
                self.node.get_logger().warning(
                    'ContinueStateが中断されました。'
                )
                return 'failed'

            try:
                answer_text = input(
                    '\n'
                    'もう一度じゃんけんをしますか？\n'
                    '  yes / y / はい : 続ける\n'
                    '  no  / n / いいえ : 終了する\n'
                    '> '
                )
            except (EOFError, KeyboardInterrupt):
                print()
                self.node.get_logger().warning('ターミナル入力が中断されました。')
                return 'failed'

            normalized_answer = self._normalize_text(answer_text)

            if normalized_answer in self.YES_INPUTS:
                blackboard.continue_answer = 'yes'
                self.node.get_logger().info('もう一度じゃんけんをします。')
                return 'yes'

            if normalized_answer in self.NO_INPUTS:
                blackboard.continue_answer = 'no'
                self.node.get_logger().info('じゃんけんを終了します。')
                return 'no'

            self.node.get_logger().warning(
                f'入力を判定できませんでした: '
                f'"{answer_text}" '
                f'({attempt}/{self.max_attempts})'
            )

        self.node.get_logger().error(
            '入力回数の上限を超えました。'
        )
        return 'failed'