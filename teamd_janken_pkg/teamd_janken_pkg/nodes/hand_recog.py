#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web カメラ画像から手のランドマークを検出するノード."""

from pathlib import Path
import site
import sys


def _prefer_mediapipe_user_site() -> None:
    """このプロセスだけMediaPipe互換のユーザー環境を優先する."""
    user_site = site.getusersitepackages()
    user_site_path = Path(user_site)
    if not (
        (user_site_path / 'mediapipe').is_dir()
        and (user_site_path / 'google' / 'protobuf').is_dir()
    ):
        return

    # ROS 2コンテナのprotobufを変更せず、MediaPipeと同じ場所に導入した
    # 互換版protobufを、このノードのプロセス内だけで使用します。
    if user_site in sys.path:
        sys.path.remove(user_site)
    sys.path.insert(0, user_site)


_prefer_mediapipe_user_site()

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402


WINDOW_NAME = 'Hand recognition'


class HandRecogNode(Node):
    """検出した手のランドマークをカメラ画像に描画する."""

    def __init__(self):
        """ROS パラメータ、MediaPipe、カメラを初期化する."""
        super().__init__('hand_recog')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('show_image', True)
        self.declare_parameter('min_detection_confidence', 0.7)
        self.declare_parameter('min_tracking_confidence', 0.7)

        camera_index = self.get_parameter('camera_index').value
        self.show_image = self.get_parameter('show_image').value
        detection_confidence = self.get_parameter(
            'min_detection_confidence'
        ).value
        tracking_confidence = self.get_parameter(
            'min_tracking_confidence'
        ).value

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=float(detection_confidence),
            min_tracking_confidence=float(tracking_confidence),
        )

        self.cap = cv2.VideoCapture(int(camera_index))
        if not self.cap.isOpened():
            self.cap.release()
            self.hands.close()
            raise RuntimeError(
                'カメラを開けませんでした: '
                f'camera_index={camera_index}'
            )

        self.get_logger().info(
            f'カメラを起動しました: camera_index={camera_index}'
        )
        if self.show_image:
            self.get_logger().info(
                '画像ウィンドウで q を押すと終了します。'
            )

    def process_frame(self) -> bool:
        """画像を1フレーム処理し、処理を継続するか返す."""
        success, frame = self.cap.read()
        if not success:
            self.get_logger().error(
                'カメラ画像を取得できませんでした。'
            )
            return False

        # 鏡のように見えるよう左右反転してから検出します。
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.lm = hand_landmarks.landmark

                hand_sign = "judge,,,"

                index_extended = self.lm[8].y < self.lm[6].y
                middle_extended = self.lm[12].y < self.lm[10].y
                ring_extended = self.lm[16].y < self.lm[14].y
                pinky_extended = self.lm[20].y < self.lm[18].y

                extended_count = sum(
                    [index_extended, middle_extended, ring_extended, pinky_extended]
                )

                if extended_count == 0:
                    hand_sign = "gu"
                elif extended_count == 2 and index_extended and middle_extended:
                    hand_sign = "kyoki"
                elif extended_count >= 4:
                    hand_sign = "pa"
                else:
                    hand_sign = "judge,,,"

                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )


            cv2.putText(
                frame,
                f"Te: {hand_sign}",
                (30,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
        if self.show_image:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        return True

    def run(self) -> None:
        """ROS イベントとカメラ画像を連続処理する."""
        while rclpy.ok() and self.cap.isOpened():
            rclpy.spin_once(self, timeout_sec=0.0)
            if not self.process_frame():
                break

    def close(self) -> None:
        """カメラと MediaPipe のリソースを解放する."""
        self.cap.release()
        self.hands.close()
        if self.show_image:
            cv2.destroyAllWindows()


def main(args=None) -> None:
    """ROS 2 ノードを初期化して手認識を実行する."""
    rclpy.init(args=args)
    node = None
    try:
        node = HandRecogNode()
        node.run()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info('手認識を終了します。')
    except RuntimeError as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f'[hand_recog] {error}')
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
