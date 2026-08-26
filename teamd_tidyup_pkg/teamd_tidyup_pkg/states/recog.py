#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv8 の検出結果から把持姿勢を求めるステート."""

import math
import time

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401
import tf_transformations as tft
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from grasp_point_detection_interfaces.srv import GraspPointService
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer
from tf2_ros import TransformException
from yasmin import Blackboard
from yasmin import State
from yolov8_detection_interfaces.srv import ObjectDetectionService

from carrobo_manipulation_pkg.hsrif import HSRInterfaces


# 掴みたい物体名を指定します。
# 既存の把持例と同じ対象を初期値にしています。
# 空文字にするとロボットから最も近い物体を選びます。
# ロボット自身を選ぶ可能性があるため、通常は物体名を指定してください。
TARGET_NAME = ''

CONFIDENCE_THRESHOLD = 0.25
MAX_GRASP_DISTANCE = 2.0
DEPTH_SCALES = {
    '16UC1': 0.001,
    '32FC1': 1.0,
}
TALL_THRESHOLD = 0.15
HEAD_TILT = math.radians(-50.0)
MAX_EMPTY_DETECTIONS = 3


class RecogState(State):
    """物体検出と把持点推定を行うステート."""

    def __init__(
        self,
        node: Node,
        hsrif: HSRInterfaces,
        tf_buffer: Buffer,
    ):
        """サービスクライアントを生成する."""
        super().__init__(outcomes=['succeeded', 'failed', 'none'])
        self.node = node
        self.hsrif = hsrif
        self.tf_buffer = tf_buffer
        self.empty_detection_count = 0
        self.bridge = CvBridge()

        self.detect_client = self.node.create_client(
            ObjectDetectionService, '/yolov8_detection/service'
        )
        self.grasp_client = self.node.create_client(
            GraspPointService, '/grasp_point_detection/service'
        )

    def _wait_for_service(self, client, service_name: str) -> bool:
        """サービスが利用可能になるまで待つ."""
        while rclpy.ok():
            if client.wait_for_service(timeout_sec=1.0):
                return True
            self.node.get_logger().info(
                f'{service_name} が見つかりません。起動を待っています...'
            )
        return False

    def _select_target(self, detections) -> int:
        """対象名に一致する検出のうち、ロボットに最も近い添字を返す."""
        candidate_indices = [
            index
            for index, bbox in enumerate(detections.bbox)
            if not TARGET_NAME or bbox.name == TARGET_NAME
        ]
        if not candidate_indices:
            return -1

        depth_scale = DEPTH_SCALES.get(detections.depth.encoding)
        if depth_scale is None:
            self.node.get_logger().error(
                '最も近い物体の選択では未対応の'
                '深度エンコーディングです: '
                f'{detections.depth.encoding}'
            )
            return -1

        try:
            depth = self.bridge.imgmsg_to_cv2(
                detections.depth,
                desired_encoding=detections.depth.encoding,
            ).astype(np.float64) * depth_scale
        except Exception as error:  # cv_bridge の例外型は環境で異なる
            self.node.get_logger().error(
                f'距離選択用の深度画像を変換できませんでした: {error}'
            )
            return -1

        k = detections.camera_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        if fx == 0.0 or fy == 0.0:
            self.node.get_logger().error(
                '距離選択に必要な camera_info が空です。'
            )
            return -1

        candidates = []
        for index in candidate_indices:
            bbox = detections.bbox[index]
            if index >= len(detections.segments):
                self.node.get_logger().warning(
                    f'{bbox.name} に対応するマスクがありません。'
                )
                continue

            try:
                mask = self.bridge.imgmsg_to_cv2(
                    detections.segments[index],
                    desired_encoding='mono8',
                )
            except Exception as error:  # cv_bridge の例外型は環境で異なる
                self.node.get_logger().warning(
                    f'{bbox.name} のマスクを変換できませんでした: {error}'
                )
                continue

            if depth.shape[:2] != mask.shape[:2]:
                self.node.get_logger().warning(
                    f'{bbox.name} の深度画像 {depth.shape[:2]} と'
                    f'マスク {mask.shape[:2]} の解像度が異なります。'
                )
                continue

            valid = (
                (mask != 0)
                & np.isfinite(depth)
                & (depth > 0.0)
                & (depth <= MAX_GRASP_DISTANCE)
            )
            ys, xs = np.nonzero(valid)
            if ys.size == 0:
                self.node.get_logger().warning(
                    f'{bbox.name} の有効な深度を取得できませんでした。'
                )
                continue

            distances = depth[ys, xs]
            median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median))) * 1.4826
            inliers = np.abs(distances - median) <= max(3.0 * mad, 0.005)
            distances = distances[inliers]
            xs = xs[inliers]
            ys = ys[inliers]

            camera_x = np.mean((xs - cx) * distances / fx)
            camera_y = np.mean((ys - cy) * distances / fy)
            camera_z = np.mean(distances)

            stamped = PoseStamped()
            stamped.header = detections.camera_info.header
            stamped.pose.position.x = float(camera_x)
            stamped.pose.position.y = float(camera_y)
            stamped.pose.position.z = float(camera_z)
            stamped.pose.orientation.w = 1.0
            try:
                base_pose = self.tf_buffer.transform(
                    stamped,
                    'base_link',
                    timeout=Duration(seconds=2.0),
                ).pose
            except TransformException as error:
                self.node.get_logger().warning(
                    f'{bbox.name} の距離を base_link 基準に変換'
                    f'できませんでした: {error}'
                )
                continue

            position = base_pose.position
            distance = math.sqrt(
                position.x ** 2 + position.y ** 2 + position.z ** 2
            )
            self.node.get_logger().info(
                f'把持候補 {bbox.name}: ロボットから {distance:.3f} m'
            )
            candidates.append((distance, index))

        if not candidates:
            return -1

        distance, index = min(candidates, key=lambda item: item[0])
        self.node.get_logger().info(
            f'最も近い物体 {detections.bbox[index].name} '
            f'({distance:.3f} m) を選択しました。'
        )
        return index

    def execute(self, blackboard: Blackboard) -> str:
        """対象物体を認識し、把持前姿勢を Blackboard に保存する."""
        self.node.get_logger().info('Executing state Recog')

        if not self._wait_for_service(
            self.detect_client, '/yolov8_detection/service'
        ):
            return 'failed'
        if not self._wait_for_service(
            self.grasp_client, '/grasp_point_detection/service'
        ):
            return 'failed'

        # move_to_go はカメラを腕で隠すため、認識時は neutral 姿勢です。
        self.hsrif.whole_body.move_to_neutral(sync=True)
        self.hsrif.whole_body.move_to_joint_positions(
            {'head_pan_joint': 0.0, 'head_tilt_joint': HEAD_TILT},
            sync=True,
        )
        time.sleep(2.0)

        # TF を受信して Buffer に溜めてからサービスを呼びます。
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        detect_req = ObjectDetectionService.Request()
        detect_req.confidence_th = CONFIDENCE_THRESHOLD
        future = self.detect_client.call_async(detect_req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response is None:
            self.node.get_logger().error(
                '物体検出サービスから応答がありません。'
            )
            return 'failed'

        detections = response.detections
        names = [bbox.name for bbox in detections.bbox]
        self.node.get_logger().info(f'検出した物体: {names}')

        index = self._select_target(detections)
        if index < 0:
            self.empty_detection_count += 1
            target = TARGET_NAME or '物体'
            self.node.get_logger().warning(
                f'{target} が見つかりません '
                f'({self.empty_detection_count}/{MAX_EMPTY_DETECTIONS})。'
            )
            if self.empty_detection_count >= MAX_EMPTY_DETECTIONS:
                self.empty_detection_count = 0
                self.node.get_logger().info(
                    f'{MAX_EMPTY_DETECTIONS}回連続で物体を検出できなかったため、'
                    'none を返します。'
                )
                return 'none'
            return 'failed'

        # 物体を検出できた場合、連続未検出回数をリセットします。
        self.empty_detection_count = 0
        if index >= len(detections.segments):
            self.node.get_logger().error(
                '検出物体に対応するマスクがありません。'
            )
            return 'failed'

        grasp_req = GraspPointService.Request()
        grasp_req.depth = detections.depth
        grasp_req.mask = detections.segments[index]
        grasp_req.camera_info = detections.camera_info
        grasp_req.max_distance = MAX_GRASP_DISTANCE

        future = self.grasp_client.call_async(grasp_req)
        rclpy.spin_until_future_complete(self.node, future)
        grasp_response = future.result()

        if grasp_response is None:
            self.node.get_logger().error(
                '把持点推定サービスから応答がありません。'
            )
            return 'failed'
        if not grasp_response.success:
            self.node.get_logger().error(
                f'把持点を推定できませんでした: {grasp_response.message}'
            )
            return 'failed'

        # 推定結果を、腕を動かす base_link 基準へ変換します。
        stamped = PoseStamped()
        stamped.header = detections.camera_info.header
        stamped.pose = grasp_response.grasp.pose
        try:
            object_pose = self.tf_buffer.transform(
                stamped,
                'base_link',
                timeout=Duration(seconds=2.0),
            ).pose
        except TransformException as error:
            self.node.get_logger().error(
                f'把持姿勢の TF 変換に失敗しました: {error}'
            )
            return 'failed'

        # carrobo_manipulation_pkg の把持例と同じルールで掴む向きを決めます。
        height = grasp_response.grasp.size.z
        if height > TALL_THRESHOLD:
            self.node.get_logger().info('背が高い物体なので横から掴みます。')
            roll = math.pi
            pitch = -math.pi / 2.0
            object_pose.position.x -= 0.1
            approach = 0.1
        else:
            self.node.get_logger().info('平たい物体なので上から掴みます。')
            roll = math.pi
            pitch = 0.0
            object_pose.position.z += 0.1
            approach = 0.05

        qx, qy, qz, qw = tft.quaternion_from_euler(roll, pitch, 0.0)
        object_pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        blackboard.grasp_pose = object_pose
        blackboard.grasp_approach = approach
        blackboard.target_name = detections.bbox[index].name
        self.node.get_logger().info(
            f'{blackboard.target_name} の把持姿勢を Blackboard に保存しました。'
        )
        return 'succeeded'
