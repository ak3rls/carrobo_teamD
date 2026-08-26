#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-guided drawer-opening task for the Isaac Carrobo competition scene.

The robot drives roughly in front of the stair-like cabinet, then finds each
white drawer knob with a YOLOE text prompt ("white drawer knob").  The knob is
localised in base_footprint from its mask and the registered depth image, the
mobile base is aligned so the knob sits exactly on the arm's calibrated grasp
point, and the drawer is pulled open by driving the base straight backwards.

The earlier fixed-route version aimed the hand from dead-reckoned odometry.
That was measured drifting 6-13 cm and ~4 deg over the 2 m approach, which is
far more than the 2 cm window between the knob's plate and its base disc, so
the hand hit the drawer fronts.  Detecting the knob removes that error: the
same measurement put the detected knob within 1 cm of ground truth.
"""

import math
import time

import numpy as np
import rclpy
import tf2_ros
import tf_transformations as tft
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from yasmin import Blackboard
from yasmin import State
from yolov8_detection_interfaces.srv import ObjectDetectionService

from carrobo_manipulation_pkg.hsrif import HSRInterfaces


DRAWER_PARAMETER_DEFAULTS = {
    # --- coarse approach ------------------------------------------------
    # Only has to put the cabinet in view; the detection corrects the rest.
    'first_forward_distance': 1.10,
    'right_turn_degrees': 90.0,
    'second_forward_distance': 0.90,
    'navigation_timeout': 60.0,
    'require_entrance_start': True,
    'entrance_tolerance': 0.25,

    # --- knob detection -------------------------------------------------
    'knob_service': '/drawer_knob_detection/service',
    'knob_confidence': 0.03,
    # The false positives seen in the scene are 12-22 px blobs; every real
    # knob measured 36-45 px, so the size window alone separates them.
    'knob_min_box_pixels': 25.0,
    'knob_max_box_pixels': 120.0,
    'knob_search_tilts': [-0.35, -0.55, -0.20, -0.75],
    'knob_max_distance': 2.0,
    'knob_min_height': 0.10,
    'knob_max_height': 0.95,
    # Boundary between the lower and the upper shelf level.
    'knob_level_split': 0.40,
    # The mask sees the knob's front plate; the stem we close on sits this
    # much further away from the robot.
    'knob_depth_bias': 0.020,
    # The mask centroid sits on the knob's rounded upper face, so the measured
    # height runs ~1 cm high (0.288-0.300 measured for a knob truly at 0.280).
    # Uncorrected, the fingers ride above the 3 cm plate and slip off.
    'knob_height_bias': -0.010,
    'detection_settle_seconds': 1.5,
    # Debug aid: write the head camera frame just before the fingers close.
    'pregrasp_image_path': '',

    # --- calibrated grasp point of the preparation posture ---------------
    # Measured with the gripper open, in base_footprint.  arm_flex is pitched
    # further down than the -1.5708 "arm horizontal" pose and the wrist is
    # pitched back by the same amount, which keeps the hand level while
    # putting the lower knob within reach.  The pairing also keeps arm_lift
    # off its 0 limit for the lower drawers: at the limit the arm sits on the
    # edge of its workspace and the trajectory controller intermittently
    # failed, leaving the fingers up to 6 cm short.
    'grasp_reach_x': 0.6858,
    'grasp_offset_y': 0.0792,
    'grasp_height_at_zero_lift': 0.2513,
    # Measured d(tip z) / d(arm_lift_joint).
    'arm_lift_gain': 0.945,
    'arm_lift_min': 0.0,
    'arm_lift_max': 0.66,
    'preparation_arm_flex': -1.7500,
    'preparation_wrist_flex': 0.1792,
    'preparation_wrist_roll': 0.10,
    # The arm is commanded again if the fingers land further than this from
    # the requested height.
    'arm_settle_tolerance': 0.010,
    'arm_settle_attempts': 3,

    # --- base alignment --------------------------------------------------
    'align_tolerance': 0.008,
    'align_speed': 0.06,
    'align_max_iterations': 2,
    'align_timeout': 25.0,
    'correct_yaw_from_knobs': True,
    'yaw_tolerance_degrees': 1.5,
    'yaw_speed': 0.25,

    # --- grasp and pull ---------------------------------------------------
    'gripper_open_angle': 0.90,
    'gripper_open_duration': 0.70,
    # hsrb_interface's gripper.command() takes an ANGLE, not a force. Closing
    # to 0 squeezes the 3 cm knob stem; the old 0.40 stopped ~5 cm apart.
    'gripper_close_angle': 0.0,
    'gripper_close_duration': 2.0,
    'use_gripper_force': False,
    'gripper_hold_force': 0.40,
    # The stem holds the fingers open past the commanded angle by this much.
    'grasp_angle_margin': 0.03,
    'grasp_settle_seconds': 0.50,
    'pull_distance': 0.20,
    'base_pull_speed': 0.08,
    'base_motion_weight': 100.0,
    'retreat_distance': 0.10,

    # --- sequence ---------------------------------------------------------
    'route_only': False,
    'detect_only': False,
    'align_only': False,
    'max_drawers': 3,
    # /arm_trajectory_controller fails intermittently and leaves the hand up
    # to 12 cm off the commanded pose, so a single attempt is not reliable.
    # Re-detecting and re-aligning recovers it.
    'grasp_attempts': 3,
    # The fingers reach the commanded angle on empty air but are held open by
    # the 3 cm stem, so hand_motor_joint is a free, reliable grasp check.
    'verify_grasp': True,
    # Two knobs count as the same one when their odom positions are closer
    # than this, which is how an already-opened drawer is skipped.
    'same_knob_radius': 0.12,
    'post_task_retreat_distance': 0.10,
    'post_task_retreat_speed': 0.08,
}

DEPTH_SCALES = {'16UC1': 0.001, '32FC1': 1.0}


def launch_default(value) -> str:
    """Convert a typed default into ROS launch argument text."""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class Knob:
    """One detected drawer knob, expressed in base_footprint."""

    def __init__(self, x, y, z, score, odom=None):
        """Store the knob position, its detection score and its odom anchor."""
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.score = float(score)
        self.odom = odom

    def __repr__(self):
        """Return a compact form for the task log."""
        return (f'Knob(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, '
                f'score={self.score:.3f})')


class DrawerOpenTask:
    """Reusable drawer task that runs on an existing ROS node and HSR interface."""

    def __init__(
        self,
        node: Node,
        hsrif: HSRInterfaces = None,
        tf_buffer=None,
    ) -> None:
        """Attach the task to shared ROS and robot interfaces."""
        self.node = node
        self._declare_parameters()
        if tf_buffer is None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(
                self.tf_buffer, self.node
            )
        else:
            self.tf_buffer = tf_buffer
            self.tf_listener = None
        # Direct wheel command: every base move in this task is a straight
        # translation, so it never goes through the navigation stack.
        self.base_cmd_pub = self.node.create_publisher(
            Twist, '/omni_base_controller/cmd_vel', 10
        )
        self.bridge = CvBridge()
        self._joint_state = None
        self.node.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10
        )
        self._rgb = None
        self.node.create_subscription(
            Image,
            '/head_rgbd_sensor/rgb/image_rect_color',
            self._on_rgb,
            qos_profile_sensor_data,
        )
        self.knob_client = self.node.create_client(
            ObjectDetectionService, str(self._value('knob_service'))
        )
        self.hsrif = hsrif

    def _on_joint_state(self, msg) -> None:
        self._joint_state = msg

    def _on_rgb(self, msg) -> None:
        self._rgb = msg

    def _save_pregrasp_image(self, tag: str) -> None:
        """Write the current camera frame so a grasp can be inspected later."""
        path = str(self._value('pregrasp_image_path'))
        if not path:
            return
        self._spin(0.6)
        if self._rgb is None:
            self.get_logger().warn('保存する画像がまだ届いていません。')
            return
        import cv2
        image = self.bridge.imgmsg_to_cv2(self._rgb, desired_encoding='bgr8')
        name = f'{path}_{tag}.png'
        cv2.imwrite(name, image)
        self.get_logger().info(f'把持直前の画像を保存しました: {name}')

    def _declare_parameters(self) -> None:
        """Declare the same calibrated parameters for standalone and SM use."""
        for name, default in DRAWER_PARAMETER_DEFAULTS.items():
            self.node.declare_parameter(name, default)

    def get_logger(self):
        """Return the logger owned by the shared ROS node."""
        return self.node.get_logger()

    def stop_base(self) -> None:
        """Publish an explicit zero velocity during cleanup or handoff."""
        self.base_cmd_pub.publish(Twist())

    def _value(self, name: str):
        return self.node.get_parameter(name).value

    def _spin(self, seconds: float) -> None:
        """Service callbacks and TF for a fixed wall-clock time."""
        end = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    # ------------------------------------------------------------------
    # Coarse approach
    # ------------------------------------------------------------------

    def _base_pose(self):
        """Return the current odom pose of base_footprint."""
        last_error = None
        for _ in range(30):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'odom', 'base_footprint', rclpy.time.Time()
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                yaw = math.atan2(
                    2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
                    1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
                )
                return translation.x, translation.y, yaw
            except Exception as error:
                last_error = error
                rclpy.spin_once(self.node, timeout_sec=0.1)
        raise RuntimeError(f'odom/base_footprint TF を取得できません: {last_error}')

    def _go_abs(self, x: float, y: float, yaw: float, label: str) -> None:
        """Drive to one route waypoint and fail if navigation reports failure."""
        self.get_logger().info(
            f'{label}: odom goal=(x={x:.3f}, y={y:.3f}, '
            f'yaw={math.degrees(yaw):.1f} deg)'
        )
        result = self.hsrif.omni_base.go_abs(
            x, y, yaw, timeout=float(self._value('navigation_timeout')), sync=True
        )
        if result is False:
            raise RuntimeError(f'{label} へ到達できませんでした。')

    def _drive_from_entrance(self) -> None:
        """Forward -> right turn -> forward from the initial odom pose."""
        start_x, start_y, start_yaw = self._base_pose()
        self.get_logger().info(
            f'開始位置: odom=(x={start_x:.3f}, y={start_y:.3f}, '
            f'yaw={math.degrees(start_yaw):.1f} deg)'
        )
        if bool(self._value('require_entrance_start')):
            tolerance = float(self._value('entrance_tolerance'))
            if math.hypot(start_x, start_y) > tolerance or abs(start_yaw) > 0.20:
                raise RuntimeError(
                    'ロボットが入口の初期位置にいません。Isaac Sim をリセットしてから再実行してください。'
                )
        first = float(self._value('first_forward_distance'))
        first_x = start_x + first * math.cos(start_yaw)
        first_y = start_y + first * math.sin(start_yaw)
        self._go_abs(first_x, first_y, start_yaw, '1/3 前進')
        turn_yaw = wrap_angle(
            start_yaw - math.radians(float(self._value('right_turn_degrees')))
        )
        self._go_abs(first_x, first_y, turn_yaw, '2/3 右回転')
        second = float(self._value('second_forward_distance'))
        second_x = first_x + second * math.cos(turn_yaw)
        second_y = first_y + second * math.sin(turn_yaw)
        self._go_abs(second_x, second_y, turn_yaw, '3/3 前進')
        self.get_logger().info(
            '棚の前に到着しました。ここから先はつまみの検出結果で位置を合わせます。'
        )

    # ------------------------------------------------------------------
    # Knob detection
    # ------------------------------------------------------------------

    def _look(self, tilt: float) -> None:
        """Point the head at the cabinet without the arm blocking the camera."""
        self.hsrif.whole_body.move_to_joint_positions(
            {'head_pan_joint': 0.0, 'head_tilt_joint': float(tilt)},
            sync=True,
        )
        self._spin(float(self._value('detection_settle_seconds')))

    def _call_knob_service(self):
        """Run one detection and return the raw ObjectDetection message."""
        service = str(self._value('knob_service'))
        if not self.knob_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError(f'{service} が見つかりません。')
        request = ObjectDetectionService.Request()
        request.confidence_th = float(self._value('knob_confidence'))
        future = self.knob_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        response = future.result()
        if response is None:
            raise RuntimeError(f'{service} から応答がありません。')
        return response.detections

    def _knobs_from_detections(self, detections):
        """Turn masks plus registered depth into knob points in base_footprint."""
        if not detections.bbox:
            return []
        scale = DEPTH_SCALES.get(detections.depth.encoding)
        if scale is None:
            raise RuntimeError(
                f'未対応の深度エンコーディングです: {detections.depth.encoding}'
            )
        depth = self.bridge.imgmsg_to_cv2(
            detections.depth, desired_encoding=detections.depth.encoding
        ).astype(np.float64) * scale

        matrix = detections.camera_info.k
        fx, fy, cx, cy = matrix[0], matrix[4], matrix[2], matrix[5]
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError('camera_info が空です。')

        camera_frame = detections.camera_info.header.frame_id
        transform = self.tf_buffer.lookup_transform(
            'base_footprint', camera_frame, rclpy.time.Time()
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        camera_to_base = tft.quaternion_matrix(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        )
        camera_to_base[:3, 3] = [translation.x, translation.y, translation.z]

        min_box = float(self._value('knob_min_box_pixels'))
        max_box = float(self._value('knob_max_box_pixels'))
        max_distance = float(self._value('knob_max_distance'))

        knobs = []
        for index, bbox in enumerate(detections.bbox):
            width, height = float(bbox.w), float(bbox.h)
            if not (min_box <= width <= max_box
                    and min_box <= height <= max_box):
                continue
            if index >= len(detections.segments):
                continue
            mask = self.bridge.imgmsg_to_cv2(
                detections.segments[index], desired_encoding='mono8'
            )
            if mask.shape[:2] != depth.shape[:2]:
                continue
            valid = (
                (mask != 0)
                & np.isfinite(depth)
                & (depth > 0.10)
                & (depth <= max_distance)
            )
            ys, xs = np.nonzero(valid)
            if ys.size < 20:
                continue
            distances = depth[ys, xs]
            # The mask spans the knob's rounded face; its near half is the
            # front plate, which is the surface we measure against.
            near = distances <= np.percentile(distances, 40.0)
            ys, xs, distances = ys[near], xs[near], distances[near]
            points = np.stack(
                [
                    (xs - cx) * distances / fx,
                    (ys - cy) * distances / fy,
                    distances,
                ],
                axis=1,
            )
            in_base = (camera_to_base[:3, :3] @ points.T).T + camera_to_base[:3, 3]
            centre = in_base.mean(axis=0)
            if not (float(self._value('knob_min_height'))
                    <= centre[2]
                    <= float(self._value('knob_max_height'))):
                continue
            knobs.append(Knob(centre[0], centre[1], centre[2], bbox.score))

        knobs.sort(key=lambda knob: -knob.score)
        return knobs

    def _detect_knobs(self):
        """Detect from several head tilts and fuse the sightings in odom.

        A single tilt is not enough: a steep tilt sees all three knobs but
        measures their height ~2 cm high, while a shallow one is accurate but
        misses the lower shelf.  Clustering every sighting in odom and taking
        a score-weighted mean keeps both the coverage and the accuracy.
        """
        radius = float(self._value('same_knob_radius'))
        clusters = []
        for tilt in list(self._value('knob_search_tilts')):
            self._look(float(tilt))
            try:
                detections = self._call_knob_service()
                knobs = self._knobs_from_detections(detections)
            except Exception as error:
                self.get_logger().warn(
                    f'tilt={float(tilt):.2f} での検出に失敗しました: {error}'
                )
                continue
            self.get_logger().info(
                f'tilt={float(tilt):.2f}: つまみ {len(knobs)} 個 -> {knobs}'
            )
            for knob in knobs:
                anchored = self._to_odom(knob)
                for cluster in clusters:
                    if math.dist(cluster[0].odom, anchored.odom) < radius:
                        cluster.append(anchored)
                        break
                else:
                    clusters.append([anchored])

        base_x, base_y, yaw = self._base_pose()
        fused = []
        for cluster in clusters:
            weights = np.array([max(knob.score, 1e-3) for knob in cluster])
            odom = np.array([knob.odom for knob in cluster])
            mean = (odom * weights[:, None]).sum(axis=0) / weights.sum()
            # Bring the fused odom point back into the current base frame.
            dx, dy = mean[0] - base_x, mean[1] - base_y
            knob = Knob(
                dx * math.cos(yaw) + dy * math.sin(yaw),
                -dx * math.sin(yaw) + dy * math.cos(yaw),
                mean[2],
                float(weights.max()),
            )
            knob.odom = (mean[0], mean[1], mean[2])
            fused.append(knob)
        fused.sort(key=lambda knob: -knob.score)
        self.get_logger().info(f'統合したつまみ {len(fused)} 個 -> {fused}')
        return fused

    def _in_current_base(self, knob: Knob):
        """Express an odom-anchored knob in the base frame we occupy now."""
        base_x, base_y, yaw = self._base_pose()
        dx, dy = knob.odom[0] - base_x, knob.odom[1] - base_y
        return (
            dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw),
            knob.odom[2],
        )

    def _to_odom(self, knob: Knob) -> Knob:
        """Anchor a knob in odom so an opened drawer can be recognised later."""
        base_x, base_y, yaw = self._base_pose()
        knob.odom = (
            base_x + knob.x * math.cos(yaw) - knob.y * math.sin(yaw),
            base_y + knob.x * math.sin(yaw) + knob.y * math.cos(yaw),
            knob.z,
        )
        return knob

    def _select_knob(self, knobs, level: str, prefer_left: bool, opened):
        """Pick the next knob to open at the requested shelf level."""
        split = float(self._value('knob_level_split'))
        radius = float(self._value('same_knob_radius'))
        candidates = []
        for knob in knobs:
            if level == 'bottom' and knob.z >= split:
                continue
            if level == 'upper' and knob.z < split:
                continue
            anchored = self._to_odom(knob)
            # Compare in 3D: the two centre drawers sit at the same xy and are
            # told apart only by height.
            if any(math.dist(anchored.odom, done) < radius
                   for done in opened):
                continue
            candidates.append(anchored)
        if not candidates:
            return None
        if prefer_left:
            # base +Y is the robot's left.
            return max(candidates, key=lambda knob: knob.y)
        centre = float(self._value('grasp_offset_y'))
        return min(candidates, key=lambda knob: abs(knob.y - centre))

    # ------------------------------------------------------------------
    # Base alignment
    # ------------------------------------------------------------------

    def _drive_base_relative(
        self, forward: float, left: float, label: str, speed: float = None
    ) -> None:
        """Translate the base by a base-frame offset, closing the loop on odom."""
        tolerance = float(self._value('align_tolerance'))
        if math.hypot(forward, left) < tolerance:
            return
        if speed is None:
            speed = float(self._value('align_speed'))
        speed = abs(speed)
        timeout = float(self._value('align_timeout'))
        start_x, start_y, start_yaw = self._base_pose()
        # Express the goal in odom so wheel slip cannot accumulate.
        goal_x = start_x + forward * math.cos(start_yaw) - left * math.sin(start_yaw)
        goal_y = start_y + forward * math.sin(start_yaw) + left * math.cos(start_yaw)
        self.get_logger().info(
            f'{label}: 前後 {forward:+.3f} m, 左右 {left:+.3f} m 台車を平行移動します。'
        )
        started = time.monotonic()
        command = Twist()
        try:
            while rclpy.ok() and time.monotonic() - started < timeout:
                current_x, current_y, yaw = self._base_pose()
                error_x = goal_x - current_x
                error_y = goal_y - current_y
                distance = math.hypot(error_x, error_y)
                if distance < tolerance:
                    break
                # Rotate the odom-frame error into the base frame.
                base_forward = error_x * math.cos(yaw) + error_y * math.sin(yaw)
                base_left = -error_x * math.sin(yaw) + error_y * math.cos(yaw)
                gain = min(1.0, distance / 0.05)
                command.linear.x = speed * gain * base_forward / distance
                command.linear.y = speed * gain * base_left / distance
                self.base_cmd_pub.publish(command)
                rclpy.spin_once(self.node, timeout_sec=0.02)
            else:
                self.get_logger().warn(f'{label}: 整定前にタイムアウトしました。')
        finally:
            self.base_cmd_pub.publish(Twist())
        final_x, final_y, _ = self._base_pose()
        self.get_logger().info(
            f'{label}: 残差 {math.dist((final_x, final_y), (goal_x, goal_y)):.4f} m'
        )

    def _drive_base_straight(
        self, distance: float, speed: float, label: str, axis: str = 'x'
    ) -> None:
        """Drive along one current base axis for a calibrated duration."""
        if distance <= 0.0:
            return
        if axis not in ('x', 'y'):
            raise ValueError("axis は 'x' または 'y' にしてください。")
        speed = math.copysign(max(0.01, abs(speed)), speed)
        duration = distance / abs(speed)
        command = Twist()
        if axis == 'x':
            command.linear.x = speed
        else:
            command.linear.y = speed
        self.get_logger().info(
            f'{label}: 回転せず、台車を {distance:.3f} m '
            f'({abs(speed):.3f} m/s, {duration:.2f} 秒) 動かします。'
        )
        started = time.monotonic()
        try:
            while rclpy.ok() and time.monotonic() - started < duration:
                self.base_cmd_pub.publish(command)
                rclpy.spin_once(self.node, timeout_sec=0.05)
        finally:
            self.base_cmd_pub.publish(Twist())

    def _correct_yaw(self, knobs) -> bool:
        """Square up to the cabinet using two knobs on the same shelf level."""
        if not bool(self._value('correct_yaw_from_knobs')) or len(knobs) < 2:
            return False
        split = float(self._value('knob_level_split'))
        for level in ('bottom', 'upper'):
            same = [
                knob for knob in knobs
                if (knob.z < split) == (level == 'bottom')
            ]
            if len(same) < 2:
                continue
            left = max(same, key=lambda knob: knob.y)
            right = min(same, key=lambda knob: knob.y)
            span = left.y - right.y
            if span < 0.15:
                continue
            # Knobs on one level lie on the cabinet front, so the line through
            # them is parallel to it.  Any depth difference is our yaw error.
            error = math.atan2(left.x - right.x, span)
            if abs(error) < math.radians(float(self._value('yaw_tolerance_degrees'))):
                return False
            self.get_logger().info(
                f'{level} の2つのつまみから yaw ずれ '
                f'{math.degrees(error):+.2f} 度を検出しました。補正します。'
            )
            # error is how far the robot is turned away from square, so the
            # base has to rotate back by the same amount.
            self._rotate_base(-error)
            return True
        return False

    def _rotate_base(self, delta_yaw: float) -> None:
        """Rotate in place by a small odom-closed-loop angle."""
        speed = abs(float(self._value('yaw_speed')))
        timeout = float(self._value('align_timeout'))
        _, _, start_yaw = self._base_pose()
        goal_yaw = wrap_angle(start_yaw + delta_yaw)
        started = time.monotonic()
        command = Twist()
        try:
            while rclpy.ok() and time.monotonic() - started < timeout:
                _, _, yaw = self._base_pose()
                error = wrap_angle(goal_yaw - yaw)
                if abs(error) < math.radians(0.5):
                    break
                command.angular.z = math.copysign(
                    speed * min(1.0, abs(error) / 0.15), error
                )
                self.base_cmd_pub.publish(command)
                rclpy.spin_once(self.node, timeout_sec=0.02)
        finally:
            self.base_cmd_pub.publish(Twist())

    def _align_to_knob(self, knob: Knob) -> Knob:
        """Move the base until the knob sits on the arm's grasp point."""
        reach = float(self._value('grasp_reach_x'))
        offset = float(self._value('grasp_offset_y'))
        bias = float(self._value('knob_depth_bias'))
        tolerance = float(self._value('align_tolerance'))
        target = knob
        for iteration in range(int(self._value('align_max_iterations'))):
            forward = (target.x + bias) - reach
            left = target.y - offset
            self.get_logger().info(
                f'位置合わせ {iteration + 1}: つまみ {target} -> '
                f'前後 {forward:+.3f} m, 左右 {left:+.3f} m'
            )
            if math.hypot(forward, left) < tolerance:
                self.get_logger().info('すでに把持位置に合っています。')
                return target
            self._drive_base_relative(forward, left, f'位置合わせ {iteration + 1}')
            if iteration + 1 >= int(self._value('align_max_iterations')):
                break
            knobs = self._detect_knobs()
            refreshed = self._nearest(knobs, target)
            if refreshed is None:
                self.get_logger().warn(
                    '位置合わせ後につまみを再検出できませんでした。'
                    '直前の推定値のまま把持します。'
                )
                break
            target = refreshed
        return target

    def _nearest(self, knobs, reference: Knob):
        """Find the same knob again after the base moved."""
        if not knobs:
            return None
        radius = float(self._value('same_knob_radius'))
        anchored = [self._to_odom(knob) for knob in knobs]
        # The two centre drawers share an xy position and differ only in
        # height, so the match has to be in 3D or the levels get swapped.
        best = min(
            anchored,
            key=lambda knob: math.dist(knob.odom, reference.odom),
        )
        if math.dist(best.odom, reference.odom) > radius:
            return None
        return best

    # ------------------------------------------------------------------
    # Grasp and pull
    # ------------------------------------------------------------------

    def _arm_lift_for(self, knob_z: float) -> float:
        """Return the arm_lift that puts the open fingers at the knob height."""
        base_height = float(self._value('grasp_height_at_zero_lift'))
        gain = float(self._value('arm_lift_gain'))
        corrected = knob_z + float(self._value('knob_height_bias'))
        lift = (corrected - base_height) / gain
        return min(
            max(lift, float(self._value('arm_lift_min'))),
            float(self._value('arm_lift_max')),
        )

    def _prepare_arm(self, knob: Knob) -> None:
        """Open the hand and hold the calibrated posture at the knob height."""
        lift = self._arm_lift_for(knob.z)
        self.get_logger().info(
            f'つまみ高さ 検出 {knob.z:.3f} m '
            f'(補正 {knob.z + float(self._value("knob_height_bias")):.3f} m) '
            f'に合わせて arm_lift={lift:.3f} にします。'
        )
        self.hsrif.gripper.command(
            float(self._value('gripper_open_angle')),
            float(self._value('gripper_open_duration')),
        )
        self._spin(0.5)

        # /arm_trajectory_controller intermittently reports a failed
        # trajectory and leaves the hand short of the commanded pose, by up to
        # 6 cm in a measured sweep.  Re-send until the fingers actually arrive.
        target_z = (
            float(self._value('grasp_height_at_zero_lift'))
            + float(self._value('arm_lift_gain')) * lift
        )
        tolerance = float(self._value('arm_settle_tolerance'))
        attempts = int(self._value('arm_settle_attempts'))
        for attempt in range(attempts):
            self.hsrif.whole_body.move_to_joint_positions(
                {
                    'arm_lift_joint': lift,
                    'arm_flex_joint': float(self._value('preparation_arm_flex')),
                    'arm_roll_joint': 0.0,
                    'wrist_flex_joint': float(
                        self._value('preparation_wrist_flex')
                    ),
                    'wrist_roll_joint': float(
                        self._value('preparation_wrist_roll')
                    ),
                },
                sync=True,
            )
            self._spin(1.0)
            tip = self._fingertip()
            if tip is None:
                break
            error = math.hypot(
                tip[2] - target_z,
                tip[0] - float(self._value('grasp_reach_x')),
            )
            self.get_logger().info(
                f'腕の整定 {attempt + 1}/{attempts}: '
                f'指先=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) '
                f'目標=(x {float(self._value("grasp_reach_x")):.3f}, '
                f'z {target_z:.3f}) ずれ={error:.4f} m'
            )
            if error <= tolerance:
                break
        self._nudge_base_to_fingertip(knob)
        self._log_grasp_residual(knob)

    def _fingertip(self):
        """Return the midpoint of the two fingertips in base_footprint."""
        try:
            points = []
            for frame in ('hand_l_finger_tip_frame',
                          'hand_r_finger_tip_frame'):
                translation = self.tf_buffer.lookup_transform(
                    'base_footprint', frame, rclpy.time.Time()
                ).transform.translation
                points.append(
                    [translation.x, translation.y, translation.z]
                )
            return np.mean(points, axis=0)
        except Exception as error:
            self.get_logger().warn(f'指先 TF を取得できません: {error}')
            return None

    def _nudge_base_to_fingertip(self, knob: Knob) -> None:
        """Close the last centimetres with the base, which lands within 8 mm.

        The arm's own repeatability is the weak link here, so the final planar
        correction is measured from where the fingers actually are rather than
        from where they were asked to be.
        """
        tip = self._fingertip()
        if tip is None:
            return
        current = self._in_current_base(knob)
        axis_x = current[0] + float(self._value('knob_depth_bias'))
        forward = axis_x - float(tip[0])
        left = current[1] - float(tip[1])
        self.get_logger().info(
            f'指先実測にもとづく最終補正: 前後 {forward:+.3f} m, '
            f'左右 {left:+.3f} m'
        )
        self._drive_base_relative(forward, left, '把持直前の最終補正')

    def _log_grasp_residual(self, knob: Knob) -> None:
        """Report where the fingers actually ended up relative to the knob."""
        tip = self._fingertip()
        if tip is None:
            return
        bias = float(self._value('knob_depth_bias'))
        # knob.x/y were measured before the alignment moved the base, so use
        # the odom anchor to express the knob in the base frame we are in now.
        current = self._in_current_base(knob)
        target = np.array([
            current[0] + bias,
            current[1],
            current[2] + float(self._value('knob_height_bias')),
        ])
        error = tip - target
        self._save_pregrasp_image(f'z{tip[2]:.3f}')
        self.get_logger().info(
            f'把持直前: 指先=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) '
            f'つまみ軸=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) '
            f'残差=({error[0]:+.3f}, {error[1]:+.3f}, {error[2]:+.3f})'
        )

    def _close_on_knob(self) -> None:
        """Close the fingers around the knob stem and keep holding."""
        self.get_logger().info('つまみを挟むためハンドを閉じます。')
        self.hsrif.gripper.command(
            float(self._value('gripper_close_angle')),
            float(self._value('gripper_close_duration')),
        )
        if bool(self._value('use_gripper_force')):
            try:
                self.hsrif.gripper.apply_force(
                    float(self._value('gripper_hold_force')),
                    delicate=True,
                    sync=False,
                )
            except Exception as error:
                # The position-close still holds in simulators without the
                # optional force-control action.
                self.get_logger().warn(
                    f'把持力制御を使えません。位置閉じで継続します: {error}'
                )
        self._spin(float(self._value('grasp_settle_seconds')))

    def _hand_motor_angle(self):
        """Return the current hand_motor_joint angle, or None if unseen."""
        for _ in range(40):
            state = getattr(self, '_joint_state', None)
            if state is not None and 'hand_motor_joint' in state.name:
                return float(
                    state.position[state.name.index('hand_motor_joint')]
                )
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return None

    def _gripper_holds_knob(self) -> bool:
        """Confirm that the fingers stopped on the knob, not on empty air.

        ``gripper.get_distance()`` is not usable here: in this simulator it
        raised and blocked the task for minutes.  The commanded close angle is
        below the angle the stem physically allows, so a hand that reached its
        target closed on nothing.
        """
        self._spin(0.4)
        angle = self._hand_motor_angle()
        if angle is None:
            self.get_logger().warn(
                'hand_motor_joint を取得できないため把持確認を省略します。'
            )
            return True
        commanded = float(self._value('gripper_close_angle'))
        margin = float(self._value('grasp_angle_margin'))
        held = angle > commanded + margin
        self.get_logger().info(
            f'つまみ把持確認: hand_motor={angle:+.4f} rad '
            f'(指令 {commanded:+.3f}, 判定 {"把持" if held else "空振り"})'
        )
        return held

    def _pull_drawer(self) -> None:
        """Pull straight backward without issuing a turn or arm command.

        This closes the loop on odom rather than timing a velocity: pulling a
        loaded drawer open-loop under-delivered, moving the drawer 0.117 m for
        a commanded 0.20 m.
        """
        self._drive_base_relative(
            -max(0.0, float(self._value('pull_distance'))),
            0.0,
            '引き出しを引くため台車後退',
            speed=float(self._value('base_pull_speed')),
        )

    def _release_gripper_only(self) -> None:
        """Release the knob without moving the base, arm, or wrist."""
        self.hsrif.gripper.command(
            1.0, float(self._value('gripper_open_duration'))
        )
        self._spin(0.5)
        self.get_logger().info('ハンドを開いてつまみを離しました。')

    def _release_and_retreat(self) -> None:
        """Leave the drawer front clear after an unsuccessful grasp."""
        self._release_gripper_only()
        self._drive_base_straight(
            max(0.0, float(self._value('retreat_distance'))),
            -float(self._value('base_pull_speed')),
            '把持失敗後の後退',
        )
        self.hsrif.whole_body.move_to_go(sync=True)

    def _prepare_for_navigation(self) -> None:
        """Clear the cabinet and fold the robot into its travel posture."""
        self._drive_base_straight(
            max(0.0, float(self._value('post_task_retreat_distance'))),
            -abs(float(self._value('post_task_retreat_speed'))),
            '全引き出し完了後のナビゲーション用後退',
        )
        self.hsrif.whole_body.move_to_go(sync=True)
        self.stop_base()
        self.get_logger().info(
            '全引き出しが完了しました。腕を go 姿勢に収納してナビゲーションへ引き渡します。'
        )

    def _open_one_drawer(self, knob: Knob) -> bool:
        """Align, grasp and pull one detected drawer, retrying a failed grasp."""
        attempts = max(1, int(self._value('grasp_attempts')))
        target = knob
        for attempt in range(attempts):
            if attempt > 0:
                self.get_logger().info(
                    f'把持をやり直します ({attempt + 1}/{attempts})。'
                )
                self._release_gripper_only()
                self.hsrif.whole_body.move_to_neutral(sync=True)
                refreshed = self._nearest(self._detect_knobs(), target)
                if refreshed is None:
                    self.get_logger().error(
                        'やり直しのためのつまみを再検出できませんでした。'
                    )
                    break
                target = refreshed
            aligned = self._align_to_knob(target)
            if bool(self._value('align_only')):
                self.get_logger().info('align_only=true: 位置合わせで停止します。')
                return False
            self._prepare_arm(aligned)
            self._close_on_knob()
            if (not bool(self._value('verify_grasp'))
                    or self._gripper_holds_knob()):
                self._pull_drawer()
                self._release_gripper_only()
                self.get_logger().info('引き出しを1つ開けました。')
                return True
            self.get_logger().warn(
                f'つまみを掴めませんでした ({attempt + 1}/{attempts})。'
            )
            target = aligned
        self.get_logger().error('規定回数の把持に失敗しました。退避します。')
        self._release_and_retreat()
        return False

    # ------------------------------------------------------------------
    # Task entry point
    # ------------------------------------------------------------------

    def _validate_full_sequence(self) -> None:
        """Reject calibration modes when invoked as the first YASMIN task."""
        for name in ('route_only', 'detect_only', 'align_only'):
            if bool(self._value(name)):
                raise RuntimeError(f'state machine で {name} は使用できません。')
        if int(self._value('max_drawers')) < 1:
            raise RuntimeError('state machine では max_drawers を1以上にしてください。')

    def run_task(self, require_full_sequence: bool = False) -> bool:
        """Detect every drawer knob in turn and pull each drawer open."""
        whole_body = None
        original_linear_weight = None
        original_angular_weight = None
        manipulation_started = False
        try:
            if require_full_sequence:
                self._validate_full_sequence()
            if self.hsrif is None:
                self.hsrif = HSRInterfaces()

            self._drive_from_entrance()
            if bool(self._value('route_only')):
                self.get_logger().info(
                    'route_only=true: 移動のみ完了しました。'
                )
                return True

            # move_to_go hides the camera behind the arm; neutral does not.
            self.hsrif.whole_body.move_to_neutral(sync=True)

            whole_body = self.hsrif.whole_body._wb
            try:
                original_linear_weight = whole_body.linear_weight
                original_angular_weight = whole_body.angular_weight
            except Exception as error:
                raise RuntimeError(
                    f'元の全身プランナ重みを読み取れません: {error}'
                ) from error
            weight = float(self._value('base_motion_weight'))
            whole_body.linear_weight = weight
            whole_body.angular_weight = weight

            knobs = self._detect_knobs()
            if not knobs:
                raise RuntimeError('引き出しのつまみを検出できませんでした。')
            if self._correct_yaw(knobs):
                knobs = self._detect_knobs()
            if bool(self._value('detect_only')):
                self.get_logger().info(f'detect_only=true: 検出結果 {knobs}')
                return True

            opened = []
            plan = [('bottom', False), ('upper', False), ('bottom', True)]
            for index, (level, prefer_left) in enumerate(plan):
                if len(opened) >= int(self._value('max_drawers')):
                    break
                if index > 0:
                    self.hsrif.whole_body.move_to_neutral(sync=True)
                    knobs = self._detect_knobs()
                target = self._select_knob(knobs, level, prefer_left, opened)
                if target is None:
                    self.get_logger().warn(
                        f'{level} の未開封の引き出しが見つかりません。次へ進みます。'
                    )
                    continue
                self.get_logger().info(
                    f'{index + 1} 番目の対象: {level} {target}'
                )
                manipulation_started = True
                if not self._open_one_drawer(target):
                    return False
                opened.append(target.odom)

            if not opened:
                raise RuntimeError('引き出しを1つも開けられませんでした。')
            self._prepare_for_navigation()
            self.get_logger().info(f'{len(opened)} 個の引き出しを開けました。')
            return True
        except Exception as error:
            self.get_logger().error(f'引き出し操作に失敗しました: {error}')
            try:
                self.stop_base()
            except Exception as stop_error:
                self.get_logger().error(f'台車の停止にも失敗しました: {stop_error}')
            try:
                if self.hsrif is not None and manipulation_started:
                    self._release_and_retreat()
            except Exception as retreat_error:
                self.get_logger().error(f'退避にも失敗しました: {retreat_error}')
            return False
        finally:
            # The teammate grasp/manipulation states reuse this interface.
            if whole_body is not None:
                try:
                    if original_linear_weight is not None:
                        whole_body.linear_weight = original_linear_weight
                    if original_angular_weight is not None:
                        whole_body.angular_weight = original_angular_weight
                except Exception as error:
                    self.get_logger().warn(
                        f'全身プランナ重みの復元に失敗しました: {error}'
                    )


class OpenDrawersState(State):
    """Open all drawers, clear the cabinet, and hand off to navigation."""

    def __init__(self, node: Node, drawer_task: DrawerOpenTask):
        """Store the shared node and reusable drawer controller."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.drawer_task = drawer_task

    def execute(self, blackboard: Blackboard) -> str:
        """Run the drawer sequence once before the object-grasp states."""
        self.node.get_logger().info('Executing state OpenDrawers')
        blackboard.drawers_opened = False

        if not self.drawer_task.run_task(require_full_sequence=True):
            self.node.get_logger().error(
                '引き出しタスクに失敗したため、物体把持へは進みません。'
            )
            return 'failed'

        blackboard.drawers_opened = True
        self.node.get_logger().info(
            '引き出しタスク完了。次のステートへ遷移します。'
        )
        return 'succeeded'


class DrawerOpenNode(Node):
    """Thin standalone ROS wrapper around :class:`DrawerOpenTask`."""

    def __init__(self) -> None:
        """Create the standalone node used by the open_drawer executable."""
        super().__init__('drawer_open')
        self.task = DrawerOpenTask(self)

    def run_task(self) -> bool:
        """Run the reusable drawer task in standalone mode."""
        return self.task.run_task()


def main(args=None) -> None:
    """ROS 2 console entry point."""
    rclpy.init(args=args)
    node = DrawerOpenNode()
    try:
        node.run_task()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
