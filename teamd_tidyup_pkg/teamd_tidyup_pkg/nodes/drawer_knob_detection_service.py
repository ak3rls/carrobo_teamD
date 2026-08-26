#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引き出しの白いつまみを text prompt で検出する YOLOE サービス.

片付けタスクの物体検出 (44クラス固定モデル) には引き出しのつまみが無いため、
open-vocabulary な YOLOE を prompt 付きで別サービスとして立てる。
サービス名は launch の remap で /drawer_knob_detection/service にする。
"""

from importlib.resources import files
from pathlib import Path

import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLOE
from yolov8_detection.detection_service import YOLOv8Node


PROMPT_EMBEDDINGS = 'drawer_knob_yoloe_embeddings.npz'
DEFAULT_MODEL = 'yoloe-11l-seg.pt'
# Knobs measure 0.10-0.17; keep the model itself permissive.
MODEL_CONFIDENCE = 0.02


class DrawerKnobDetectionNode(YOLOv8Node):
    """YOLOE につまみ prompt を設定して検出サービスを提供する."""

    def __init__(self) -> None:
        """基底ノードを初期化してから、モデルを YOLOE へ差し替える."""
        super().__init__()

        model_path = (
            self.get_parameter('model_path')
            .get_parameter_value()
            .string_value
        )
        if DEFAULT_MODEL not in Path(model_path).name:
            model_path = str(
                Path(get_package_share_directory('teamd_tidyup_pkg'))
                / 'models'
                / DEFAULT_MODEL
            )
            self.get_logger().info(
                f'つまみ検出には YOLOE を使うため model_path を {model_path} '
                'に切り替えました。'
            )
            # The base class already loaded a detector from the old path.
            # Drop it so only the YOLOE weights stay resident.
            self.model = None

        # set_classes(prompts) をそのまま呼ぶと CLIP と MobileCLIP が要るので、
        # objectlist 側と同じく生成済みの埋め込みだけを同梱して読み込む。
        embedding_path = files('teamd_tidyup_pkg.nodes').joinpath(
            PROMPT_EMBEDDINGS
        )
        data = np.load(str(embedding_path), allow_pickle=True)
        names = [str(name) for name in data['names']]
        embeddings = torch.from_numpy(np.asarray(data['embeddings']))

        self.model = YOLOE(model_path)
        self.model.set_classes(names, embeddings)
        self.class_names = self.model.names
        self.colors = {index: [255, 255, 255] for index in range(len(names))}
        self.get_logger().info(
            f'YOLOE へつまみ prompt を設定しました: {names}'
        )

    def inference(self, img):
        """Run YOLOE with a low threshold so small knobs survive to the caller.

        The base class calls ``self.model(img)``, which uses ultralytics'
        default conf=0.25 and drops every detection before the service's own
        ``confidence_th`` is applied.  Measured knob scores are 0.10-0.17, so
        that default hid all of them.  The caller filters by score and by box
        size instead.
        """
        result = self.model(img, conf=MODEL_CONFIDENCE)[0]
        masks = None if result.masks is None else result.masks.data
        return (
            masks,
            result.boxes.xyxy,
            result.boxes.cls,
            result.boxes.conf,
            img.copy(),
        )


def main(args=None) -> None:
    """つまみ検出サービスを実行する."""
    rclpy.init(args=args)
    node = DrawerKnobDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
