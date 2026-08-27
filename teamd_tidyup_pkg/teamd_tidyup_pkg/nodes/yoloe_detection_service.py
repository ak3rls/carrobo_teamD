#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOE promptsを設定した物体検出サービス."""

from importlib.resources import files
import rclpy
from yolov8_detection.detection_service import YOLOv8Node

from teamd_tidyup_pkg.objectlist import load_objectlist

PROMPT_EMBEDDINGS = 'objectlist_yoloe_embeddings.npz'


def _load_objectlist():
    """YAMLからプロンプト一覧と物体名からカテゴリへの対応を読み込む."""
    return load_objectlist()


def _objectlist_names(model_names, prompts):
    """同じ44クラスのモデル名をobjectlist表記へ対応付ける."""
    def key(name):
        normalized = name.lower().replace('nine', '9')
        return ''.join(
            character for character in normalized if character.isalnum()
        )

    objectlist_by_key = {key(name): name for name in prompts}
    names = (
        model_names.items()
        if isinstance(model_names, dict)
        else enumerate(model_names)
    )
    names = list(names)
    if len(names) != len(prompts):
        return None

    mapped = {}
    for index, name in names:
        display_name = objectlist_by_key.get(key(name))
        if display_name is None:
            return None
        mapped[index] = display_name
    return mapped


class YOLOEDetectionNode(YOLOv8Node):
    """YOLOEへobjectlistのクラスを設定して検出サービスを提供する."""

    def __init__(self):
        """モデルを読み込み、YOLOEの場合は保存済みpromptを適用する."""
        super().__init__()
        prompts, self.category_by_name = _load_objectlist()

        if not hasattr(self.model, 'load_prompt_embeddings'):
            objectlist_names = _objectlist_names(self.model.names, prompts)
            if objectlist_names is not None:
                self.class_names = objectlist_names
                self.get_logger().info(
                    'モデル内蔵44クラスをobjectlist表記へ統一しました。'
                )
                return
            self.get_logger().info(
                'YOLOE以外のモデルなので、モデル内蔵クラスを使用します。'
            )
            return

        # set_classes(prompts)だけでは起動時にCLIPと約572MBの
        # MobileCLIPモデルが必要になるため、生成済み埋め込みだけを同梱します。
        embedding_path = files('teamd_tidyup_pkg.nodes').joinpath(
            PROMPT_EMBEDDINGS
        )
        self.model.load_prompt_embeddings(str(embedding_path))
        cached_names = list(self.model.names.values())
        if cached_names != prompts:
            raise ValueError('YOLOE promptと保存済み埋め込みが一致しません。')
        self.model.set_classes(prompts, self.model.model.pe)
        self.class_names = self.model.names
        self.colors = {
            index: [
                (37 * index + 97) % 256,
                (67 * index + 53) % 256,
                (29 * index + 193) % 256,
            ]
            for index in range(len(self.class_names))
        }
        self.get_logger().info(
            f'YOLOEへobjectlist promptを設定しました: {prompts}'
        )

    def create_object_detection_msg(self, bboxes, scores, labels, masks):
        """検出メッセージを作り、物体名・カテゴリ・信頼度をログへ出す."""
        msg = super().create_object_detection_msg(
            bboxes, scores, labels, masks
        )
        for bbox in msg.bbox:
            category = self.category_by_name.get(bbox.name, 'Unknown')
            self.get_logger().info(
                '認識結果: '
                f'category={category}, object={bbox.name}, '
                f'score={bbox.score:.3f}'
            )
        return msg


def main(args=None):
    """YOLOE物体検出サービスを実行する."""
    rclpy.init(args=args)
    node = YOLOEDetectionNode()
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
