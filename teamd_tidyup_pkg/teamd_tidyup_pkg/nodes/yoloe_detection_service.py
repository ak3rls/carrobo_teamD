#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOE promptsを設定した物体検出サービス."""

from importlib.resources import files
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from yolov8_detection.detection_service import YOLOv8Node


OBJECTLIST_FILE = 'objectlist.yaml'
PROMPT_EMBEDDINGS = 'objectlist_yoloe_embeddings.npz'


def _load_objectlist():
    """YAMLからプロンプト一覧と物体名からカテゴリへの対応を読み込む."""
    objectlist_path = (
        Path(get_package_share_directory('teamd_tidyup_pkg'))
        / 'models'
        / OBJECTLIST_FILE
    )
    with objectlist_path.open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)

    categories = data.get('categories') if isinstance(data, dict) else None
    if not isinstance(categories, dict):
        raise ValueError('objectlist.yamlにcategoriesがありません。')

    prompts = []
    category_by_name = {}
    for category, names in categories.items():
        if not isinstance(category, str) or not isinstance(names, list):
            raise ValueError('objectlist.yamlのカテゴリ形式が不正です。')
        for name in names:
            if not isinstance(name, str) or name in category_by_name:
                raise ValueError('objectlist.yamlの物体名が不正または重複しています。')
            prompts.append(name)
            category_by_name[name] = category

    return prompts, category_by_name


def _objectlist_names(model_names, prompts):
    """同じ44クラスのモデル名をobjectlist表記へ対応付ける."""
    def key(name):
        normalized = name.lower().replace('nine', '9')
        return ''.join(character for character in normalized if character.isalnum())

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
