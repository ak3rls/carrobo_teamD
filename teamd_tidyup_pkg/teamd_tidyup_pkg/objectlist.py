"""片付けタスクで共有する物体リストと YOLOE prompt の読み込み."""

from __future__ import annotations

from pathlib import Path

import yaml


OBJECTLIST_FILE = 'objectlist.yaml'


def objectlist_path() -> Path:
    """インストール済み、またはソースツリーの物体リストへのパスを返す."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = Path(get_package_share_directory('teamd_tidyup_pkg'))
        path = share_dir / 'models' / OBJECTLIST_FILE
        if path.is_file():
            return path
    except Exception:
        # ソースツリーから単体テストや直接実行をするときは ament index が
        # 未登録の場合があるため、下のフォールバックを使います。
        pass

    return Path(__file__).resolve().parents[1] / 'models' / OBJECTLIST_FILE


def load_objectlist() -> tuple[list[str], dict[str, str]]:
    """YAMLから YOLOE prompt 一覧と物体名ごとのカテゴリを読み込む."""
    path = objectlist_path()
    with path.open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)

    categories = data.get('categories') if isinstance(data, dict) else None
    if not isinstance(categories, dict):
        raise ValueError(f'{path} に categories がありません。')

    prompts: list[str] = []
    category_by_name: dict[str, str] = {}
    for category, names in categories.items():
        if not isinstance(category, str) or not isinstance(names, list):
            raise ValueError(f'{path} のカテゴリ形式が不正です。')
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f'{path} の物体名が不正です。')
            if name in category_by_name:
                raise ValueError(f'{path} の物体名が重複しています: {name}')
            prompts.append(name)
            category_by_name[name] = category

    if not prompts:
        raise ValueError(f'{path} に物体名がありません。')
    return prompts, category_by_name


__all__ = ['OBJECTLIST_FILE', 'load_objectlist', 'objectlist_path']
