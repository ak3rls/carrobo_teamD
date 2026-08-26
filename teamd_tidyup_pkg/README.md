# carrobo_tidyup_pkg

ステートは次の順に遷移します

```text
Move2GraspPoint -> Recog -> Grasp -> Move2PlacePoint -> Place
```


## ビルドと実行

```bash
cd ~/hma2_ws
colcon_build_release_single teamd_tidyup_pkg
source install/setup.bash
ros2 launch teamd_tidyup_pkg tidyup.launch.py
```

`tidyup.launch.py` は YOLOv8 検出サービス（yolov8_detection）、把持点推定サービス（grasp_detection_point）、YASMIN
Viewer と本ステートマシンを起動します。
bring upとnavigation、Issac Simは立ち上げておいてください。

## YOLO未検出時のRex-Omni最終確認

`Recog`はYOLOで3回連続して物体を検出できなかった場合、同じ視点の最新1フレームを
Rex-Omniへ`object`プロンプトで渡します。Rex-Omniも0件なら`none`を返し、
物体があればSAM2のmaskとdepthを既存の把持点推定へ渡します。

Rex-Omniは最終確認時だけモデルをロードし、推論後に停止してVRAMを解放します。
通常は既定のweights位置を使用します。別の場所にある場合は次のように指定します。

```bash
ros2 launch teamd_tidyup_pkg tidyup.launch.py \
  rex_omni_weights_dir:=/path/to/weights/Rex-Omni
```

Rex-Omniを起動しない調整時は`use_rex_omni:=false`を指定できます。ただし、その状態で
YOLOが3回連続未検出になると、最終確認サービスがないため`Recog`は`failed`を返します。

**Rex-Omniのtokenizer不整合について（2026-08-26に修正済み）**: HuggingFace配布の
`IDEA-Research/Rex-Omni`の`vocab.json`/`merges.txt`は、ベースモデル
（`Qwen/Qwen2.5-VL-7B-Instruct`）のものと同一で、独自のbin座標トークン
（`<0>`〜`<999>`）用に語彙を差し替えるべき箇所が未編集のまま配布されていた。
このため`AutoTokenizer`で読み込むと`<|im_end|>`等の特殊トークンが埋め込み行列
（151936行）の範囲を超えるIDに再割当てされ、推論時に
`CUDA error: device-side assert triggered`でクラッシュしていた。

対処として、ダウンロード先`weights/Rex-Omni/`内の`vocab.json`・`merges.txt`から、
`tokenizer_config.json`の`added_tokens_decoder`が想定するID範囲
（150643〜151664）と衝突する末尾1000件（稀少なUnicode BPEトークン）を除去済み
（元ファイルは同ディレクトリに`*.orig_backup`として保存）。この修正は
`download_rex_omni_weights`で再ダウンロードすると失われるため、再取得時は再度
同じ手順（末尾1000件の除去）を行う必要がある。

## SAM2による引き出し取っ手検出

`drawer_handle_detector`はGroundingDINOへ取っ手名をプロンプトとして渡し、
そのbboxをSAM2でsegmentします。マスク内のdepthから得た3D中心は
`base_link`へTF変換され、次のインターフェースへ出力されます。

- サービス: `/drawer_handle/detect`
  (`hma_object_detection2_interfaces/srv/ObjectDetectionService`)
- 選択候補の可視化用姿勢: `/drawer_handle/pose`
  (`geometry_msgs/PoseStamped`)
- SAM2の重畳画像: `/grounding_dino_sam2/result_image/compressed`

SAM2.1 Hiera Tinyのcheckpointを既定の場所へ用意します。

```bash
mkdir -p ~/hma2_ws/src/5_perception/hma_object_detection2/hma_object_detection2/models/sam2
wget -P ~/hma2_ws/src/5_perception/hma_object_detection2/hma_object_detection2/models/sam2 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

検出だけを確認する場合は次を実行します。

```bash
ros2 launch teamd_tidyup_pkg drawer_handle_sam.launch.py
ros2 service call /drawer_handle/detect \
  hma_object_detection2_interfaces/srv/ObjectDetectionService \
  "{confidence_th: 0.2, iou_th: 0.0, use_latest_image: true, max_distance: 1.5, specific_id: 'drawer handle . cabinet handle'}"
```

drawer操作へ接続する場合は、Isaac Simをリセットした後に明示的に有効化します。
各drawerの直前に候補を再検出し、下段・上段の想定高さに最も近い候補を選び、
検出中心の10 cm手前から手先ローカル`+Z`方向へ接近します。

```bash
ros2 launch teamd_tidyup_pkg tidyup.launch.py use_drawer_sam:=true
```

取っ手が検出できない場合は、既定では腕を動かさずdrawer stateを失敗させます。
調整時のみ`require_handle_detection:=false`で固定座標へフォールバックできます。
