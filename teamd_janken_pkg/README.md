# carrobo_tidyup_pkg

ステートは次の順に遷移します

```text
Move2GraspPoint -> Recog -> Grasp -> Move2PlacePoint -> Place
```


## ビルドと実行

```bash
cd ~/hma2_ws
colcon_build_release_single teamd_janken_pkg
source install/setup.bash
ros2 launch teamd_janken_pkg tidyup.launch.py
```

`tidyup.launch.py` は YOLOv8 検出サービス（yolov8_detection）、把持点推定サービス（grasp_detection_point）、YASMIN
Viewer と本ステートマシンを起動します。
bring upとnavigation、Issac Simは立ち上げておいてください。

## 手認識ノード

MediaPipe は事前に Python 環境へインストールしてください。

```bash
python3 -m pip install mediapipe
cd ~/hma2_ws
colcon build --packages-select teamd_janken_pkg --symlink-install
source install/setup.bash
ros2 run teamd_janken_pkg hand_recog
```

別のカメラを使う場合は `camera_index` を変更できます。GUI を表示できない
環境では `show_image` を `false` にしてください。

```bash
ros2 run teamd_janken_pkg hand_recog --ros-args \
  -p camera_index:=1 -p show_image:=false
```
