# lang_sam_ros2 の Docker 環境

`ros2 launch lang_sam_executor lang_sam_executor.launch.py` をコンテナで動かすための環境です。
ホスト側に ROS 2 も venv も EdgeTAM も不要で、Docker だけあれば動きます。

イメージの中身:

| 場所 | 内容 |
| --- | --- |
| `/opt/ros/jazzy` | ROS 2 Jazzy (ros-base) + usb_cam + cv_bridge |
| `/opt/venv` | lang_sam_tracker ノード用 venv (torch, transformers, sam2, supervision …) |
| `/root/EdgeTAM` | EdgeTAM + `ov_backend.py` / `export_openvino.py`、`.venv` (OpenVINO)、`ov_models/` (ビルド時に変換済み) |
| `/opt/models` | GroundingDINO と SAM2.1 の重み (ビルド時にダウンロード済み、オフラインで起動可) |
| `/ws/install` | このリポジトリを colcon build したもの |

## 必要なもの

- Docker Engine + Docker Compose v2 (`docker compose`)
- x86_64 の Linux (Ubuntu 推奨) または WSL2
- ディスク空き 20GB 程度、ビルド時のメモリ 8GB 程度

## ビルド

リポジトリのルートで実行します。

```bash
git clone -b feat/edgetam-tracking https://github.com/zinbekun/lang_sam_ros2.git
cd lang_sam_ros2
docker compose -f docker/compose.yaml build
```

ビルド引数:

| 引数 | 既定値 | 説明 |
| --- | --- | --- |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cpu` | NVIDIA GPU で LangSAM を回すなら `.../whl/cu130` など |
| `INTEL_GPU` | `1` | `0` にすると Intel GPU ランタイムを入れない (OpenVINO は CPU のみ) |
| `SAM_MODEL` | `sam2.1_hiera_small` | 焼き込む SAM2.1 の重み。`params.yaml` の `sam_model` と合わせる |

例: `docker compose -f docker/compose.yaml build --build-arg INTEL_GPU=0`

## 実行

```bash
xhost +local:            # cv2.imshow のウィンドウを出すため (Linux デスクトップのみ)
docker compose -f docker/compose.yaml up
```

WSL2 の場合は WSL 用の設定を重ねます (`xhost` は不要):

```bash
docker compose -f docker/compose.yaml -f docker/compose.wsl.yaml up
```

launch 引数を付けたり、別のコマンドを実行したいとき:

```bash
docker compose -f docker/compose.yaml run --rm lang_sam \
  ros2 launch lang_sam_executor lang_sam_executor.launch.py use_usb_cam:=false
docker compose -f docker/compose.yaml run --rm lang_sam bash
```

`network_mode: host` なので、ホストや同じネットワークの PC から `ros2 topic echo /lang_sam/tracks` などがそのまま見えます
(`ROS_DOMAIN_ID` を揃えること)。

## パラメータ

`lang_sam_executor/config/params.yaml` をコンテナにマウントしているので、
編集してコンテナを再起動するだけで反映されます (再ビルド不要)。

PC ごとに見直す項目:

- `usb_cam.video_device` : カメラのデバイス (`v4l2-ctl --list-devices` で確認)
- `usb_cam.pixel_format` : `mjpeg2rgb` は WSL2(usbipd) 向け。ネイティブ Linux なら `yuyv2rgb` でもよい
- `edgetam_ov_device` : Intel GPU がない PC では `CPU` にする (`edgetam_ov_precision` も `f32` か `bf16`)
- `tracker_backend` : EdgeTAM を使わないなら `klt`

`edgetam_dir: ~/EdgeTAM` はコンテナ内の `/root/EdgeTAM` を指すのでそのままで動きます。

## ソースを変更したとき

ソースはイメージにコピーしてビルドしているので、ノードのコードを変えたら再ビルドが必要です
(重い層はキャッシュされるので colcon build の部分だけ走ります)。

```bash
docker compose -f docker/compose.yaml build
```

`~/EdgeTAM` の `ov_backend.py` / `export_openvino.py` を変えた場合は `docker/edgetam/` にコピーしてから再ビルドしてください。

## ネットに繋がらない PC へ持っていく

```bash
docker save lang_sam_ros2:jazzy | gzip > lang_sam_ros2.tar.gz   # ビルドした PC で
docker load < lang_sam_ros2.tar.gz                              # 持っていった PC で
```

持っていった PC でもこのリポジトリ (`docker/` と `params.yaml`) は必要です (`compose.yaml` が参照するため)。

## 確認用コマンド

```bash
# OpenVINO から見えるデバイス (GPU が出れば Intel GPU が使える)
docker compose -f docker/compose.yaml run --rm lang_sam \
  /root/EdgeTAM/.venv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"
# カメラ
docker compose -f docker/compose.yaml run --rm lang_sam v4l2-ctl --list-devices
```
