# syntax=docker/dockerfile:1
# lang_sam_ros2 (lang_sam_executor.launch.py) の実行環境
#
#   ビルドコンテキストはリポジトリのルート (このファイルの 1 つ上)
#   docker compose -f docker/compose.yaml build
#
# 中身はホスト (WSL2 Ubuntu 24.04) の構成をそのまま再現している:
#   - ROS 2 Jazzy + usb_cam + cv_bridge
#   - /opt/venv           : lang_sam_tracker ノード用 (LangSAM = GroundingDINO + SAM2.1)
#   - /root/EdgeTAM/.venv : EdgeTAM ワーカー用 (OpenVINO バックエンド、別プロセスで起動される)
#     params.yaml の edgetam_dir: ~/EdgeTAM がそのまま使えるように /root/EdgeTAM に置く

FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# torch のホイール取得先。NVIDIA GPU で LangSAM を回すなら
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# 1 なら OpenVINO の GPU (Intel iGPU/Arc) 用ランタイムを入れる
ARG INTEL_GPU=1

# ---------------------------------------------------------------------------
# apt  (libgl1 以降は opencv-python の cv2.imshow (Qt/xcb) 用)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-usb-cam \
        ros-jazzy-cv-bridge \
        ros-jazzy-geometry-msgs \
        ros-jazzy-rosidl-default-generators \
        python3-colcon-common-extensions \
        python3-venv python3-dev python3-pip \
        build-essential git wget ca-certificates software-properties-common \
        libopencv-dev v4l-utils \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxcb-shape0 libxcb-cursor0 \
    && if [ "${INTEL_GPU}" = "1" ]; then \
        add-apt-repository -y ppa:kobuk-team/intel-graphics \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
            libze-intel-gpu1 libze1 intel-opencl-icd clinfo ; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# EdgeTAM (OpenVINO) : ~/EdgeTAM と同じ構成
#   upstream 7711e01 + 自作の ov_backend.py / export_openvino.py
# ---------------------------------------------------------------------------
ARG EDGETAM_COMMIT=7711e01
WORKDIR /root/EdgeTAM
RUN git clone https://github.com/facebookresearch/EdgeTAM.git . \
    && git checkout ${EDGETAM_COMMIT} \
    && rm -rf .git \
    && test "$(stat -c%s checkpoints/edgetam.pt)" -gt 1000000

COPY docker/requirements-edgetam.txt /tmp/requirements-edgetam.txt
RUN python3 -m venv .venv \
    && .venv/bin/pip install --upgrade pip setuptools wheel \
    && .venv/bin/pip install --index-url ${TORCH_INDEX_URL} torch==2.14.1 torchvision==0.29.1 \
    && .venv/bin/pip install -r /tmp/requirements-edgetam.txt \
    && SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation --no-deps -e .

COPY docker/edgetam/ov_backend.py docker/edgetam/export_openvino.py ./
# OpenVINO IR (ov_models/*.xml) をビルド時に作っておく。デバイス非依存なので CPU で変換してよい
# (GPU 用のコンパイルキャッシュ ov_models/cache は実行時に volume へ作られる)
RUN PYTHONPATH=. .venv/bin/python export_openvino.py

# ---------------------------------------------------------------------------
# LangSAM 用 venv : 開発環境の .venv と同じ構成
#   ROS の apt パッケージ (rclpy の依存など) も見えるよう --system-site-packages
# ---------------------------------------------------------------------------
ARG SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
COPY docker/requirements-langsam.txt /tmp/requirements-langsam.txt
RUN python3 -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --index-url ${TORCH_INDEX_URL} torch==2.14.0 torchvision==0.29.0 \
    && /opt/venv/bin/pip install -r /tmp/requirements-langsam.txt \
    && SAM2_BUILD_CUDA=0 /opt/venv/bin/pip install --no-build-isolation --no-deps \
        "git+https://github.com/facebookresearch/sam2.git@${SAM2_COMMIT}"
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# ---------------------------------------------------------------------------
# モデルの重みを焼き込む (起動のたびにダウンロードしない / オフラインでも動く)
#   GroundingDINO: IDEA-Research/grounding-dino-base (Hugging Face)
#   SAM2.1       : params.yaml の sam_model (torch.hub キャッシュ)
# ---------------------------------------------------------------------------
ARG SAM_MODEL=sam2.1_hiera_small
ENV HF_HOME=/opt/models/hf \
    TORCH_HOME=/opt/models/torch
RUN python - <<EOF
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import torch
AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
torch.hub.load_state_dict_from_url(
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM_MODEL}.pt", map_location="cpu")
EOF
# EdgeTAM の画像エンコーダ (timm repvit) も起動時に Hugging Face から重みを取りに行くので焼き込む
RUN /root/EdgeTAM/.venv/bin/python -c \
    "import timm; timm.create_model('repvit_m1.dist_in1k', pretrained=True)"

# ---------------------------------------------------------------------------
# ROS 2 ワークスペース
# ---------------------------------------------------------------------------
WORKDIR /ws
COPY . ./src/lang_sam_ros2
RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --event-handlers console_direct+ \
    && rm -rf build log

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "lang_sam_executor", "lang_sam_executor.launch.py"]
