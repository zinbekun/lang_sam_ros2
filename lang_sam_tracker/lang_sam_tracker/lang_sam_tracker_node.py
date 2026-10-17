#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage

from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
import numpy as np
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from lang_sam import LangSAM
from lang_sam.utils import draw_image  # 可視化ユーティリティ
from lang_sam.models.utils import DEVICE  # 推論デバイス（cuda/cpu取得）
from lang_sam_msgs.msg import TrackArray, Track # カスタムメッセージ
from lang_sam_tracker.edgetam_client import EdgeTAMClient  # EdgeTAM追跡(別プロセス)

class LangSamTrackerNode(Node):
    def __init__(self):
        super().__init__('lang_sam_tracker')
        self.logger = self.get_logger()
        self.logger.info('Initializing LangSAM Tracker Node...')

        # パラメータ宣言・取得
        self._setup_parameters()

        # 使用デバイス（外部ユーティリティから自動選択）
        self.device = DEVICE

        # LangSAMモデルのロード（GroundingDINO+SAMの複合推論）
        self.model = LangSAM()

        # ROS <-> OpenCV画像変換
        self.bridge = CvBridge()

        # KLTトラッキング状態
        self.tracks = []
        self.prev_gray = None
        self.next_track_id = 0
        self.latest_bgr = None  # タイマー検出用に最新フレームを保持

        # EdgeTAM追跡: 検出スレッドから渡された初期化要求 (image_callback側で処理)
        self.pending_init = None
        self.edgetam = None
        if self.tracker_backend == 'edgetam':
            self.logger.info(f'Starting EdgeTAM worker ({self.edgetam_backend}, {self.edgetam_ov_device})...')
            self.edgetam = EdgeTAMClient(
                self.edgetam_dir,
                python=self.edgetam_python or None,
                backend=self.edgetam_backend,
                ov_device=self.edgetam_ov_device,
                ov_precision=self.edgetam_ov_precision,
            )
            self.logger.info(f'EdgeTAM worker ready: {self.edgetam.info}')

        # 共有状態ロックと検出用スレッドプール
        self.state_lock = threading.Lock()
        self.detector_pool = ThreadPoolExecutor(max_workers=1)
        # バックグラウンド検出の状態: None または Future
        self.det_future = None

        # I/O: 入力画像サブスク / 出力画像パブリッシュ
        self.image_sub = self.create_subscription(ROSImage, self.image_topic, self.image_callback, 1)
        # ...既存のトピックパブリッシャー作成は削除: image_detection_pub / image_tracking_pub は使用しない...
        self.tracks_pub = self.create_publisher(TrackArray, '/lang_sam/tracks', 1)

        # 可視化用共有イメージ（検出スレッド -> 表示）
        self.latest_det_vis = None  # OpenCV BGR image or None

        # 表示FPS用（トラッキング）
        # トラッキングFPS計測
        self.track_last_time = time.time()
        self.track_fps = 0.0

        # 平滑化係数 (EMA) を両方で共通化
        self.fps_alpha = 0.1
        self.det_last_time = None
        self.det_fps = 0.0

        # 検出はタイマーで実行（detection_interval_secを周期として使用）
        self.detection_timer = self.create_timer(float(self.detection_interval_sec), self.timer_callback)

        # ログ
        self.get_logger().info(f'Using device: {self.device}')
        self.get_logger().info(f'Using SAM model: {self.sam_model}')
        self.get_logger().info(f'Using text prompt: {self.text_prompt}')
        self.get_logger().info(f'Detection interval (sec): {self.detection_interval_sec}')
        self.get_logger().info(f'Image topic: {self.image_topic}')
        self.get_logger().info(f'Tracker backend: {self.tracker_backend}')
        self.get_logger().info('LangSAM model initialized.')

    # パラメータ取得用の関数
    def _setup_parameters(self):
        # 注意: launch/config側で上書き可能
        self.declare_parameter('sam_model', 'sam2.1_hiera_small')
        self.declare_parameter('text_prompt', 'wheel. car.')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('detection_interval_sec', 2.0)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('tracker_backend', 'klt')     # 'klt' or 'edgetam'

        # EdgeTAMのROSパラメータ (tracker_backend: edgetam のとき使用)
        self.declare_parameter('edgetam_dir', '~/EdgeTAM')
        self.declare_parameter('edgetam_python', '')          # 空なら <edgetam_dir>/.venv/bin/python
        self.declare_parameter('edgetam_backend', 'openvino') # 'openvino' or 'torch'
        self.declare_parameter('edgetam_ov_device', 'GPU')
        self.declare_parameter('edgetam_ov_precision', 'f16')
        self.declare_parameter('edgetam_max_objects', 3)
        self.declare_parameter('edgetam_init_with_mask', True)

        # KLT(LK光学フロー)のROSパラメータ
        # - 窓サイズ、ピラミッド段数、収束条件、最低存続点数
        self.declare_parameter('klt_win_size', [15, 15])      # integer_array [w, h]
        self.declare_parameter('klt_max_level', 3)            # integer
        self.declare_parameter('klt_criteria_count', 30)      # integer
        self.declare_parameter('klt_criteria_eps', 0.03)      # double
        self.declare_parameter('klt_min_points', 5)           # integer: 維持すべき最小追跡点数

        # GFTT(Shi-Tomasi)のROSパラメータ
        self.declare_parameter('gftt_max_corners', 120)       # integer
        self.declare_parameter('gftt_quality_level', 0.01)    # double
        self.declare_parameter('gftt_min_distance', 3.0)      # double(画素)

        self.sam_model = self.get_parameter('sam_model').get_parameter_value().string_value
        self.text_prompt = self.get_parameter('text_prompt').get_parameter_value().string_value
        self.box_threshold = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        self.detection_interval_sec = self.get_parameter('detection_interval_sec').get_parameter_value().double_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.tracker_backend = self.get_parameter('tracker_backend').get_parameter_value().string_value
        if self.tracker_backend not in ('klt', 'edgetam'):
            raise ValueError(f"tracker_backend は 'klt' か 'edgetam': {self.tracker_backend}")

        self.edgetam_dir = self.get_parameter('edgetam_dir').get_parameter_value().string_value
        self.edgetam_python = self.get_parameter('edgetam_python').get_parameter_value().string_value
        self.edgetam_backend = self.get_parameter('edgetam_backend').get_parameter_value().string_value
        self.edgetam_ov_device = self.get_parameter('edgetam_ov_device').get_parameter_value().string_value
        self.edgetam_ov_precision = self.get_parameter('edgetam_ov_precision').get_parameter_value().string_value
        self.edgetam_max_objects = int(self.get_parameter('edgetam_max_objects').get_parameter_value().integer_value)
        self.edgetam_init_with_mask = self.get_parameter('edgetam_init_with_mask').get_parameter_value().bool_value

        # KLTパラメータの取得と整形
        ws = self.get_parameter('klt_win_size').get_parameter_value().integer_array_value
        self.klt_win_size = (int(ws[0]), int(ws[1])) if len(ws) >= 2 else (15, 15)
        self.klt_max_level = int(self.get_parameter('klt_max_level').get_parameter_value().integer_value)
        self.klt_criteria_count = int(self.get_parameter('klt_criteria_count').get_parameter_value().integer_value)
        self.klt_criteria_eps = float(self.get_parameter('klt_criteria_eps').get_parameter_value().double_value)
        self.klt_min_points = int(self.get_parameter('klt_min_points').get_parameter_value().integer_value)

        # GFTTパラメータの取得
        self.gftt_max_corners = int(self.get_parameter('gftt_max_corners').get_parameter_value().integer_value)
        self.gftt_quality_level = float(self.get_parameter('gftt_quality_level').get_parameter_value().double_value)
        self.gftt_min_distance = float(self.get_parameter('gftt_min_distance').get_parameter_value().double_value)

    # --- tensor/torch -> numpy 変換 ---
    def _to_numpy(self, x):
        if x is None:
            return None
        if hasattr(x, 'cpu'):
            try:
                x = x.cpu().numpy()
            except Exception:
                # 非 tensor オブジェクトなどをそのまま配列化
                pass
        return np.asarray(x)

    # --- EMA による FPS 更新の共通処理 ---
    def _update_ema_fps(self, now, last_name, fps_name):
        last = getattr(self, last_name, None)
        fps = getattr(self, fps_name, 0.0)
        if last is None:
            setattr(self, last_name, now)
            if fps == 0.0:
                setattr(self, fps_name, 0.0)
            return getattr(self, fps_name)
        dt = now - last
        if dt > 1e-6:
            inst = 1.0 / dt
            new = (1.0 - self.fps_alpha) * fps + self.fps_alpha * inst if fps > 0 else inst
            setattr(self, fps_name, new)
        setattr(self, last_name, now)
        return getattr(self, fps_name)

    def _init_tracks_from_detections(self, cv_image, boxes, labels, scores, masks_bool):
        # 検出結果からトラック群を初期化
        # - マスク領域からgoodFeaturesToTrackでKLTの初期点をサンプリング
        # - bbox/label/score/maskをtrack辞書に格納
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.tracks = []
        h, w = gray.shape
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]

            # マスクが無い検出はトラックを生成しない（疑似マスクは作らない設計）
            if i >= masks_bool.shape[0] or masks_bool[i].dtype != bool or not masks_bool[i].any():
                continue
            mask_uint8 = (masks_bool[i].astype(np.uint8)) * 255  # goodFeaturesToTrackがuint8マスクを要求

            # GFTTパラメータをROSから取得した値で適用
            pts = cv2.goodFeaturesToTrack(
                image=gray,
                maxCorners=self.gftt_max_corners,
                qualityLevel=self.gftt_quality_level,
                minDistance=self.gftt_min_distance,
                mask=mask_uint8
            )
            # 最低点数をROSパラメータで判定
            if pts is None or pts.shape[0] < self.klt_min_points:
                continue

            track = {
                'id': self.next_track_id,
                'label': labels[i] if i < len(labels) else 'obj',
                'score': float(scores[i]) if i < len(scores) else 1.0,
                'points': pts,              # (N,1,2) float32
                'box': [x1, y1, x2, y2],
                'mask': masks_bool[i]       # 検出時点のマスク（bool, HxW）
            }
            self.tracks.append(track)
            self.next_track_id += 1

    def _mask_from_points(self, points, shape):
        # KLT更新後の特徴点群から各特徴点をそのまま描画してマスクを再構成
        # points: (M,1,2) または (M,2)、shape: (H,W)
        if points is None:
            return np.zeros(shape, dtype=bool)

        # 正規化して (N,2) 形状にする
        try:
            pts = points.reshape(-1, 2)
        except Exception:
            return np.zeros(shape, dtype=bool)

        if pts.shape[0] < 1:
            return np.zeros(shape, dtype=bool)

        mask = np.zeros(shape, dtype=np.uint8)
        # 点を描画する半径（ピクセル）。必要ならパラメータ化可能。
        radius = 5
        thickness = -1  # 塗りつぶし
        for (x_f, y_f) in pts:
            x = int(np.round(x_f))
            y = int(np.round(y_f))
            # 画像境界内のみ描画
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                cv2.circle(mask, (x, y), radius, 255, thickness)

        return mask.astype(bool)

    def _update_tracks_with_klt(self, cv_image):
        # 直前(prev_gray)と現在フレームの間でピラミッドLKを計算し、各トラックの特徴点/マスク/bboxを更新
        if self.prev_gray is None or len(self.tracks) == 0:
            return
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        new_tracks = []
        # KLTパラメータをROSから取得した値で適用
        lk_params = dict(
            winSize=tuple(map(int, self.klt_win_size)),
            maxLevel=int(self.klt_max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      int(self.klt_criteria_count),
                      float(self.klt_criteria_eps))
        )
        h, w = gray.shape
        for track in self.tracks:
            pts = track['points'].astype(np.float32)  # (N,1,2)
            new_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, **lk_params)
            if new_pts is None or st is None:
                continue

            # 追跡成功点のみ抽出
            st_flat = st.flatten().astype(bool)
            good = new_pts[st_flat]

            # 形状を(N,2)に正規化
            if good.ndim == 3 and good.shape[1] == 1 and good.shape[2] == 2:
                good_xy = good[:, 0, :]
            elif good.ndim == 2 and good.shape[1] == 2:
                good_xy = good
            else:
                try:
                    good_xy = good.reshape(-1, 2)
                except Exception:
                    continue

            # 最低点数をROSパラメータで判定
            if good_xy.shape[0] < self.klt_min_points:
                continue

            # bboxは特徴点のmin/maxから更新（画像境界でクリップ）
            x_min = int(np.clip(np.min(good_xy[:, 0]), 0, w - 1))
            y_min = int(np.clip(np.min(good_xy[:, 1]), 0, h - 1))
            x_max = int(np.clip(np.max(good_xy[:, 0]), 0, w - 1))
            y_max = int(np.clip(np.max(good_xy[:, 1]), 0, h - 1))

            # 特徴点の凸包からマスクを再構成（可視化で利用）
            good_pts = good_xy.reshape(-1, 1, 2).astype(np.float32)
            mask_bool = self._mask_from_points(good_pts, (h, w))

            # トラック更新
            track['points'] = good_pts
            track['box'] = [x_min, y_min, x_max, y_max]
            track['mask'] = mask_bool
            new_tracks.append(track)

        # 次フレームのKLTに備えてprev_gray更新
        self.tracks = new_tracks
        self.prev_gray = gray

    def _update_tracks_with_edgetam(self, cv_image):
        # 検出結果があればEdgeTAMを検出フレームで初期化し、現在フレームまで追跡してマスク/bboxを更新
        # ワーカーとの通信は重いのでロック外で行う (ワーカーを触るのはこのスレッドだけ)
        with self.state_lock:
            pending, self.pending_init = self.pending_init, None
            tracks = self.tracks

        try:
            if pending is not None:
                det_image, boxes, labels, scores, masks_bool = pending
                # マスクのある検出だけをスコア順に最大 edgetam_max_objects 個追跡する
                idx = [i for i in range(len(boxes)) if i < masks_bool.shape[0] and masks_bool[i].any()]
                idx = sorted(idx, key=lambda i: -(scores[i] if i < len(scores) else 0.0))
                idx = idx[:max(1, self.edgetam_max_objects)]
                # 検出0件 (対象が画面外など) のときは初期化せず、EdgeTAMの追跡を続ける
                if idx:
                    tracks = []
                    init_masks = [masks_bool[i] for i in idx] if self.edgetam_init_with_mask else None
                    self.edgetam.init(det_image, [boxes[i] for i in idx], init_masks)
                    for i in idx:
                        tracks.append({
                            'id': self.next_track_id,
                            'label': labels[i] if i < len(labels) else 'obj',
                            'score': float(scores[i]) if i < len(scores) else 1.0,
                            'box': [int(v) for v in boxes[i]],
                            'mask': masks_bool[i],
                            'visible': True,
                        })
                        self.next_track_id += 1

            if tracks:
                for track, (mask, visible) in zip(tracks, self.edgetam.track(cv_image)):
                    # 隠れた/画面外の物体もトラックに残して追跡を続ける (EdgeTAMのメモリで再び出てきたら復帰する)
                    # 見えていない間は表示・配信しない
                    track['visible'] = bool(visible and mask.any())
                    if track['visible']:
                        ys, xs = np.where(mask)
                        track['box'] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                        track['mask'] = mask
        except RuntimeError as e:
            self.get_logger().error(f'EdgeTAM追跡で例外: {e}')
            tracks = []

        with self.state_lock:
            self.tracks = tracks

    def image_callback(self, msg):
        # 入力: ROS Image -> OpenCV(BGR)
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # タイマー検出用に最新フレームを保持
        self.latest_bgr = cv_image

        if self.edgetam is not None:
            self._update_tracks_with_edgetam(cv_image)

        # KLT更新は共有状態を保護
        with self.state_lock:
            if self.edgetam is None:
                self._update_tracks_with_klt(cv_image)
            # 可視化用にスナップショットを作る（ロック時間を短くするため必要最小限をコピー）
            visible_tracks = [t for t in self.tracks if t.get('visible', True)]
            if visible_tracks:
                boxes_for_draw = np.asarray([t['box'] for t in visible_tracks], dtype=np.int32)
                labels_for_draw = [t['label'] for t in visible_tracks]
                scores_for_draw = np.asarray([t['score'] for t in visible_tracks], dtype=np.float32)
                masks_for_draw = np.asarray([t['mask'] for t in visible_tracks], dtype=bool)
            else:
                h, w, _ = cv_image.shape
                boxes_for_draw = np.zeros((0, 4), dtype=np.int32)
                labels_for_draw = []
                scores_for_draw = np.zeros((0,), dtype=np.float32)
                masks_for_draw = np.zeros((0, h, w), dtype=bool)

        pil_base = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        try:
            track_image_pil = draw_image(
                image_rgb=pil_base,
                masks=masks_for_draw,
                xyxy=boxes_for_draw,
                probs=scores_for_draw,
                labels=labels_for_draw,
            )
        except Exception as e:
            self.get_logger().warning(f'draw_image失敗(tracking): {e}')
            return

        track_image_cv = cv2.cvtColor(np.array(track_image_pil), cv2.COLOR_RGB2BGR)

        # --- 検出可視化と追跡可視化を横並びで表示（FPSを描画） ---
        # トラッキングFPS更新（image_callback の呼び出し周期を利用）
        now = time.time()
        # 共通ヘルパーでトラックFPSを更新
        self._update_ema_fps(now, 'track_last_time', 'track_fps')

        # 検出可視化をスレッドセーフに取得（同時にdet_fpsも読み出す）
        with self.state_lock:
            det_vis = None if self.latest_det_vis is None else self.latest_det_vis.copy()
            det_fps = float(self.det_fps)

        # 検出画像がない場合は空白を作る
        if det_vis is None:
            h_t, w_t, _ = track_image_cv.shape
            det_vis = np.zeros((h_t, w_t, 3), dtype=np.uint8)
            cv2.putText(det_vis, 'No detection yet', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        # --- 検出FPSをdet_vis上に描画（スタイルを統一） ---
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_det = max(0.6, det_vis.shape[1] / 1000.0)
            thickness_det = 2
            det_text = f'FPS: {det_fps:.1f}'
            (tw, th), _ = cv2.getTextSize(det_text, font, font_scale_det, thickness_det)
            x_det = max(10, det_vis.shape[1] - 10 - tw)  # 右端に寄せ、最小余白を確保
            y_det = 10 + th  # 上から少し下げて描画
            cv2.putText(det_vis, det_text, (x_det, y_det), font, font_scale_det, (0, 255, 255), thickness_det, cv2.LINE_AA)
        except Exception:
            pass

        # --- トラッキングFPSをtrack_image_cv上に描画（同じスタイル） ---
        try:
            font_scale_trk = max(0.6, track_image_cv.shape[1] / 1000.0)
            thickness_trk = 2
            trk_text = f'FPS: {self.track_fps:.1f}'
            (tw_t, th_t), _ = cv2.getTextSize(trk_text, font, font_scale_trk, thickness_trk)
            x_trk = max(10, track_image_cv.shape[1] - 10 - tw_t)
            y_trk = 10 + th_t
            cv2.putText(track_image_cv, trk_text, (x_trk, y_trk), font, font_scale_trk, (0, 255, 255), thickness_trk, cv2.LINE_AA)
        except Exception:
            pass

        # サイズ合わせ（高さを基準に揃える）
        h_det, w_det, _ = det_vis.shape
        h_trk, w_trk, _ = track_image_cv.shape
        if h_det != h_trk:
            scale = h_trk / h_det
            new_w = int(w_det * scale)
            det_vis = cv2.resize(det_vis, (new_w, h_trk))
            w_det = new_w

        # 横並び合成（左右にラベル）
        combined = np.hstack([det_vis, track_image_cv])

        # ラベル描画: 左=Detection, 右=Tracking
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, combined.shape[1] / 1000.0)
        thickness = 2
        # Detectionラベル位置
        cv2.putText(combined, 'Detection', (10, 30), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        # Trackingラベル位置（右側の画像の左端を計算）
        x_right = det_vis.shape[1] + 10
        cv2.putText(combined, 'Tracking', (x_right, 30), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        # FPS描画（右上）
        # fps_text = f'FPS: {self.fps:.1f}'
        # cv2.putText(combined, fps_text, (combined.shape[1] - 200, 30), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        # （注）個別画像上にFPSを描画済みのため、合成後の汎用FPS描画は不要

        # 非ブロッキング表示
        cv2.imshow('LangSAM', combined)
        cv2.waitKey(1)

        # トラック情報を/custom_msgs/TrackArrayで配信（既存）
        msg_tracks = TrackArray()
        msg_tracks.header.stamp = self.get_clock().now().to_msg()
        msg_tracks.header.frame_id = 'camera'
        with self.state_lock:
            for t in self.tracks:
                if not t.get('visible', True):
                    continue
                tr = Track()
                tr.id = int(t['id'])
                tr.label = str(t['label'])
                tr.score = float(t['score'])
                x1, y1, x2, y2 = t['box']
                tr.x_min = int(x1); tr.y_min = int(y1)
                tr.x_max = int(x2); tr.y_max = int(y2)
                msg_tracks.tracks.append(tr)
        self.tracks_pub.publish(msg_tracks)

    def timer_callback(self):
        # 最新フレームがなければスキップ
        if self.latest_bgr is None:
            return
        # 既にバックグラウンド推論が走っていれば重複起動しない (Futureベース)
        if self.det_future is not None and not self.det_future.done():
            return

        frame = self.latest_bgr.copy()
        # バックグラウンドで推論・描画・トラック初期化
        self.det_future = self.detector_pool.submit(self._run_detection_job, frame)

    def _run_detection_job(self, cv_image):
        try:
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                # 元の重い処理（GPU/CPU同期もここで実施）
                results = self.model.predict([pil_image], [self.text_prompt])

            det = results[0]

            # boxes/masks/scores/labels を numpy に正規化
            boxes = self._to_numpy(det.get('boxes', None))
            boxes_np = np.zeros((0, 4), dtype=np.float32) if boxes is None else np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

            h, w, _ = cv_image.shape
            masks = self._to_numpy(det.get('masks', None))
            if masks is None:
                masks_np = np.zeros((boxes_np.shape[0], h, w), dtype=bool)
            else:
                masks_np = np.asarray(masks)
                # normalize dims: expect (N, H, W) or (N,1,H,W)
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np[:, 0]
                if masks_np.ndim == 2:
                    masks_np = masks_np[0:1, ...]
                if masks_np.ndim != 3:
                    masks_np = np.zeros((boxes_np.shape[0], h, w), dtype=bool)
                masks_np = masks_np.astype(bool)

            labels = det.get('labels', [])
            labels_det = [str(l) for l in labels] if len(labels) > 0 else []

            scores = self._to_numpy(det.get('scores', None))
            scores_np = np.zeros((boxes_np.shape[0],), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32).reshape(-1)

            # 検出可視化を作成
            det_image_pil = draw_image(
                image_rgb=pil_image,
                masks=masks_np,
                xyxy=boxes_np,
                probs=scores_np,
                labels=labels_det,
            )
            det_image_cv = cv2.cvtColor(np.array(det_image_pil), cv2.COLOR_RGB2BGR)

            # 検出完了時刻でdet_fpsを更新（スレッドセーフ、共通alphaを使用）および可視化/トラック初期化
            now = time.time()
            with self.state_lock:
                # 更新
                self._update_ema_fps(now, 'det_last_time', 'det_fps')
                # 可視化保存 (コピーして共有)
                self.latest_det_vis = det_image_cv.copy()
                # トラック初期化
                if self.edgetam is not None:
                    # EdgeTAMの初期化はワーカーを使うimage_callback側で行う
                    self.pending_init = (cv_image, boxes_np.tolist(), labels_det, scores_np.tolist(), masks_np)
                else:
                    self._init_tracks_from_detections(
                        cv_image,
                        boxes_np.tolist(),
                        labels_det,
                        scores_np.tolist(),
                        masks_np
                    )
        except Exception as e:
            self.get_logger().error(f'バックグラウンド検出で例外: {e}')
        finally:
            # Future ベースなので特別にフラグを消す必要はないが参照を解放して GC を助ける
            try:
                self.det_future = None
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = LangSamTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.edgetam is not None:
            node.edgetam.close()
        node.destroy_node()
        # OpenCV ウィンドウを破棄
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
