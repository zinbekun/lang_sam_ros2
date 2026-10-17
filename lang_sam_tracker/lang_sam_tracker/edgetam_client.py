"""edgetam_worker.py を別プロセスで起動し、フレームを渡してマスクを受け取るクライアント。"""
import os
import subprocess

import numpy as np

from lang_sam_tracker.edgetam_worker import read_msg, write_msg

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edgetam_worker.py')


class EdgeTAMClient:
    def __init__(self, edgetam_dir, python=None, backend='openvino',
                 ov_device='GPU', ov_precision='f16'):
        edgetam_dir = os.path.abspath(os.path.expanduser(edgetam_dir))
        python = os.path.expanduser(python) if python else os.path.join(edgetam_dir, '.venv', 'bin', 'python')
        cmd = [
            python, WORKER_PATH,
            '--edgetam-dir', edgetam_dir,
            '--backend', backend,
            '--ov-device', ov_device,
            '--ov-precision', ov_precision,
        ]
        # ROS 側 (lang_sam の venv) の PYTHONPATH には公式 sam2 が含まれ得るので引き継がない
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'VIRTUAL_ENV')}
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        # モデル読み込み・OpenVINOコンパイルが終わるまで待つ
        self.info = self._recv()
        if self.info is None:
            raise RuntimeError(f'EdgeTAMワーカーの起動に失敗しました (exit={self.proc.poll()})')

    def _recv(self):
        return read_msg(self.proc.stdout)

    def _request(self, req):
        if self.proc.poll() is not None:
            raise RuntimeError(f'EdgeTAMワーカーが終了しています (exit={self.proc.returncode})')
        write_msg(self.proc.stdin, req)
        res = self._recv()
        if res is None:
            raise RuntimeError('EdgeTAMワーカーから応答がありません')
        if not res.get('ok'):
            raise RuntimeError(res.get('error', 'unknown error'))
        return res

    @staticmethod
    def _encode_frame(frame_bgr):
        frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        return {'shape': tuple(int(s) for s in frame_bgr.shape), 'data': frame_bgr.tobytes()}

    @staticmethod
    def _decode_results(res, hw):
        n = hw[0] * hw[1]
        masks = [
            np.unpackbits(np.frombuffer(m, dtype=np.uint8))[:n].reshape(hw).astype(bool)
            for m in res['masks']
        ]
        return list(zip(masks, res['visible']))

    def init(self, frame_bgr, boxes, masks=None):
        """boxes: [[x1,y1,x2,y2], ...]、masks: bool(HxW)のリスト (指定時はマスクで初期化)。
        戻り値: [(mask bool HxW, visible bool), ...] (boxes と同じ順)"""
        req = {
            'cmd': 'init',
            'frame': self._encode_frame(frame_bgr),
            'boxes': [[float(v) for v in b] for b in boxes],
            'masks': None if masks is None else [np.packbits(m.astype(bool)).tobytes() for m in masks],
        }
        return self._decode_results(self._request(req), frame_bgr.shape[:2])

    def track(self, frame_bgr):
        req = {'cmd': 'track', 'frame': self._encode_frame(frame_bgr)}
        return self._decode_results(self._request(req), frame_bgr.shape[:2])

    def close(self):
        if self.proc.poll() is None:
            try:
                write_msg(self.proc.stdin, {'cmd': 'quit'})
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
