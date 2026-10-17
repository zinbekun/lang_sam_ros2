#!/usr/bin/env python3
"""EdgeTAM による追跡を別プロセスで実行するワーカー。

LangSAM が使う公式 sam2 と EdgeTAM の sam2 はパッケージ名が衝突するため同じプロセスに
読み込めない。そこで lang_sam_tracker_node から EdgeTAM の venv の Python でこのスクリプトを
起動し、stdin/stdout のパイプ越しにフレームとマスクをやり取りする。

    <EdgeTAM>/.venv/bin/python edgetam_worker.py --edgetam-dir <EdgeTAM> [--backend openvino ...]

プロトコル: 4byte長 + pickle(dict)。numpy のバージョン差で壊れないよう、
dict/list/bytes/int/float/bool/str だけを送る。
  -> {"cmd": "init", "frame": F, "boxes": [[x1,y1,x2,y2], ...], "masks": [M, ...] or None}
  -> {"cmd": "track", "frame": F}
  <- {"ok": True, "masks": [M, ...], "visible": [bool, ...]} / {"ok": False, "error": str}
  F = {"shape": (h, w, 3), "data": BGR uint8 bytes}, M = np.packbits(bool HxW) の bytes
"""
import argparse
import os
import pickle
import struct
import sys
import traceback

import cv2
import numpy as np


def read_msg(stream):
    head = stream.read(4)
    if len(head) < 4:
        return None
    (n,) = struct.unpack("<I", head)
    return pickle.loads(stream.read(n))


def write_msg(stream, obj):
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


class MultiObjectTracker:
    """EdgeTAM の track_step を1フレームずつ呼ぶ複数物体トラッカー。
    画像エンコーダ (一番重い部分) はフレームごとに1回だけ実行し、物体ごとにメモリを持つ。"""

    def __init__(self, predictor, device):
        import torch

        self.torch = torch
        self.p = predictor
        self.device = device
        self.size = predictor.image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # メモリとして参照されるのは直近 num_maskmem フレームと
        # 直近 max_obj_ptrs_in_encoder フレームのポインタだけなので、それ以上は捨てる
        self.keep = max(predictor.num_maskmem, predictor.max_obj_ptrs_in_encoder) + 1
        self.objects = []
        self.frame_idx = 0

    def _features(self, frame_bgr):
        torch = self.torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size))
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img = ((img - self.mean) / self.std).unsqueeze(0).to(self.device)
        backbone_out = self.p.forward_image(img)
        _, feats, pos, sizes = self.p._prepare_backbone_features(backbone_out)
        return feats, pos, sizes

    def _step_object(self, output_dict, feats, pos, sizes, point_inputs, mask_inputs, hw):
        torch = self.torch
        is_init = point_inputs is not None or mask_inputs is not None
        out = self.p.track_step(
            frame_idx=self.frame_idx,
            is_init_cond_frame=is_init,
            current_vision_feats=feats,
            current_vision_pos_embeds=pos,
            feat_sizes=sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=self.frame_idx + 1,
            track_in_reverse=False,
            run_mem_encoder=True,
        )
        key = "cond_frame_outputs" if is_init else "non_cond_frame_outputs"
        output_dict[key][self.frame_idx] = {
            "maskmem_features": out["maskmem_features"],
            "maskmem_pos_enc": out["maskmem_pos_enc"],
            "pred_masks": out["pred_masks"],
            "obj_ptr": out["obj_ptr"],
            "object_score_logits": out["object_score_logits"],
        }
        # 古いメモリを削除してメモリ使用量を一定に保つ
        output_dict["non_cond_frame_outputs"].pop(self.frame_idx - self.keep, None)

        logits = torch.nn.functional.interpolate(
            out["pred_masks"], size=hw, mode="bilinear", align_corners=False
        )
        visible = out["object_score_logits"].item() > 0  # 画面外/隠れたら False
        return (logits[0, 0] > 0).cpu().numpy(), visible

    def init(self, frame_bgr, boxes, masks=None):
        """boxes: [[x1, y1, x2, y2], ...] 元画像座標。masks があればボックスの代わりにマスクで初期化"""
        torch = self.torch
        h, w = frame_bgr.shape[:2]
        self.frame_idx = 0
        self.objects = []
        feats, pos, sizes = self._features(frame_bgr)
        results = []
        for i, box in enumerate(boxes):
            point_inputs, mask_inputs = None, None
            if masks is not None:
                m = torch.from_numpy(masks[i].astype(np.float32))[None, None]
                m = torch.nn.functional.interpolate(
                    m, size=(self.size, self.size), mode="bilinear", align_corners=False
                )
                mask_inputs = (m >= 0.5).float().to(self.device)
            else:
                coords = torch.tensor([box[:2], box[2:]], dtype=torch.float32)
                coords = coords / torch.tensor([w, h]) * self.size
                point_inputs = {
                    "point_coords": coords.unsqueeze(0).to(self.device),
                    "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=self.device),
                }
            output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            results.append(
                self._step_object(output_dict, feats, pos, sizes, point_inputs, mask_inputs, (h, w))
            )
            self.objects.append(output_dict)
        self.frame_idx += 1
        return results

    def track(self, frame_bgr):
        if not self.objects:
            return []
        h, w = frame_bgr.shape[:2]
        feats, pos, sizes = self._features(frame_bgr)
        results = [
            self._step_object(od, feats, pos, sizes, None, None, (h, w)) for od in self.objects
        ]
        self.frame_idx += 1
        return results


def decode_frame(f):
    return np.frombuffer(f["data"], dtype=np.uint8).reshape(f["shape"])


def decode_mask(data, hw):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: hw[0] * hw[1]].reshape(hw).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgetam-dir", required=True)
    ap.add_argument("--config", default="configs/edgetam.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/edgetam.pt")
    ap.add_argument("--backend", choices=["openvino", "torch"], default="openvino")
    ap.add_argument("--ov-device", default="GPU")
    ap.add_argument("--ov-precision", default="f16")
    ap.add_argument("--ov-model-dir", default="ov_models")
    args = ap.parse_args()

    # stdout はプロトコル専用にし、ライブラリの print は stderr に逃がす
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    edgetam_dir = os.path.abspath(os.path.expanduser(args.edgetam_dir))
    os.chdir(edgetam_dir)  # checkpoints/ や ov_models/ を相対パスで解決する
    sys.path.insert(0, edgetam_dir)

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)
    if args.backend == "openvino":
        from ov_backend import use_openvino

        use_openvino(
            predictor, model_dir=args.ov_model_dir,
            device=args.ov_device, precision=args.ov_precision,
        )
    tracker = MultiObjectTracker(predictor, device)
    write_msg(proto_out, {"ok": True, "ready": True, "device": str(device), "backend": args.backend})

    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        while True:
            req = read_msg(proto_in)
            if req is None or req.get("cmd") == "quit":
                break
            try:
                frame = decode_frame(req["frame"])
                hw = frame.shape[:2]
                if req["cmd"] == "init":
                    masks = req.get("masks")
                    if masks is not None:
                        masks = [decode_mask(m, hw) for m in masks]
                    results = tracker.init(frame, req["boxes"], masks)
                elif req["cmd"] == "track":
                    results = tracker.track(frame)
                else:
                    raise ValueError(f"unknown cmd: {req['cmd']}")
                write_msg(proto_out, {
                    "ok": True,
                    "masks": [np.packbits(m).tobytes() for m, _ in results],
                    "visible": [bool(v) for _, v in results],
                })
            except Exception as e:
                traceback.print_exc()
                write_msg(proto_out, {"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
