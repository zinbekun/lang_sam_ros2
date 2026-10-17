"""EdgeTAM の重い部分 (画像エンコーダ / メモリアテンション / メモリエンコーダ) を
OpenVINO で実行するように predictor を差し替える。事前に export_openvino.py を実行しておくこと。

    predictor = build_sam2_video_predictor(...)
    use_openvino(predictor, device="GPU")
"""
import glob
import os
import time

import openvino as ov
import torch


class OpenVINOBackend:
    def __init__(
        self, predictor, model_dir="ov_models", device="GPU", precision="f16", torch_threads=4
    ):
        # PyTorch に残る処理 (マスクデコーダ周りの後処理など) と OpenVINO が CPU を取り合うと
        # 遅くなるので、PyTorch のスレッド数を絞る (i7-1360P では 4 が最速だった)
        torch.set_num_threads(torch_threads)
        core = ov.Core()
        # コンパイル結果をキャッシュして 2 回目以降の起動を速くする
        core.set_property({"CACHE_DIR": os.path.join(model_dir, "cache")})
        cfg = {"INFERENCE_PRECISION_HINT": precision, "PERFORMANCE_HINT": "LATENCY"}

        t = time.time()
        self.image_encoder = core.compile_model(f"{model_dir}/image_encoder.xml", device, cfg)
        self.memory_encoder = core.compile_model(f"{model_dir}/memory_encoder.xml", device, cfg)
        self.mask_decoder = core.compile_model(f"{model_dir}/mask_decoder.xml", device, cfg)
        self.mask_decoder_sparse_len = self.mask_decoder.inputs[2].get_shape()[1]
        # メモリ長 (入力 memory の 0 次元) -> コンパイル済みモデル
        self.memory_attention = {}
        for path in glob.glob(f"{model_dir}/memory_attention_k*.xml"):
            model = core.read_model(path)
            mem_len = model.inputs[1].get_partial_shape()[0].get_length()
            self.memory_attention[mem_len] = core.compile_model(model, device, cfg)
        if not self.memory_attention:
            raise FileNotFoundError(f"{model_dir} にモデルがありません。export_openvino.py を実行してください")
        print(f"OpenVINO ({device}, {precision}) ready in {time.time() - t:.1f}s")

        self.p = predictor
        self.torch_memory_attention = predictor.memory_attention.forward
        predictor.forward_image = self.forward_image
        predictor.memory_attention.forward = self.forward_memory_attention
        predictor.memory_encoder.forward = self.forward_memory_encoder
        self.torch_predict_masks = predictor.sam_mask_decoder.predict_masks
        predictor.sam_mask_decoder.predict_masks = self.predict_masks

    @staticmethod
    def _np(x):
        return x.detach().float().cpu().numpy()

    @staticmethod
    def _torch(outputs, device):
        return [torch.from_numpy(outputs[i]).to(device) for i in range(len(outputs))]

    def forward_image(self, img_batch):
        out = self._torch(self.image_encoder(self._np(img_batch)), img_batch.device)
        n = len(out) // 2
        return {
            "vision_features": out[n - 1],
            "vision_pos_enc": out[n:],
            "backbone_fpn": out[:n],
        }

    def forward_memory_attention(
        self, curr, memory, curr_pos=None, memory_pos=None,
        num_obj_ptr_tokens=0, num_spatial_mem=-1,
    ):
        model = self.memory_attention.get(memory.shape[0])
        c = curr[0] if isinstance(curr, list) else curr
        if model is None or c.shape[1] != 1:  # 想定外の形 (複数物体など) は PyTorch で計算
            return self.torch_memory_attention(
                curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens, num_spatial_mem
            )
        cp = curr_pos[0] if isinstance(curr_pos, list) else curr_pos
        out = model([self._np(c), self._np(memory), self._np(cp), self._np(memory_pos)])
        return torch.from_numpy(out[0]).to(memory.device)

    def predict_masks(
        self, image_embeddings, image_pe, sparse_prompt_embeddings,
        dense_prompt_embeddings, repeat_image, high_res_features=None,
    ):
        # ボックス入力などでプロンプト数が違う初期フレームは PyTorch で計算
        if (
            sparse_prompt_embeddings.shape[:2] != (1, self.mask_decoder_sparse_len)
            or image_embeddings.shape[0] != 1
            or repeat_image
        ):
            return self.torch_predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                repeat_image=repeat_image,
                high_res_features=high_res_features,
            )
        inputs = [image_embeddings, image_pe, sparse_prompt_embeddings,
                  dense_prompt_embeddings, *high_res_features]
        out = self.mask_decoder([self._np(x) for x in inputs])
        return tuple(self._torch(out, image_embeddings.device))

    def forward_memory_encoder(self, pix_feat, masks, skip_mask_sigmoid=False):
        if not skip_mask_sigmoid:
            masks = torch.sigmoid(masks)
        feat, pos = self._torch(
            self.memory_encoder([self._np(pix_feat), self._np(masks)]), pix_feat.device
        )
        return {"vision_features": feat, "vision_pos_enc": [pos]}


def use_openvino(predictor, **kwargs):
    return OpenVINOBackend(predictor, **kwargs)
