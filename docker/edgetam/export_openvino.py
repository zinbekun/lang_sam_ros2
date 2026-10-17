"""EdgeTAM の重い部分を OpenVINO IR に変換する。

  PYTHONPATH=. .venv/bin/python export_openvino.py

出力 (ov_models/):
  image_encoder.xml           画像エンコーダ (1x3x1024x1024)
  mask_decoder.xml            マスクデコーダ (追跡中のフレーム用)
  memory_encoder.xml          メモリエンコーダ
  memory_attention_k{1..16}.xml
      メモリアテンション。過去フレームの数 k によって入力長が変わり、
      その長さが計算グラフに焼き込まれるので k ごとに別モデルにする
      (k >= 16 以降は常に k16 を使う)
"""
import os

import openvino as ov
import torch

import sam2.modeling.sam.transformer as transformer
from sam2.build_sam import build_sam2_video_predictor

OUT_DIR = "ov_models"


# ---------------------------------------------------------------------------
# RoPE を複素数を使わない実装に置き換える (OpenVINO は complex 型を変換できない)
# freqs は [N, D/2, 2] (cos, sin) の実数テンソルとして持たせる
# ---------------------------------------------------------------------------
def _rotate(x, freqs):
    """x: [..., N, D] を (x0 + i x1) * (cos + i sin) と同じ回転をかける。"""
    shape = x.shape
    x = x.float().reshape(*shape[:-1], -1, 2)
    x0, x1 = x[..., 0], x[..., 1]
    cos, sin = freqs[..., 0], freqs[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.reshape(shape)


def apply_rotary_enc_real(xq, xk, freqs_cis, repeat_freqs_k=False):
    xq_out = _rotate(xq, freqs_cis).type_as(xq)
    if xk.shape[-2] == 0:
        return xq_out, xk
    if repeat_freqs_k:
        r = xk.shape[-2] // xq.shape[-2]
        freqs_cis = freqs_cis.repeat(r, 1, 1)
    return xq_out, _rotate(xk, freqs_cis).type_as(xk)


def apply_rotary_enc_v2_real(x, freqs_cis, repeat_freqs):
    if x.shape[-2] == 0:
        return x
    B, N_heads, N_tokens, C_per_head = x.shape
    if N_tokens == freqs_cis.shape[0] * repeat_freqs:
        x_rope, x_no_rope = x, None
    else:
        rope_tokens = freqs_cis.shape[0]
        no_rope_tokens = N_tokens // repeat_freqs - rope_tokens
        x = x.view(B, N_heads, repeat_freqs, N_tokens // repeat_freqs, C_per_head)
        x_rope = x[..., no_rope_tokens:, :].reshape(B, N_heads, -1, C_per_head)
        x_no_rope = x[..., :no_rope_tokens, :].reshape(B, N_heads, -1, C_per_head)
    if repeat_freqs > 1:
        freqs_cis = freqs_cis.repeat(repeat_freqs, 1, 1)
    x_out = _rotate(x_rope, freqs_cis).type_as(x)
    if x_no_rope is not None:
        x_out = x_out.view(B, N_heads, repeat_freqs, -1, C_per_head)
        x_no_rope = x_no_rope.view(B, N_heads, repeat_freqs, -1, C_per_head)
        x_out = torch.cat((x_no_rope, x_out), dim=3).view(
            B, N_heads, N_tokens, C_per_head
        )
    return x_out


def _attend(self, q, k, v):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return self.out_proj(self._recombine_heads(out))


def rope_attention_forward(self, q, k, v, num_k_exclude_rope=0):
    """RoPEAttention.forward から in-place のスライス代入を除いたもの。"""
    q = self._separate_heads(self.q_proj(q), self.num_heads)
    k = self._separate_heads(self.k_proj(k), self.num_heads)
    v = self._separate_heads(self.v_proj(v), self.num_heads)
    num_k_rope = k.size(-2) - num_k_exclude_rope
    q, k_rope = apply_rotary_enc_real(
        q, k[:, :, :num_k_rope], self.freqs_cis, repeat_freqs_k=self.rope_k_repeat
    )
    k = torch.cat([k_rope, k[:, :, num_k_rope:]], dim=2)
    return _attend(self, q, k, v)


def rope_attention_v2_forward(self, q, k, v, num_k_exclude_rope=0, rope_k_repeat=-1):
    """RoPEAttentionv2.forward から in-place のスライス代入を除いたもの。"""
    q = self._separate_heads(self.q_proj(q), self.num_heads)
    k = self._separate_heads(self.k_proj(k), self.num_heads)
    v = self._separate_heads(self.v_proj(v), self.num_heads)
    q = apply_rotary_enc_v2_real(q, self.freqs_cis_q, repeat_freqs=1)
    num_k_rope = k.size(-2) - num_k_exclude_rope
    k_rope = apply_rotary_enc_v2_real(
        k[:, :, :num_k_rope], self.freqs_cis_k, repeat_freqs=rope_k_repeat
    )
    k = torch.cat([k_rope, k[:, :, num_k_rope:]], dim=2)
    return _attend(self, q, k, v)


def to_real(freqs_cis):
    return torch.view_as_real(freqs_cis).contiguous() if freqs_cis.is_complex() else freqs_cis


def patch_rope_for_export(model):
    for m in model.modules():
        if isinstance(m, transformer.RoPEAttention):
            # 64x64 特徴 (1024 入力) 用の周波数を実数で持たせる
            m.freqs_cis = to_real(m.compute_cis(end_x=64, end_y=64))
            m.forward = rope_attention_forward.__get__(m)
        elif isinstance(m, transformer.RoPEAttentionv2):
            m.freqs_cis_q = to_real(m.freqs_cis_q)
            m.freqs_cis_k = to_real(m.freqs_cis_k)
            m.forward = rope_attention_v2_forward.__get__(m)


# ---------------------------------------------------------------------------
# 変換用ラッパー (OpenVINO にはテンソルのタプルを入出力させる)
# ---------------------------------------------------------------------------
class ImageEncoder(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        out = self.p.forward_image(x)
        return tuple(out["backbone_fpn"]) + tuple(out["vision_pos_enc"])


class MemoryEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pix_feat, masks):
        out = self.m(pix_feat, masks, skip_mask_sigmoid=True)
        return out["vision_features"], out["vision_pos_enc"][0]


class MaskDecoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image_embeddings, image_pe, sparse, dense, feat_s0, feat_s1):
        return self.m.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=[feat_s0, feat_s1],
        )


class MemoryAttention(torch.nn.Module):
    def __init__(self, m, num_obj_ptr_tokens, num_spatial_mem):
        super().__init__()
        self.m = m
        self.num_obj_ptr_tokens = num_obj_ptr_tokens
        self.num_spatial_mem = num_spatial_mem

    def forward(self, curr, memory, curr_pos, memory_pos):
        return self.m(
            curr=[curr],
            memory=memory,
            curr_pos=[curr_pos],
            memory_pos=memory_pos,
            num_obj_ptr_tokens=self.num_obj_ptr_tokens,
            num_spatial_mem=self.num_spatial_mem,
        )


def memory_attention_config(p, k):
    """過去 k フレーム分のメモリがあるときの入力形状。"""
    num_spatial_mem = min(k, p.num_maskmem)
    # 1 フレームの obj_ptr (hidden_dim) は mem_dim ごとに分割されてトークンになる
    num_obj_ptr_tokens = k * (p.hidden_dim // p.mem_dim)
    tokens_per_mem = p.spatial_perceiver.num_latents + p.spatial_perceiver.num_latents_2d
    mem_len = num_spatial_mem * tokens_per_mem + num_obj_ptr_tokens
    return num_spatial_mem, num_obj_ptr_tokens, mem_len


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    p = build_sam2_video_predictor(
        "configs/edgetam.yaml", "./checkpoints/edgetam.pt", device="cpu"
    ).eval()
    patch_rope_for_export(p)

    with torch.no_grad():
        print("image_encoder ...")
        x = torch.randn(1, 3, p.image_size, p.image_size)
        om = ov.convert_model(ImageEncoder(p), example_input=x, input=[x.shape])
        ov.save_model(om, f"{OUT_DIR}/image_encoder.xml", compress_to_fp16=False)

        print("mask_decoder ...")
        feat = p.image_size // 16
        d = p.hidden_dim
        # 追跡中のフレームはプロンプトなし (パディング点のみ) で sparse は 2 トークン
        ex = (
            torch.randn(1, d, feat, feat),
            torch.randn(1, d, feat, feat),
            torch.randn(1, 2, d),
            torch.randn(1, d, feat, feat),
            torch.randn(1, d // 8, feat * 4, feat * 4),
            torch.randn(1, d // 4, feat * 2, feat * 2),
        )
        om = ov.convert_model(
            MaskDecoder(p.sam_mask_decoder), example_input=ex, input=[e.shape for e in ex]
        )
        ov.save_model(om, f"{OUT_DIR}/mask_decoder.xml", compress_to_fp16=False)

        print("memory_encoder ...")
        ex = (
            torch.randn(1, p.hidden_dim, feat, feat),
            torch.randn(1, 1, p.image_size, p.image_size),
        )
        om = ov.convert_model(
            MemoryEncoder(p.memory_encoder), example_input=ex, input=[e.shape for e in ex]
        )
        ov.save_model(om, f"{OUT_DIR}/memory_encoder.xml", compress_to_fp16=False)

        n = feat * feat
        for k in range(1, p.max_obj_ptrs_in_encoder + 1):
            nsp, nptr, mem_len = memory_attention_config(p, k)
            print(f"memory_attention k={k} (mem_len={mem_len}) ...")
            ex = (
                torch.randn(n, 1, p.hidden_dim),
                torch.randn(mem_len, 1, p.mem_dim),
                torch.randn(n, 1, p.hidden_dim),
                torch.randn(mem_len, 1, p.mem_dim),
            )
            # k[:, :, :n] = ... の in-place 代入で trace の自己チェックが誤検知するので
            # チェックなしで trace する (正しさは verify_openvino.py で数値比較する)
            traced = torch.jit.trace(
                MemoryAttention(p.memory_attention, nptr, nsp), ex, check_trace=False
            )
            om = ov.convert_model(traced, example_input=ex, input=[e.shape for e in ex])
            ov.save_model(om, f"{OUT_DIR}/memory_attention_k{k}.xml", compress_to_fp16=False)
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
