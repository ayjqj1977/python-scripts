# -*- coding: utf-8 -*-
"""
eval_glyph_focus.py — 统一形近字指标评估（旧/新模型，任意 checkpoint）
用法:
  python eval_glyph_focus.py --model old --ckpt <path>
  python eval_glyph_focus.py --model spatial --ckpt <path>
环境: D:/MICODA/envs/py312_torch/python.exe
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

ORIG_DIR = "D:/desktop/学习结果小文件/dotmatrix_glyph"
SPATIAL_DIR = "D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"

FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("未", "末"),
    ("干", "千"), ("干", "于"), ("千", "于"),
    ("大", "太"),
    ("王", "玉"),
    ("日", "曰"),
]


def cosine_sim(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["old", "spatial"], required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    if args.model == "old":
        sys.path.insert(0, ORIG_DIR)
        from model_pointnet import PointMatrixModel
        base = ORIG_DIR
        ModelCls = PointMatrixModel
    else:
        sys.path.insert(0, SPATIAL_DIR)
        from model_spatial import PointMatrixSpatialModel
        base = SPATIAL_DIR
        ModelCls = PointMatrixSpatialModel

    char_to_idx_path = os.path.join(base, "char_to_idx_full.json")
    hzk_path = os.path.join(base, "HZK32")

    with open(char_to_idx_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    sys.path.insert(0, base)
    from hzk32_reader import HZK32Reader
    reader = HZK32Reader(hzk_path)

    model = ModelCls(vocab_size=vocab_size)
    glyph_matrix = np.zeros((vocab_size, 1024), dtype=np.float32)
    loaded = 0
    for idx in range(vocab_size):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>":
            try:
                glyph_matrix[idx] = reader.get_glyph_vector(ch).astype(np.float32)
                loaded += 1
            except (ValueError, IOError):
                pass
    model.load_glyph_matrix(glyph_matrix)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    @torch.no_grad()
    def get_trained(ch):
        idx = char_to_idx.get(ch, 0)
        raw = model.glyph_matrix[idx:idx + 1]
        return model.point_encoder(raw).squeeze(0).numpy()

    def get_raw(ch):
        try:
            return reader.get_glyph_vector(ch).astype(np.float64)
        except (ValueError, IOError):
            return np.zeros(1024, np.float64)

    trained, raw = [], []
    for a, b in FOCUS_PAIRS:
        st = cosine_sim(get_trained(a), get_trained(b))
        sr = cosine_sim(get_raw(a), get_raw(b))
        trained.append(st)
        raw.append(sr)
    avg_t = float(np.mean(trained))
    avg_r = float(np.mean(raw))
    step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    print(f"[RESULT] model={args.model} step={step} "
          f"focus_avg_trained={avg_t:.6f} focus_avg_raw={avg_r:.6f} "
          f"separation={avg_r - avg_t:.6f}")
    # 同时逐对打印
    for (a, b), st, sr in zip(FOCUS_PAIRS, trained, raw):
        print(f"  {a}/{b}: trained={st:.4f} raw={sr:.4f}")


if __name__ == "__main__":
    main()
