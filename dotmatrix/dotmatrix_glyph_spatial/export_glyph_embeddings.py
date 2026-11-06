# -*- coding: utf-8 -*-
"""
export_glyph_embeddings.py
把已训练好的三个字形编码器(S结构化 / R纯像素 / F融合头)对全部字库跑一遍,
导出每个字的 64 维字形嵌入, 供"字形判别小应用"离线使用。
输出: glyph_embeddings.json  (char -> {S:[64], R:[64], F:[64]})
"""
import os, json, base64, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRUCT_JSON = "D:/desktop/学习结果小文件/neuroglyph34_project/struct32_aligned.json"
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
PIXEL_CACHE = os.path.join(BASE_DIR, "PIXEL_cache.npy")
DEVICE = torch.device("cpu")


class PixelEncoder(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


class StructEncoder(nn.Module):
    def __init__(self, in_dim=26, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


class FusionEncoder(nn.Module):
    def __init__(self, p_in=1024, s_in=26, pH=256, sH=64, out_dim=64):
        super().__init__()
        self.pixel = nn.Sequential(nn.Linear(p_in, pH), nn.ReLU())
        self.struct = nn.Sequential(nn.Linear(s_in, sH), nn.ReLU())
        self.head = nn.Linear(pH + sH, out_dim)
    def forward(self, p, s):
        return F.normalize(self.head(torch.cat([self.pixel(p), self.struct(s)], -1)), dim=-1)


def main():
    c2i = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    i2c = {v: k for k, v in c2i.items()}
    V = len(c2i)
    print(f"[vocab] {V} 字")

    PIXEL = np.load(PIXEL_CACHE).astype(np.float32)
    assert PIXEL.shape[0] == V, f"PIXEL {PIXEL.shape} vs vocab {V}"

    sj = json.load(open(STRUCT_JSON, encoding="utf-8"))
    nopy26 = sj["nopy26"]
    STRUCT = np.zeros((V, 26), np.float32)
    for idx in range(V):
        ch = i2c.get(idx, "")
        if ch in nopy26:
            arr = np.asarray(nopy26[ch], np.float32)
            if arr.shape[0] >= 26:
                STRUCT[idx] = arr[:26]
    print(f"[struct] 有结构码 {int((STRUCT.sum(1) != 0).sum())} 字")

    PIXEL_t = torch.from_numpy(PIXEL).to(DEVICE)
    STRUCT_t = torch.from_numpy(STRUCT).to(DEVICE)

    encS = StructEncoder().to(DEVICE); encS.load_state_dict(torch.load(os.path.join(BASE_DIR, "enc_S_struct.pt"), map_location=DEVICE))
    encR = PixelEncoder().to(DEVICE); encR.load_state_dict(torch.load(os.path.join(BASE_DIR, "enc_R_pixel.pt"), map_location=DEVICE))
    encF = FusionEncoder().to(DEVICE); encF.load_state_dict(torch.load(os.path.join(BASE_DIR, "enc_F_fusion.pt"), map_location=DEVICE))
    encS.eval(); encR.eval(); encF.eval()
    print("[load] 三个编码器就绪")

    with torch.no_grad():
        E_S = encS(STRUCT_t).numpy().astype(np.float32)
        E_R = encR(PIXEL_t).numpy().astype(np.float32)
        E_F = encF(PIXEL_t, STRUCT_t).numpy().astype(np.float32)
    print(f"[embed] S{E_S.shape} R{E_R.shape} F{E_F.shape}")

    # 组装: 只导出真正有输入数据的字(避免零向量喂出伪嵌入)
    px_norm = np.linalg.norm(PIXEL, axis=1)
    st_norm = np.linalg.norm(STRUCT, axis=1)
    cov = {
        "S": st_norm > 0.5,
        "R": px_norm > 0.5,
        "F": (st_norm > 0.5) & (px_norm > 0.5),
    }
    out = {}
    for mode, E in [("S", E_S), ("R", E_R), ("F", E_F)]:
        mask = cov[mode]
        d = {}
        for idx in np.where(mask)[0]:
            ch = i2c[idx]
            d[ch] = [round(float(x), 4) for x in E[idx]]
        out[mode] = d
    print(f"[covered] S={cov['S'].sum()} R={cov['R'].sum()} F={cov['F'].sum()}")

    meta = {
        "note": "字形判别嵌入: S=预先结构化(26维结构码) / R=纯随机像素(1024) / F=融合头",
        "dim": 64, "vocab": V,
        "covered": {k: int(v.sum()) for k, v in cov.items()},
        "version": "hardneg-443/111, EPOCHS=40",
    }
    json.dump({"meta": meta, "embed": out},
              open(os.path.join(BASE_DIR, "glyph_embeddings.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    sz = os.path.getsize(os.path.join(BASE_DIR, "glyph_embeddings.json")) / 1e6
    print(f"[save] glyph_embeddings.json ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
