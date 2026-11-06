# -*- coding: utf-8 -*-
"""
fuse_struct_pixel_v2.py — 用"32维向量_完整数字版.xlsx"的更丰富结构码重测融合
====================================================================================
与 v1 的区别：v1 用的结构码只有 四角+五笔+声调+类型(12维，无笔顺)，
本版用这张新表的结构码，含关键新增维度 笔顺(10)——它能区分 王/玉、风/凤。

对比表示：
  P    : 像素64维 (point_encoder 输出, 现状)
  S12  : 旧18维设计码去拼音(四角+五笔+声调+类型) —— v1 的结果, 作基线对照
  S26  : 新表 纯结构码(去拼音去符号) = 四角5+五笔4+总笔画1+部首索引3+部首外笔画1+同部同画序号1+结构1+笔顺10
  S32  : 新表 全部32维(含拼音)
  S28  : S26 + 符号2
  F26c : concat([P, S26]) -> l2norm
  F26a : l2norm(P + 0.5*orth(S26->64))
  (F32/F28 类似)

重点回答：带笔顺的更丰富结构码, 融合后能否改善(而非像 v1 那样恶化)字形分离度?
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_spatial import PointMatrixSpatialModel
from hzk32_reader import HZK32Reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints_spatial", "ckpt_step050000.pt")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx_full.json")
HZK32_PATH = os.path.join(BASE_DIR, "HZK32")
STRUCT_JSON = "D:/desktop/学习结果小文件/neuroglyph34_project/struct32_aligned.json"
OLD_TABLE = "D:/desktop/学习结果小文件/1_18维向量表/汉字向量字库_扩展.json"
DEVICE = torch.device("cpu")

# 旧18维设计码维度: 0-4四角/5-8五笔/9-14拼音/15声调/16-17类型; 去拼音保留0-8+15-17=12
KEEP_OLD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17]

FOCUS_PAIRS = [
    ("己", "已"), ("己", "巳"), ("已", "巳"),
    ("未", "末"), ("干", "千"), ("干", "于"), ("千", "于"),
    ("大", "太"), ("王", "玉"), ("日", "曰"),
    ("王", "主"), ("风", "凤"), ("人", "入"), ("土", "士"), ("天", "夫"),
]


def cosine_sim(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def l2norm(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1
    return mat / n


def load_all():
    with open(CHAR_TO_IDX_PATH, encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    V = len(char_to_idx)
    model = PointMatrixSpatialModel(vocab_size=V)
    reader = HZK32Reader(HZK32_PATH)
    gm = np.zeros((V, 1024), np.float32)
    loaded = 0
    for idx in range(V):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>":
            try:
                gm[idx] = reader.get_glyph_vector(ch).astype(np.float32)
                loaded += 1
            except (ValueError, IOError):
                pass
    model.load_glyph_matrix(gm)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    print(f"[Load] 点阵 {loaded}/{V}, ckpt {os.path.basename(CKPT_PATH)}")
    return model, char_to_idx, idx_to_char, gm, V


@torch.no_grad()
def extract_pixel64(model, gm):
    return model.point_encoder(model.glyph_matrix).numpy().astype(np.float64)


def build_struct(mat_or_dict, char_to_idx, idx_to_char, V, dim):
    """mat_or_dict: dict char->list, or None(旧表走 special)"""
    S = np.zeros((V, dim), np.float64)
    hit = 0
    for idx in range(V):
        ch = idx_to_char.get(idx, "")
        if ch in mat_or_dict:
            arr = np.asarray(mat_or_dict[ch], np.float64)
            if arr.shape[0] >= dim:
                S[idx] = arr[:dim]
                hit += 1
    return S, hit


def build_old12(char_to_idx, idx_to_char, V):
    d = json.load(open(OLD_TABLE, encoding="utf-8"))
    m = {}
    for it in d["characters"]:
        v = it.get("vector")
        if isinstance(v, list) and len(v) == 18:
            arr = np.asarray(v, np.float64)
            if not (np.all(arr == 0) or np.all(arr[5:9] == 25)):
                m[it["char"]] = arr
    S = np.zeros((V, 12), np.float64)
    hit = 0
    for idx in range(V):
        ch = idx_to_char.get(idx, "")
        if ch in m:
            S[idx] = m[ch][KEEP_OLD]
            hit += 1
    return S, hit


def pair_cos(mat, a, b, char_to_idx):
    return cosine_sim(mat[char_to_idx.get(a, 0)], mat[char_to_idx.get(b, 0)])


def global_baseline(mat, char_to_idx, idx_to_char, n=500):
    top = []
    for idx in range(1, mat.shape[0]):
        ch = idx_to_char.get(idx, "")
        if ch and ch != "<UNK>" and len(ch) == 1:
            try:
                gb = ch.encode("gb2312")
                if len(gb) == 2 and gb[0] >= 0xB0:
                    top.append(ch)
            except UnicodeEncodeError:
                continue
        if len(top) >= n:
            break
    vecs = l2norm(np.stack([mat[char_to_idx[c]] for c in top], 0))
    sm = vecs @ vecs.T
    iu = np.triu_indices(len(top), k=1)
    return float(sm[iu].mean())


def main():
    model, c2i, i2c, gm, V = load_all()
    P = l2norm(extract_pixel64(model, gm))

    sj = json.load(open(STRUCT_JSON, encoding="utf-8"))
    all32, nopy26, withsym28 = sj["all32"], sj["nopy26"], sj["withsym28"]

    S26, h26 = build_struct(nopy26, c2i, i2c, V, 26)
    S32, h32 = build_struct(all32, c2i, i2c, V, 32)
    S28, h28 = build_struct(withsym28, c2i, i2c, V, 28)
    S12, h12 = build_old12(c2i, i2c, V)
    print(f"[Align] S26={h26} S32={h32} S28={h28} S12(old)={h12} / {V}")

    Sn26, Sn32, Sn28, Sn12 = l2norm(S26), l2norm(S32), l2norm(S28), l2norm(S12)

    rng = np.random.default_rng(0)
    def add_fuse(Sn, dim):
        W, _ = np.linalg.qr(rng.standard_normal((64, dim)))
        return l2norm(P + 0.5 * (Sn @ W.T))
    F26a = add_fuse(Sn26, 26)
    F32a = add_fuse(Sn32, 32)
    F28a = add_fuse(Sn28, 28)
    F26c = l2norm(np.concatenate([P, Sn26], 1))
    F12a = add_fuse(Sn12, 12)

    reps = {
        "P_像素64": P,
        "S12_旧结构(无笔顺)": Sn12,
        "S26_新结构(去拼音)": Sn26,
        "S32_新全32(含拼音)": Sn32,
        "S28_新.+符号": Sn28,
        "F12a_旧加法": F12a,
        "F26a_新加法": F26a,
        "F26c_新concat": F26c,
        "F32a_新全32加法": F32a,
        "F28a_新+符号加法": F28a,
    }

    print("\n" + "=" * 78)
    print("融合对比 v2（带笔顺的更丰富结构码）：字形分离度")
    print("=" * 78)
    print(f"{'表示':<22}{'重点对均cos':>12}{'随机基线':>11}{'倍率':>8}")
    print("-" * 56)
    result = {}
    for name, mat in reps.items():
        fps = [pair_cos(mat, a, b, c2i) for a, b in FOCUS_PAIRS]
        avg = float(np.mean(fps))
        gb = global_baseline(mat, c2i, i2c)
        ratio = avg / (gb + 1e-8)
        print(f"{name:<22}{avg:>12.4f}{gb:>11.4f}{ratio:>7.2f}x")
        result[name] = {"focus_avg_cos": avg, "global_baseline": gb, "ratio": ratio,
                        "focus_pairs": {f"{a}/{b}": pair_cos(mat, a, b, c2i) for a, b in FOCUS_PAIRS}}

    # 硬骨头逐对：像素 vs 旧结构12 vs 新结构26 vs 加法融合26
    print("\n" + "=" * 78)
    print("硬骨头逐对 cosine：P像素 / S12旧 / S26新结构 / F26a加法融合")
    print("=" * 78)
    print(f"{'字对':<9}{'P像素64':>9}{'S12旧':>9}{'S26新':>9}{'F26a':>9}")
    print("-" * 47)
    for a, b in FOCUS_PAIRS:
        print(f"{a}/{b:<7}{pair_cos(P,a,b,c2i):>9.4f}{pair_cos(Sn12,a,b,c2i):>9.4f}"
              f"{pair_cos(Sn26,a,b,c2i):>9.4f}{pair_cos(F26a,a,b,c2i):>9.4f}")

    out = {"ckpt": "ckpt_step050000.pt",
           "note": "v2: 用32维表(含笔顺)重测结构码与像素64融合",
           "reps": result}
    op = os.path.join(BASE_DIR, "fuse_struct_v2_result.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[Save] {op}")


if __name__ == "__main__":
    main()
