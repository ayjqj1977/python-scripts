import json, os, math

BASE = os.path.dirname(os.path.abspath(__file__))
GRAPH = "D:/desktop/mmh_graphics.txt"
VOCAB = os.path.join(BASE, "char_to_idx_full.json")

# 1) 解析 graphics.txt
data = {}
with open(GRAPH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ch = o.get("character")
        if ch:
            data[ch] = o
print(f"[data] graphics.txt 共 {len(data)} 字")

# 2) 覆盖率
vocab = json.load(open(VOCAB, encoding="utf-8"))
have = [c for c in vocab if c in data]
print(f"[coverage] 词表 {len(vocab)} 字, 有矢量笔画数据 {len(have)} 字 ({len(have)/len(vocab)*100:.1f}%)")
miss = [c for c in vocab if c not in data][:20]
print(f"[coverage] 缺失样本: {miss}")

# 3) 笔画特征提取
def stroke_features(o):
    feats = []
    for med in o["medians"]:
        pts = [(x, y) for x, y in med]
        if len(pts) < 2:
            continue
        # 长度
        L = sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1))
        # 方向(首->尾)
        dx = pts[-1][0]-pts[0][0]
        dy = pts[-1][1]-pts[0][1]
        ang = math.atan2(dy, dx)
        # 中心(归一化到0..1)
        cx = sum(p[0] for p in pts)/len(pts)/1024.0
        cy = sum(p[1] for p in pts)/len(pts)/1024.0
        feats.append({"L": round(L/1024.0, 4), "ang": round(ang, 3),
                      "cx": round(cx, 3), "cy": round(cy, 3)})
    return feats

for ch in ["末", "未", "王", "玉", "田", "由", "风", "冈", "日", "曰", "土", "士"]:
    if ch in data:
        fs = stroke_features(data[ch])
        print(f"\n[{ch}] 笔画数={len(fs)}")
        for i, f in enumerate(fs):
            print(f"  笔{i}: L={f['L']:.3f} ang={f['ang']:.2f} 中心=({f['cx']:.2f},{f['cy']:.2f})")
    else:
        print(f"\n[{ch}] 无数据")
