#!/usr/bin/env python3
# 下载《同义词词林（扩展版）》(哈工大 IR-Lab) 原版数据
import urllib.request, zipfile, io, os, sys

OUT_DIR = r"D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
OUT_FILE = os.path.join(OUT_DIR, "cilin_extended.txt")

# 候选源：优先原版仓库 zip（含中文文件名），其次 ASCII 文件名镜像
URLS = [
    ("哈工大原版 zip", "https://github.com/One-sixth/HIT-IR-Lab-Tongyici-Cilin-Extended/archive/refs/heads/master.zip"),
    ("bububa/cilin raw", "https://raw.githubusercontent.com/bububa/cilin/master/cilin.txt"),
]

def try_zip(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    for name in z.namelist():
        if "词林" in name and name.endswith(".txt"):
            raw = z.read(name)
            # 原版要求 GB18030
            for enc in ("gb18030", "utf-8", "gbk"):
                try:
                    txt = raw.decode(enc)
                    return txt, enc
                except Exception:
                    continue
    return None, None

def try_raw(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    for enc in ("gb18030", "utf-8", "gbk"):
        try:
            return raw.decode(enc), enc
        except Exception:
            continue
    return None, None

ok = False
for label, url in URLS:
    try:
        print(f"[try] {label}: {url}")
        if url.endswith(".zip"):
            txt, enc = try_zip(url)
        else:
            txt, enc = try_raw(url)
        if txt:
            with open(OUT_FILE, "w", encoding="utf-8") as f:
                f.write(txt)
            nlines = txt.count("\n") + (0 if txt.endswith("\n") else 1)
            print(f"[OK] {label} 编码={enc} 行数={nlines} 字节={os.path.getsize(OUT_FILE)}")
            print(f"[saved] {OUT_FILE}")
            ok = True
            break
        else:
            print(f"[fail-decode] {label}")
    except Exception as e:
        print(f"[error] {label}: {e}")

if not ok:
    print("ALL_FAILED")
    sys.exit(1)
