#!/usr/bin/env python3
"""快速下载中文语料 — 直链+HF streaming"""
import os, sys, time, subprocess

OUTDIR = r"D:\Desktop\neuroglyph34_project\corpus_downloaded"
os.makedirs(OUTDIR, exist_ok=True)

# 安装依赖
print("安装依赖...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "requests", "tqdm", "-q", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from tqdm import tqdm

MAX_MB_PER_SOURCE = 200  # 每个源最多200MB

def stream_save(dataset_path, name, config=None, text_field="text", max_lines=None):
    """流式加载 HuggingFace 数据集并保存为 TXT"""
    output_path = os.path.join(OUTDIR, f"{name}.txt")
    print(f"\n[{name}] 加载 {dataset_path}...")
    try:
        kwargs = {"split": "train", "streaming": True, "trust_remote_code": True}
        if config:
            kwargs["name"] = config

        ds = load_dataset(dataset_path, **kwargs)
        count = 0
        limit = max_lines or 500000
        with open(output_path, "w", encoding="utf-8") as f:
            for item in ds:
                # 提取文本
                if isinstance(item, str):
                    text = item
                elif text_field in item:
                    text = str(item[text_field])
                elif "content" in item:
                    text = str(item["content"])
                elif "text" in item:
                    text = str(item["text"])
                else:
                    # 取最长的字符串字段
                    strs = {k: v for k, v in item.items() if isinstance(v, str)}
                    if strs:
                        text = str(max(strs.values(), key=len))
                    else:
                        continue

                text = text.strip()
                if len(text) > 40:
                    f.write(text + "\n\n")
                    count += 1

                if count >= limit:
                    break
                if count % 5000 == 0:
                    size_mb = os.path.getsize(output_path) / 1024 / 1024
                    print(f"  {count} 条 | {size_mb:.1f} MB", end="\r")
                    if size_mb >= MAX_MB_PER_SOURCE:
                        break

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n  [{name}] ✅ {count} 条, {size_mb:.1f} MB")
        return output_path
    except Exception as e:
        print(f"  [{name}] ❌ {e}")
        return None


# ====== 逐个下载 ======
sources = [
    # Wikipedia 中文 (高质量，~1.3GB 全部)
    ("pleisto/wikipedia-cn-20230720-filtered", "wiki_zh", None, "text"),

    # 新闻/百科通用中文预训练语料
    ("TigerResearch/pretrain_zh", "tiger_zh", None, "text"),

    # 百科问答（有答案的长文本）
    ("brightmart/nlp_chinese_corpus", "baike2018qa", "baike2018qa", "answer"),

    # Firefly 预训练（22GB, 只取部分）
    ("YeungNLP/firefly-pretrain-dataset", "firefly_zh", None, "text"),
]

results = {}
for path, name, cfg, field in sources:
    t0 = time.time()
    result = stream_save(path, name, cfg, field)
    results[name] = result
    elapsed = time.time() - t0
    print(f"  耗时 {elapsed:.0f}s")

# 汇总
print("\n" + "=" * 50)
print("下载汇总:")
total = 0
for name, path in results.items():
    if path and os.path.exists(path):
        mb = os.path.getsize(path) / 1024 / 1024
        total += mb
        print(f"  {name}.txt — {mb:.1f} MB")
    else:
        print(f"  {name}.txt — 未获取")
print(f"\n总计: {total:.1f} MB → {OUTDIR}")
print("完成!")