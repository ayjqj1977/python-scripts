#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语料下载脚本 — 自动拉取中文训练语料
=================================
目标: 下载 ~2-5GB 的纯文本中文语料，覆盖现代文章、百科、新闻、散文

数据源:
  1. Chinese Wikipedia (处理好的中文字条)
  2. CLUECorpusSmall (新闻+百科)
  3. Firefly 预训练语料 (22GB完整版 → 只取部分)
  
输出: corpus_downloaded/ 目录下多个 .txt 文件
"""
import os, sys, json, re, gzip, io, tarfile, zipfile
import urllib.request
import urllib.error
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# ========== 配置 ==========
OUTPUT_DIR = r"D:\Desktop\neuroglyph34_project\corpus_downloaded"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HuggingFace 镜像（国内加速）
HF_MIRROR = "https://hf-mirror.com"


# ====================================================================
# 方案1: Wikipedia 中文版 (处理后纯文本，~1.5GB)
# ====================================================================
def download_wiki():
    """下载处理好的中文维基百科文本"""
    print("\n[1/4] 中文维基百科...")

    # 使用处理好的 clean wiki text (来自 HuggingFace)
    try:
        from datasets import load_dataset
        print("  尝试通过 datasets 加载 wikipedia...")
        dataset = load_dataset(
            "wikipedia",
            "20220301.zh",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        output_path = os.path.join(OUTPUT_DIR, "wiki_zh.txt")
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                text = item.get("text", "")
                # 清理 wiki 标记
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\{\{[^}]+\}\}", "", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = text.strip()
                if len(text) > 40:
                    f.write(text + "\n\n")
                    count += 1
                    if count % 1000 == 0:
                        print(f"   已处理 {count} 条...", end="\r")
                    if count >= 500000:  # 限制50万条
                        break
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n  wiki_zh.txt: {count} 条, {size_mb:.1f} MB")
        return output_path
    except Exception as e:
        print(f"  datasets 加载失败: {e}")
        return _download_wiki_fallback()


def _download_wiki_fallback():
    """备选方案：从 GitHub release 下载预处理的wiki文本"""
    urls = [
        "https://github.com/nickzyl/cn-wiki-corpus/releases/download/v1.0/wiki_zh_2023_clean.txt.gz",
    ]
    output_path = os.path.join(OUTPUT_DIR, "wiki_zh.txt")
    for url in urls:
        try:
            print(f"  备选下载: {url}")
            gz_path = output_path + ".gz"
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rb") as gz, open(output_path, "w", encoding="utf-8") as out:
                # 边解压边写入
                for line in gz:
                    out.write(line.decode("utf-8", errors="replace"))
            os.remove(gz_path)
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"  wiki_zh.txt: {size_mb:.1f} MB")
            return output_path
        except Exception as e:
            print(f"  下载失败: {e}")
    return None


# ====================================================================
# 方案2: 新闻 + 百科语料 (CLUECorpusSmall / 直接下载)
# ====================================================================
def download_news_corpus():
    """下载新闻文本语料"""
    print("\n[2/4] 新闻语料...")
    output_path = os.path.join(OUTPUT_DIR, "news_zh.txt")

    try:
        # 尝试用 datasets 加载 news 类数据
        from datasets import load_dataset

        # CLUECorpusSmall
        try:
            print("  尝试 CLUECorpusSmall...")
            ds = load_dataset("clue", "cluecorpussmall", split="train", streaming=True)
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for item in ds:
                    text = item.get("content", "") or item.get("text", "")
                    text = str(text).strip()
                    if len(text) > 30:
                        f.write(text + "\n\n")
                        count += 1
                    if count >= 200000:
                        break
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"  news_zh.txt: {count} 条, {size_mb:.1f} MB")
            return output_path
        except:
            pass

        # 备选: THUCNews 或直接抓取
        print("  使用备选方案...")
        return _download_news_fallback()
    except Exception as e:
        print(f"  失败: {e}")
        return _download_news_fallback()


def _download_news_fallback():
    """用简单方式获取新闻文本"""
    output_path = os.path.join(OUTPUT_DIR, "news_zh.txt")

    # 从公开GitHub仓库下载预处理好的新闻
    urls = [
        # 尝试多个来源
        "https://raw.githubusercontent.com/CLUEbenchmark/CLUECorpus2020/master/examples/cluecorpussmall_sample.txt",
    ]

    for url in urls:
        try:
            print(f"  下载: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8", errors="replace")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(data)
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"  news_zh.txt: {size_mb:.1f} MB")
            return output_path
        except Exception as e:
            print(f"  失败: {e}")

    # 如果什么都下载不到，生成占位说明文件
    print("  所有新闻源不可用，跳过")
    return None


# ====================================================================
# 方案3: 百科/知识类语料 (BaiduBaike / HudongBaike)
# ====================================================================
def download_baike():
    """下载百科类知识文本"""
    print("\n[3/4] 百科知识...")
    output_path = os.path.join(OUTPUT_DIR, "baike_zh.txt")

    try:
        from datasets import load_dataset

        # baike2018qa
        try:
            print("  尝试 baike2018qa...")
            ds = load_dataset(
                "brightmart/nlp_chinese_corpus",
                "baike2018qa",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for item in ds:
                    q = item.get("question", "") or item.get("title", "")
                    a = item.get("answer", "") or item.get("text", "")
                    text = f"{q}\n{a}".strip()
                    if len(text) > 20:
                        f.write(text + "\n\n")
                        count += 1
                    if count >= 150000:
                        break
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"  baike_zh.txt: {count} 条, {size_mb:.1f} MB")
            return output_path
        except:
            pass

        # 通用中文语料
        try:
            print("  尝试 Chinese-pretraining-dataset...")
            ds = load_dataset(
                "Linly-AI/Chinese-pretraining-dataset",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for item in ds:
                    text = item.get("text", "") or str(item)
                    text = str(text).strip()
                    if len(text) > 30:
                        f.write(text + "\n\n")
                        count += 1
                    if count >= 200000:
                        break
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"  baike_zh.txt: {count} 条, {size_mb:.1f} MB")
            return output_path
        except:
            pass

        print("  没有可用的百科数据源")
        return None
    except Exception as e:
        print(f"  失败: {e}")
        return None


# ====================================================================
# 方案4: 现代散文/文章 (从 Firefly / COIG 提取)
# ====================================================================
def download_essays():
    """下载现代文章/散文"""
    print("\n[4/4] 现代文章/散文...")
    output_path = os.path.join(OUTPUT_DIR, "essays_zh.txt")

    try:
        from datasets import load_dataset

        # Firefly pretrain dataset
        try:
            print("  尝试 firefly-pretrain-dataset...")
            ds = load_dataset(
                "YeungNLP/firefly-pretrain-dataset",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for item in ds:
                    text = item.get("text", "") or str(item)
                    text = str(text).strip()
                    # 过滤太短的和纯代码
                    if len(text) > 50 and not text.startswith("{"):
                        f.write(text + "\n\n")
                        count += 1
                    if count % 1000 == 0:
                        print(f"   已处理 {count} 条...", end="\r")
                    if count >= 300000:
                        break
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"\n  essays_zh.txt: {count} 条, {size_mb:.1f} MB")
            return output_path
        except:
            pass

        # 备选: COIG 中文指令数据
        try:
            print("  尝试 COIG...")
            ds = load_dataset(
                "m-a-p/COIG",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for item in ds:
                    text = item.get("instruction", "") + "\n" + item.get("output", "")
                    text = str(text).strip()
                    if len(text) > 30:
                        f.write(text + "\n\n")
                        count += 1
                    if count >= 200000:
                        break
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"  essays_zh.txt: {count} 条, {size_mb:.1f} MB")
            return output_path
        except:
            pass

        print("  无可用数据源")
        return None
    except Exception as e:
        print(f"  失败: {e}")
        return None


# ====================================================================
# 主流程
# ====================================================================
def main():
    print("=" * 60)
    print("中文训练语料下载")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 先检查 datasets 库
    try:
        import datasets
        print(f"datasets 版本: {datasets.__version__}")
    except ImportError:
        print("\n安装 datasets...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        import datasets

    # 设置 HF 镜像
    os.environ["HF_ENDPOINT"] = HF_MIRROR

    # 并行下载
    results = {}

    # 逐个下载（streaming模式不需要并行）
    r1 = download_wiki()
    results["wiki"] = r1

    r2 = download_news_corpus()
    results["news"] = r2

    r3 = download_baike()
    results["baike"] = r3

    r4 = download_essays()
    results["essays"] = r4

    # 汇总
    print("\n" + "=" * 60)
    print("下载汇总:")
    total_size = 0
    for name, path in results.items():
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            total_size += size_mb
            print(f"  [{name}] {path} — {size_mb:.1f} MB")
        else:
            print(f"  [{name}] 未获取")
    print(f"\n总大小: {total_size:.1f} MB")

    # 生成合并文件
    if total_size > 0:
        combined_path = os.path.join(OUTPUT_DIR, "corpus_combined.txt")
        print(f"\n合并到 {combined_path}...")
        with open(combined_path, "w", encoding="utf-8") as out:
            for name, path in results.items():
                if path and os.path.exists(path) and name != "combined":
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        out.write(f.read())
        combined_size = os.path.getsize(combined_path) / 1024 / 1024
        print(f"合并完成: {combined_size:.1f} MB")

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()