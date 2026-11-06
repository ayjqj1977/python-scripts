# -*- coding: utf-8 -*-
"""
尝试从多个GitHub仓库下载更多中文语料
"""
import requests
import os
import json

TARGET_DIR = r'D:\desktop\学习结果小文件\corpus'
os.makedirs(TARGET_DIR, exist_ok=True)

# 尝试不同的仓库
repos = [
    # 中国历史仓库
    ("HarborLibrary", "Chinese-History"),
    # 其他中文古籍
    ("kanchoku", "kw"),
]

raw_base = "https://raw.githubusercontent.com/{owner}/{repo}/master/"

def try_download(owner, repo, filename):
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{filename}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.text, len(r.text)
    except: pass
    return None, 0

# 尝试的文件
test_files = [
    # Chinese-History仓库的文件
    ("%E5%8F%B2%E9%A9%AC%E5%85%89%EF%BC%9A%E8%B5%84%E6%B2%BB%E9%80%9A%E9%89%B4%20(%E6%9F%8F%E6%9D%A8%E7%99%BD%E8%AF%9D%E7%89%88).epub", "资治通鉴.epub"),
]

print("=== 尝试从更多仓库获取 ===\n")

# 尝试一些已知的大型中文语料
urls_to_try = [
    # 中文维基百科 (如果能访问)
    ("https://raw.githubusercontent.com/ArticleFragment/ChineseWikipediaExtractor/master/README.md", "wiki_readme"),
]

# 检查能获取什么
print("检查可用资源...")

# 尝试论语全本
url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E8%AE%BA%E8%AF%9D/lunyu.json"
r = requests.get(url)
print(f"论语: {r.status_code}")

# 尝试孟子
url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%9B%9B%E5%AD%90%E7%8E%8B/mengzi.json"
r = requests.get(url)
print(f"孟子: {r.status_code}")

# 尝试中庸
url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E4%B8%AD%E5%BA%B5/zhongyong.json"
r = requests.get(url)
print(f"中庸: {r.status_code}")

# 尝试大学
url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%A4%A7%E5%AD%A6/daxue.json"
r = requests.get(url)
print(f"大学: {r.status_code}")

print("\n=== 下载已有的完整文件 ===\n")

# 下载已有的
files_to_download = [
    ("四书五经", "mengzi", "mengzi.json"),
    ("四书五经", "lunyu", "lunyu.json"),
]

# 尝试获取更多五代诗词
url = "https://api.github.com/repos/chinese-poetry/chinese-poetry/contents/%E4%BA%94%E4%BB%A3%E8%AF%8D%E8%AF%8D"
r = requests.get(url)
if r.status_code == 200:
    data = r.json()
    print("五代诗词:")
    for item in data[:5]:
        print(f"  {item.get('name','')}")
