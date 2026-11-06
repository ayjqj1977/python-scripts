# -*- coding: utf-8 -*-
"""
下载更多中文古籍语料
四书五经 + 其他更多资源
"""
import requests
import urllib.parse
import os
import json

TARGET_DIR = r'D:\desktop\学习结果小文件\corpus'
os.makedirs(TARGET_DIR, exist_ok=True)

raw_base = 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/'

# 要下载的文件夹列表
folders = ['四书五经', '楚辞', '五代诗词', '幽梦影', '蒙学']

def download_file(url, filepath):
    """下载文件"""
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(r.text)
            return len(r.text)
        return 0
    except Exception as e:
        print(f"  错误: {e}")
        return 0

print("=== 下载更多中文古籍 ===\n")

total_size = 0

for folder in folders:
    print(f"\n📂 {folder}")
    url = f'https://api.github.com/repos/chinese-poetry/chinese-poetry/contents/{urllib.parse.quote(folder)}'
    r = requests.get(url)
    
    if r.status_code == 200:
        files = r.json()
        for f in files:
            if f.get('type') == 'file' and f['name'].endswith('.json'):
                raw_url = f"{raw_base}{urllib.parse.quote(folder)}/{urllib.parse.quote(f['name'])}"
                filepath = os.path.join(TARGET_DIR, f"{folder}_{f['name']}")
                size = download_file(raw_url, filepath)
                if size > 0:
                    print(f"  ✓ {f['name']} ({size:,} bytes)")
                    total_size += size
    else:
        print(f"  ✗ 无法访问: {r.status_code}")

print(f"\n=== 总计: {total_size/1024/1024:.2f} MB ===")

# 统计
print("\n=== 已下载文件 ===")
for f in sorted(os.listdir(TARGET_DIR)):
    fpath = os.path.join(TARGET_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  {f} ({size/1024:.1f}KB)")
