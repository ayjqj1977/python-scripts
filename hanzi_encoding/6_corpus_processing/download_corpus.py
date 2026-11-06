# -*- coding: utf-8 -*-
import requests
import urllib.parse
import os
import json

TARGET_DIR = r'D:\desktop\学习结果小文件\corpus'
os.makedirs(TARGET_DIR, exist_ok=True)

raw_base_url = 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/'

def download_file(url, filepath):
    """下载文件"""
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            content = r.text
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            size = len(content)
            print(f"  ✓ {os.path.basename(filepath)} ({size:,} bytes)")
            return True
        else:
            print(f"  ✗ [{r.status_code}] {os.path.basename(filepath)}")
            return False
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False

# 从API获取完整文件列表
print("=== 获取完整文件列表 ===")
folders = {
    '全唐诗': [],
    '宋词': [],
    '诗经': [],
    '元曲': [],
    '论语': []
}

for folder in folders.keys():
    url = f'https://api.github.com/repos/chinese-poetry/chinese-poetry/contents/{urllib.parse.quote(folder)}'
    r = requests.get(url)
    if r.status_code == 200:
        files = r.json()
        folders[folder] = [f['name'] for f in files if f['type'] == 'file' and f['name'].endswith('.json')]
        print(f"{folder}: {len(folders[folder])} 个JSON文件")

# 下载所有文件
print("\n=== 开始下载 ===")

# 1. 诗经
print("\n[诗经]")
for fname in folders.get('诗经', []):
    url = f"{raw_base_url}{urllib.parse.quote('诗经')}/{urllib.parse.quote(fname)}"
    download_file(url, os.path.join(TARGET_DIR, f"shijing_{fname}"))

# 2. 元曲
print("\n[元曲]")
for fname in folders.get('元曲', []):
    url = f"{raw_base_url}{urllib.parse.quote('元曲')}/{urllib.parse.quote(fname)}"
    download_file(url, os.path.join(TARGET_DIR, f"yuanqu_{fname}"))

# 3. 论语
print("\n[论语]")
for fname in folders.get('论语', []):
    url = f"{raw_base_url}{urllib.parse.quote('论语')}/{urllib.parse.quote(fname)}"
    download_file(url, os.path.join(TARGET_DIR, f"lunyu_{fname}"))

# 4. 宋词 (下载前几个)
print("\n[宋词]")
ci_files = [f for f in folders.get('宋词', []) if f.endswith('.json')][:5]
for fname in ci_files:
    url = f"{raw_base_url}{urllib.parse.quote('宋词')}/{urllib.parse.quote(fname)}"
    download_file(url, os.path.join(TARGET_DIR, f"songci_{fname}"))

# 5. 全唐诗 (下载前几个)
print("\n[全唐诗]")
tang_files = [f for f in folders.get('全唐诗', []) if f.endswith('.json') and 'poet.song' in f][:3]
for fname in tang_files:
    url = f"{raw_base_url}{urllib.parse.quote('全唐诗')}/{urllib.parse.quote(fname)}"
    download_file(url, os.path.join(TARGET_DIR, f"quantangshi_{fname}"))

print("\n=== 下载完成 ===")

# 统计
total_size = 0
for f in os.listdir(TARGET_DIR):
    fpath = os.path.join(TARGET_DIR, f)
    total_size += os.path.getsize(fpath)
print(f"总计: {len(os.listdir(TARGET_DIR))} 个文件, {total_size:,} bytes")
