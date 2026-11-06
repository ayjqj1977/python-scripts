# -*- coding: utf-8 -*-
"""
从多个GitHub仓库下载更多中文语料
"""
import requests
import os
import json
import time

WORK_DIR = r'D:\desktop\学习结果小文件\corpus'
os.makedirs(WORK_DIR, exist_ok=True)

print("=== 尝试下载更多语料 ===\n")

def download_file(url, filepath, max_size=5*1024*1024):
    """下载文件"""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            if len(r.content) > max_size:
                print(f"  跳过(太大): {os.path.basename(filepath)}")
                return False
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(r.text)
            print(f"  ✓ {os.path.basename(filepath)} ({len(r.text)/1024:.1f}KB)")
            return True
    except Exception as e:
        pass
    return False

# 尝试下载的URL列表
download_urls = [
    # 尝试更多全唐诗
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.song.2000.json', 'quantangshi_2000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.song.3000.json', 'quantangshi_3000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.song.4000.json', 'quantangshi_4000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.song.5000.json', 'quantangshi_5000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.song.6000.json', 'quantangshi_6000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.song.7000.json', 'quantangshi_7000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.song.8000.json', 'quantangshi_8000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.song.9000.json', 'quantangshi_9000.json'),
    
    # 更多宋词
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.2000.json', 'songci_2000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.3000.json', 'songci_3000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.4000.json', 'songci_4000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.5000.json', 'songci_5000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.6000.json', 'songci_6000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.7000.json', 'songci_7000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.8000.json', 'songci_8000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.9000.json', 'songci_9000.json'),
    ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/宋词/ci.song.10000.json', 'songci_10000.json'),
]

success = 0
failed = 0

for url, filename in download_urls:
    filepath = os.path.join(WORK_DIR, filename)
    if os.path.exists(filepath):
        print(f"  已存在: {filename}")
        success += 1
        continue
    
    print(f"尝试: {filename}...")
    if download_file(url, filepath):
        success += 1
    else:
        failed += 1
        print(f"  ✗ 失败")
    time.sleep(0.2)  # 避免请求过快

print(f"\n=== 结果 ===")
print(f"成功: {success}")
print(f"失败: {failed}")

# 统计
total = len([f for f in os.listdir(WORK_DIR) if f.endswith('.json')])
print(f"\n总JSON文件: {total}")

# 统计大小
total_size = sum(os.path.getsize(os.path.join(WORK_DIR, f)) 
                for f in os.listdir(WORK_DIR) if f.endswith('.json'))
print(f"总大小: {total_size/1024/1024:.2f}MB")
