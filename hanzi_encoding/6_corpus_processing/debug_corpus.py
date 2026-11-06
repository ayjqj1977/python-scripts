# -*- coding: utf-8 -*-
import json, glob, os, collections

corpus_dir = r'D:\desktop\学习结果小文件\corpus'

print("检查corpus目录...")
files = glob.glob(os.path.join(corpus_dir, '*.json'))
print(f"找到 {len(files)} 个JSON文件")

# 尝试读取一个文件
for f in files[:2]:
    print(f"\n尝试: {os.path.basename(f)}")
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        print(f"  类型: {type(data)}")
        if isinstance(data, list):
            print(f"  长度: {len(data)}")
            if len(data) > 0:
                print(f"  示例: {str(data[0])[:100]}")
                # 看看有什么字段
                if isinstance(data[0], dict):
                    print(f"  字段: {list(data[0].keys())}")
        elif isinstance(data, dict):
            print(f"  键: {list(data.keys())[:5]}")
    except Exception as e:
        print(f"  错误: {e}")
