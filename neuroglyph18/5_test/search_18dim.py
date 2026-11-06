#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18维向量检索系统 - 基于规则的精确检索
不需要神经网络，直接用编码规则
"""

import json
import os

# 设置路径
WORK_DIR = r"D:\desktop\学习结果小文件"
VECTOR_FILE = os.path.join(WORK_DIR, "1_18维向量表", "汉字向量字库_扩展.json")

print("📚 加载18维向量数据...")
with open(VECTOR_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

characters = data['characters']
print(f"   字符数量: {len(characters)}")

# 构建索引
char_data = {}
for char_info in characters:
    char = char_info['char']
    vector = char_info['vector']
    
    # 提取各维度信息
    sijiao = ''.join(map(str, vector[0:5]))  # 1-5位
    wubi = ''.join(map(str, vector[5:9]))    # 6-9位
    pinyin_raw = vector[9:15]                  # 10-15位
    pinyin = ''.join([chr(c + 96) if c > 0 else '0' for c in pinyin_raw])
    tone = vector[15]                          # 16位
    char_type = vector[16]                     # 17位
    
    char_data[char] = {
        'vector': vector,
        'sijiao': sijiao,
        'wubi': wubi,
        'pinyin': pinyin,
        'pinyin_raw': pinyin_raw,
        'tone': tone,
        'type': char_type
    }

print("✅ 索引构建完成!\n")

# ====== 检索功能 ======

def search_by_pinyin(pinyin):
    """按拼音精确/模糊检索"""
    pinyin = pinyin.lower().strip()
    results = []
    
    for char, info in char_data.items():
        py = info['pinyin']
        
        # 精确匹配
        if py == pinyin:
            results.append((char, info, 100))
        # 前缀匹配
        elif py.startswith(pinyin):
            results.append((char, info, 80))
        # 包含匹配
        elif pinyin in py:
            results.append((char, info, 60))
    
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:20]

def search_by_sijiao(sijiao):
    """按四角号码精确检索"""
    results = []
    for char, info in char_data.items():
        sj = info['sijiao']
        if sj == sijiao:
            results.append(char)
    return results

def search_by_wubi(wubi):
    """按五笔码精确检索"""
    results = []
    for char, info in char_data.items():
        wb = info['wubi']
        if wb == wubi:
            results.append(char)
    return results

def search_by_tone(tone):
    """按声调检索"""
    tone_names = {1: '平声', 2: '上声', 3: '去声', 4: '入声'}
    results = []
    for char, info in char_data.items():
        if info['tone'] == tone:
            results.append(char)
    return results, tone_names.get(tone, '未知')

def find_similar_structure(target_char):
    """找结构相似字（四角号码相近）"""
    if target_char not in char_data:
        return []
    
    target_sj = char_data[target_char]['sijiao']
    results = []
    
    for char, info in char_data.items():
        if char == target_char:
            continue
        sj = info['sijiao']
        # 前3位相同
        if len(sj) >= 3 and len(target_sj) >= 3 and sj[:3] == target_sj[:3]:
            results.append((char, sj, 90))
        # 前2位相同
        elif len(sj) >= 2 and len(target_sj) >= 2 and sj[:2] == target_sj[:2]:
            results.append((char, sj, 70))
        # 前1位相同
        elif sj and target_sj and sj[0] == target_sj[0]:
            results.append((char, sj, 50))
    
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:15]

def analyze_char(char):
    """分析单个汉字的18维编码"""
    if char not in char_data:
        print(f"   找不到: {char}")
        return
    
    info = char_data[char]
    vector = info['vector']
    
    print(f"\n   汉字: {char}")
    print(f"   向量维度: 18")
    print(f"   完整向量: {vector}")
    print(f"   " + "-"*40)
    print(f"   位置 1-5 (四角号码): {info['sijiao']}")
    print(f"   位置 6-9 (五笔字型): {info['wubi']}")
    print(f"   位置 10-15 (拼音编码): {info['pinyin']}")
    print(f"   位置 16 (声调): {info['tone']} ({'平声上声去声入声'[info['tone']-1] if 1<=info['tone']<=4 else '未知'})")
    print(f"   位置 17 (字符类型): {info['type']} ({'汉字数字字母符号'[info['type']] if info['type']<4 else '未知'})")

# ====== 主界面 ======
def main():
    print("\n" + "="*50)
    print("🔍 18维向量精确检索系统")
    print("="*50)
    print("\n基于规则的精确检索，无需训练！")
    
    while True:
        print("\n" + "-"*50)
        print("可选功能:")
        print("  1. 拼音查字 (如: yi, shi, li)")
        print("  2. 四角号码查字 (如: 3710, 4410)")
        print("  3. 五笔查字 (如: 4514, 4491)")
        print("  4. 声调分类 (1=平, 2=上, 3=去, 4=入)")
        print("  5. 结构相似 (输入汉字找相似)")
        print("  6. 分析汉字 (查看18维编码详情)")
        print("  7. 退出")
        
        choice = input("\n请选择 (1-7): ").strip()
        
        if choice == '1':
            py = input("输入拼音: ").strip()
            if py:
                results = search_by_pinyin(py)
                print(f"\n   找到 {len(results)} 个结果:")
                for char, info, score in results:
                    print(f"   {char} | 拼音:{info['pinyin']} | 四角:{info['sijiao']} | 声调:{info['tone']} | 匹配度:{score}%")
        
        elif choice == '2':
            sj = input("输入四角号码(5位): ").strip()
            if sj:
                results = search_by_sijiao(sj)
                print(f"\n   四角号码 {sj} 对应: {results if results else '无'}")
        
        elif choice == '3':
            wb = input("输入五笔码(4位): ").strip()
            if wb:
                results = search_by_wubi(wb)
                print(f"\n   五笔码 {wb} 对应: {results if results else '无'}")
        
        elif choice == '4':
            t = input("输入声调(1-4): ").strip()
            if t and t.isdigit():
                results, name = search_by_tone(int(t))
                print(f"\n   {name} ({len(results)}字): {','.join(results[:30])}...")
        
        elif choice == '5':
            char = input("输入参考汉字: ").strip()
            if char:
                results = find_similar_structure(char)
                print(f"\n   与 '{char}' 结构相似的字:")
                for c, sj, score in results:
                    print(f"   {c} | 四角:{sj} | 相似度:{score}%")
        
        elif choice == '6':
            char = input("输入要分析的汉字: ").strip()
            if char:
                analyze_char(char)
        
        elif choice == '7':
            print("再见!")
            break
        else:
            print("无效选择")

if __name__ == "__main__":
    main()
