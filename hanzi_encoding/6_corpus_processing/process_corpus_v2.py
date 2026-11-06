# -*- coding: utf-8 -*-
"""
正确处理下载的古诗词语料，生成训练文本
"""
import json
import os

corpus_dir = r'D:\desktop\学习结果小文件\corpus'
output_dir = r'D:\desktop\学习结果小文件'

# 加载词表
with open(os.path.join(output_dir, 'merged_vocab.json'), 'r', encoding='utf-8') as f:
    vocab = json.load(f)
vocab_set = set(vocab.get('char_to_idx', {}).keys())
print(f"词表大小: {len(vocab_set)}")

def is_valid_char(c):
    """检查字符是否在词表中"""
    return c in vocab_set

def filter_valid_texts(texts):
    """过滤保留只含词表字符的文本"""
    valid_texts = []
    invalid_count = 0
    for text in texts:
        # 检查所有字符是否都在词表中
        if all(is_valid_char(c) for c in text):
            valid_texts.append(text)
        else:
            invalid_count += 1
    return valid_texts, invalid_count

def extract_texts_from_json(filepath, content_key='paragraphs'):
    """从JSON文件提取文本"""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        # 尝试不同的字段名
        content = item.get(content_key) or item.get('content') or item.get('paragraphs')
        if content and isinstance(content, list):
            texts.extend(content)
    return texts

def process_corpus():
    """处理所有语料文件"""
    all_texts = []
    
    # 1. 诗经 (content字段)
    print("处理诗经...")
    texts = extract_texts_from_json(
        os.path.join(corpus_dir, 'shijing_shijing.json'), 
        content_key='content'
    )
    all_texts.extend(texts)
    print(f"  获取 {len(texts)} 条")
    
    # 2. 论语 (paragraphs字段)
    print("处理论语...")
    texts = extract_texts_from_json(
        os.path.join(corpus_dir, 'lunyu_lunyu.json'),
        content_key='paragraphs'
    )
    all_texts.extend(texts)
    print(f"  获取 {len(texts)} 条")
    
    # 3. 元曲 (paragraphs字段)
    print("处理元曲...")
    texts = extract_texts_from_json(
        os.path.join(corpus_dir, 'yuanqu_yuanqu.json'),
        content_key='paragraphs'
    )
    all_texts.extend(texts)
    print(f"  获取 {len(texts)} 条")
    
    # 4. 全唐诗 (多个文件)
    print("处理全唐诗...")
    for fname in sorted(os.listdir(corpus_dir)):
        if fname.startswith('quantangshi_poet.') and fname.endswith('.json'):
            texts = extract_texts_from_json(
                os.path.join(corpus_dir, fname),
                content_key='paragraphs'
            )
            all_texts.extend(texts)
            print(f"  {fname}: {len(texts)} 条")
    
    # 5. 宋词 (多个文件)
    print("处理宋词...")
    for fname in sorted(os.listdir(corpus_dir)):
        if fname.startswith('songci_ci.') and fname.endswith('.json'):
            texts = extract_texts_from_json(
                os.path.join(corpus_dir, fname),
                content_key='paragraphs'
            )
            all_texts.extend(texts)
            print(f"  {fname}: {len(texts)} 条")
    
    # 过滤空文本
    all_texts = [t.strip() for t in all_texts if t.strip() and len(t.strip()) > 2]
    print(f"\n原始文本总数: {len(all_texts)}")
    
    # 过滤只保留词表中的字符
    valid_texts, invalid_count = filter_valid_texts(all_texts)
    print(f"词表内文本: {len(valid_texts)}")
    print(f"含生僻字被过滤: {invalid_count}")
    
    return valid_texts

def save_training_data(texts):
    """保存训练数据"""
    output_path = os.path.join(output_dir, 'corpus_ancient')
    
    # 1. 每行一条文本
    with open(output_path + '.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    # 2. 合并为连续文本
    combined = '\n\n'.join(texts)
    with open(output_path + '_combined.txt', 'w', encoding='utf-8') as f:
        f.write(combined)
    
    # 3. 按长度排序（短到长，渐进学习）
    texts_by_len = sorted(texts, key=len)
    with open(output_path + '_sorted.txt', 'w', encoding='utf-8') as f:
        for text in texts_by_len:
            f.write(text + '\n')
    
    print(f"\n已保存:")
    print(f"  - {output_path}.txt: {len(texts)} 行")
    print(f"  - {output_path}_combined.txt: {len(combined):,} 字符")
    print(f"  - {output_path}_sorted.txt: 按长度排序")

if __name__ == '__main__':
    texts = process_corpus()
    save_training_data(texts)
    
    # 字符统计
    all_chars = ''.join(texts)
    print(f"\n最终字符统计:")
    print(f"  总字符: {len(all_chars):,}")
    print(f"  唯一字符: {len(set(all_chars)):,}")
    print(f"  平均长度: {len(all_chars)/len(texts):.1f}")
