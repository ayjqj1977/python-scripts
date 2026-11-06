# -*- coding: utf-8 -*-
"""
处理下载的古诗词语料，生成训练文本
"""
import json
import os

corpus_dir = r'D:\desktop\学习结果小文件\corpus'
output_dir = r'D:\desktop\学习结果小文件'

def extract_text_from_poem(item, format_type):
    """从不同格式的诗歌数据中提取文本"""
    texts = []
    
    if format_type == 'shijing':
        # 诗经: content 是诗句列表
        content = item.get('content', [])
        texts.extend(content)
        
    elif format_type == 'lunyu':
        # 论语: paragraphs 是段落列表
        paragraphs = item.get('paragraphs', [])
        texts.extend(paragraphs)
        
    elif format_type == 'yuanqu':
        # 元曲: paragraphs 是段落列表
        paragraphs = item.get('paragraphs', [])
        texts.extend(paragraphs)
        
    elif format_type == 'quantangshi':
        # 全唐诗: paragraphs 是诗句列表
        paragraphs = item.get('paragraphs', [])
        texts.extend(paragraphs)
        
    elif format_type == 'songci':
        # 宋词: paragraphs 是诗句列表
        paragraphs = item.get('paragraphs', [])
        texts.extend(paragraphs)
    
    return texts

def process_corpus():
    """处理所有语料文件"""
    all_texts = []
    
    # 1. 诗经
    print("处理诗经...")
    with open(os.path.join(corpus_dir, 'shijing_shijing.json'), 'r', encoding='utf-8') as f:
        shijing = json.load(f)
    for item in shijing:
        texts = extract_text_from_poem(item, 'shijing')
        all_texts.extend(texts)
    print(f"  诗经: {len(shijing)} 篇, {len(texts)} 句")
    
    # 2. 论语
    print("处理论语...")
    with open(os.path.join(corpus_dir, 'lunyu_lunyu.json'), 'r', encoding='utf-8') as f:
        lunyu = json.load(f)
    for item in lunyu:
        texts = extract_text_from_poem(item, 'lunyu')
        all_texts.extend(texts)
    print(f"  论语: {len(lunyu)} 篇, {len(texts)} 句")
    
    # 3. 元曲
    print("处理元曲...")
    with open(os.path.join(corpus_dir, 'yuanqu_yuanqu.json'), 'r', encoding='utf-8') as f:
        yuanqu = json.load(f)
    for item in yuanqu:
        texts = extract_text_from_poem(item, 'yuanqu')
        all_texts.extend(texts)
    print(f"  元曲: {len(yuanqu)} 篇, {len(texts)} 句")
    
    # 4. 全唐诗 (多个文件)
    print("处理全唐诗...")
    tang_count = 0
    for fname in os.listdir(corpus_dir):
        if fname.startswith('quantangshi_') and fname.endswith('.json'):
            with open(os.path.join(corpus_dir, fname), 'r', encoding='utf-8') as f:
                poems = json.load(f)
            for item in poems:
                texts = extract_text_from_poem(item, 'quantangshi')
                all_texts.extend(texts)
                tang_count += 1
    print(f"  全唐诗: {tang_count} 篇")
    
    # 5. 宋词 (多个文件)
    print("处理宋词...")
    ci_count = 0
    for fname in os.listdir(corpus_dir):
        if fname.startswith('songci_ci.') and fname.endswith('.json'):
            with open(os.path.join(corpus_dir, fname), 'r', encoding='utf-8') as f:
                cis = json.load(f)
            for item in cis:
                texts = extract_text_from_poem(item, 'songci')
                all_texts.extend(texts)
                ci_count += 1
    print(f"  宋词: {ci_count} 篇")
    
    # 过滤空文本
    all_texts = [t.strip() for t in all_texts if t.strip() and len(t.strip()) > 2]
    
    print(f"\n总计: {len(all_texts)} 条有效文本")
    
    return all_texts

def save_training_data(texts, output_path):
    """保存训练数据"""
    # 1. 每行一条文本
    with open(output_path + '.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    # 2. 合并为连续文本（用于GPT训练）
    # 添加特殊分隔符
    combined = '\n\n'.join(texts)
    with open(output_path + '_combined.txt', 'w', encoding='utf-8') as f:
        f.write(combined)
    
    print(f"已保存:")
    print(f"  - {output_path}.txt: {len(texts)} 行")
    print(f"  - {output_path}_combined.txt: {len(combined)} 字符")

if __name__ == '__main__':
    texts = process_corpus()
    
    # 输出文件
    output_path = os.path.join(output_dir, 'corpus_training')
    save_training_data(texts, output_path)
    
    # 统计字符
    all_chars = ''.join(texts)
    unique_chars = set(all_chars)
    print(f"\n字符统计:")
    print(f"  总字符数: {len(all_chars):,}")
    print(f"  唯一字符: {len(unique_chars):,}")
    
    # 检查有多少在词表中
    import json
    with open(os.path.join(output_dir, 'merged_vocab.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    vocab_chars = set(vocab.get('char_to_idx', {}).keys())
    covered = unique_chars & vocab_chars
    missing = unique_chars - vocab_chars
    
    print(f"  词表覆盖: {len(covered)}/{len(unique_chars)} ({100*len(covered)/len(unique_chars):.1f}%)")
    print(f"  未收录字符: {len(missing)}")
    if missing:
        print(f"  示例: {list(missing)[:20]}")
