# -*- coding: utf-8 -*-
"""
P2 探针·语义标签预处理器
把 knowledge_graph/ 下 10 个学科域中文知识图谱的字频统计成
每个汉字的「学科域分布」向量(10维软标签)，作为语义头 B 的训练目标。

信号来源：advice.docx 点名的「义类(分类词典/语料)」——这里用本项目自带的
knowledge_graph 学科域语料(纯本地、无需下载)。
"""
import json, os, re, math, collections

BASE = r'D:/desktop/学习结果小文件'
KG   = os.path.join(BASE, 'neuroglyph34_project', 'knowledge_graph')
S32  = os.path.join(BASE, 'neuroglyph34_project', 'struct32_aligned.json')
OUT  = os.path.join(BASE, 'dotmatrix_glyph_spatial', 'semantic_labels.json')

# 10 个域(文件名 -> 可读名)
DOMAINS = [
    ('agricultural_resources_environment_all.csv', '农业资源环境'),
    ('agronomy_all.csv',                            '农学'),
    ('arts_all.csv',                                '艺术'),
    ('history_all.csv',                             '历史'),
    ('horticulture_all.csv',                        '园艺'),
    ('linguistics_all.csv',                         '语言学'),
    ('literature_all.csv',                          '文学'),
    ('other_all.csv',                               '其他'),
    ('philosophy_all.csv',                          '哲学'),
    ('veterinary_medicine_all.csv',                 '兽医'),
]

def load_vocab():
    d = json.load(open(S32, encoding='utf-8'))
    nopy = d['nopy26']                        # 真正的逐字字典在 nopy26 下
    vocab = [k for k in nopy if isinstance(k, str) and '一' <= k <= '鿿']
    return set(vocab), nopy

def main():
    vocab, _ = load_vocab()
    char2idx = {c: i for i, c in enumerate(sorted(vocab))}
    nV = len(char2idx)
    cnt = collections.defaultdict(lambda: [0] * len(DOMAINS))

    cjk_re = re.compile(r'[一-鿿]')
    for di, (fname, name) in enumerate(DOMAINS):
        path = os.path.join(KG, fname)
        if not os.path.exists(path):
            print('SKIP missing', fname); continue
        tot = 0
        with open(path, encoding='utf-8', errors='ignore') as f:
            # 每隔5行采样，平衡大文件(语言学831MB)的运行时间
            for li, line in enumerate(f):
                if li % 5 != 0:
                    continue
                for ch in line:
                    if ch in char2idx:
                        cnt[ch][di] += 1
                        tot += 1
        print(f'[{name}] 命中字频合计 {tot}')

    # 大小校正: 用 P(字|域)=count(c,d)/total_d 而非原始计数占比,
    # 再跨域 softmax(温度放大) -> 突出"相对最超表示"的域, 抵消大语料体量偏差
    totals = [sum(cnt[c][d] for c in vocab) for d in range(len(DOMAINS))]
    nv = len(vocab)
    labels = {}
    for c in vocab:
        v = cnt[c]
        s = sum(v)
        if s > 0:
            ps = [(v[d] + 1.0) / (totals[d] + nv) for d in range(len(DOMAINS))]
            mx = max(ps)
            ex = [math.exp((p - mx) * 3.0) for p in ps]
            Z = sum(ex)
            labels[c] = [round(e / Z, 4) for e in ex]
        else:
            labels[c] = [round(1.0 / len(DOMAINS), 4)] * len(DOMAINS)

    out = {
        'domain_names': [n for _, n in DOMAINS],
        'labels': labels,
        'coverage': sum(1 for c in labels if max(labels[c]) > 0.25),
        'nvocab': nV,
    }
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print('SAVED', OUT, 'coverage', out['coverage'], '/', nV)

    # 抽查信号质量：几个典型字的 top 域
    def top_domains(c, k=3):
        if c not in labels: return None
        order = sorted(range(len(DOMAINS)), key=lambda i: -labels[c][i])
        return [(DOMAINS[i][1], round(labels[c][i], 3)) for i in order[:k]]
    for c in ['牛', '羊', '诗', '帝', '稻', '医', '语', '哲', '花', '金']:
        print(c, '->', top_domains(c))

if __name__ == '__main__':
    main()
