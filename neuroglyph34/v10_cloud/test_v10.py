# -*- coding: utf-8 -*-
"""
test_v10.py — V10 模型测试与验证脚本
从 test_v6_tagged.py 移植适配 V10 架构。
"""
import json
import os
import torch
import numpy as np
import torch.nn.functional as F

from model_v10 import NeuroGlyph34V10

# ──────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx.json")
EMBEDDING_PATH = os.path.join(BASE_DIR, "embedding_matrix.npy")
BEST_CKPT = os.path.join(BASE_DIR, "checkpoints_v10", "best_model.pt")

# 模型超参数 (与训练一致)
VOCAB_SIZE = 7987
D_GLYPH = 34
D_SEMANTIC = 64
D_MODEL = 512
N_LAYERS = 12
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 512


def load_char_mappings():
    char_to_idx = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    idx_to_char = {int(v): k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char


def load_model(checkpoint_path=None):
    char_to_idx, idx_to_char = load_char_mappings()

    model = NeuroGlyph34V10(
        vocab_size=VOCAB_SIZE,
        d_glyph=D_GLYPH,
        d_semantic=D_SEMANTIC,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=0.0,
        max_seq_len=MAX_SEQ_LEN,
    )

    # 加载字形嵌入矩阵
    embedding_matrix = np.load(EMBEDDING_PATH)
    model.load_glyph_matrix(embedding_matrix)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[test_v10] 已加载 checkpoint: {checkpoint_path}")
        print(f"[test_v10] epoch={ckpt.get('epoch', '?')}, val_ppl={ckpt.get('val_ppl', '?'):.2f}")

    model.to(DEVICE)
    model.eval()

    total, trainable = model.param_count()
    print(f"[test_v10] 模型参数量: {total:,} / 可训练 {trainable:,}")
    return model, char_to_idx, idx_to_char


@torch.no_grad()
def generate(model, idx_to_char, prompt_text, max_new_tokens=80, temperature=0.8, top_k=50, top_p=0.92):
    """自回归文本生成"""
    # 构建 token 序列
    char_to_idx, _ = load_char_mappings()
    prompt_ids = []
    for ch in prompt_text:
        if ch in char_to_idx:
            prompt_ids.append(char_to_idx[ch])
        else:
            prompt_ids.append(0)

    input_ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)

    generated = list(prompt_ids)
    for _ in range(max_new_tokens):
        if input_ids.shape[1] > MAX_SEQ_LEN:
            input_ids = input_ids[:, -MAX_SEQ_LEN:]

        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature

        # top-k 过滤
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, top_k)
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits[topk_idx] = topk_vals

        # top-p 过滤
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            next_logits[sorted_idx[remove_mask]] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)

        if next_token == char_to_idx.get("<EOS>", -1):
            break

    return "".join(idx_to_char.get(t, "?") for t in generated)


@torch.no_grad()
def glyph_discrimination_test(model, idx_to_char, pairs_file=None):
    """形近字判别测试"""
    from collections import OrderedDict

    glyph_pairs_data = OrderedDict(
        {
            ("未", "末"): "这两个字只有顶部横的长短不同",
            ("已", "己"): "左下角是否封口",
            ("人", "入"): "捺起笔低则人，高则入",
            ("大", "太"): "右下多一点",
            ("王", "玉"): "右下多一点",
            ("田", "由"): "竖出头方向不同",
            ("甲", "由"): "竖出头方向不同",
            ("刀", "刃"): "刃多一个点",
            ("几", "凡"): "凡多一个点",
            ("干", "千"): "千竖撇",
            ("白", "百"): "百多一横",
            ("牛", "午"): "出头方向不同",
            ("天", "夫"): "撇是否出头",
            ("土", "士"): "横长与竖的关系",
            ("乌", "鸟"): "比白一点少一横",
            ("日", "曰"): "曰比日宽扁",
        }
    )

    # 同样加载 char_to_idx
    char_to_idx, _ = load_char_mappings()

    results = []
    correct = 0
    for (a, b), desc in glyph_pairs_data.items():
        if a not in char_to_idx or b not in char_to_idx:
            continue

        id_a, id_b = char_to_idx[a], char_to_idx[b]

        # 构造上下文: "这个字是[?]"
        ctx = "这个字是"
        ctx_ids = [char_to_idx.get(c, 0) for c in ctx]

        # 分别预测
        input_a = torch.tensor([ctx_ids + [id_a]], device=DEVICE, dtype=torch.long)
        input_b = torch.tensor([ctx_ids + [id_b]], device=DEVICE, dtype=torch.long)

        logit_a = model(input_a)[0, -1, :]
        logit_b = model(input_b)[0, -1, :]

        # 检查模型是否更能预测正确的字
        pred_a = logit_a[char_to_idx.get("末", 0)].item() > logit_a[char_to_idx.get("未", 0)].item()
        pred_b = logit_b[char_to_idx.get("未", 0)].item() > logit_b[char_to_idx.get("末", 0)].item()

        # 简化: 看模型给正确字符的概率
        prob_a_correct = F.softmax(logit_a, dim=-1)[id_a].item()
        prob_b_correct = F.softmax(logit_b, dim=-1)[id_b].item()
        avg_prob = (prob_a_correct + prob_b_correct) / 2

        if avg_prob > 0.1:  # 阈值
            correct += 1
        results.append(f"  {a}↔{b} ({desc}): avg_prob={avg_prob:.4f}")

    total = len(results)
    print(f"\n[形近字判别] 正确率: {correct}/{total} ({100*correct/total:.1f}%)")
    print("\n".join(results))
    return correct, total


def main():
    print("=" * 60)
    print("NeuroGlyph34V10 — 测试脚本")
    print("=" * 60)

    if not os.path.isfile(CHAR_TO_IDX_PATH):
        print(f"[ERROR] 缺少 char_to_idx.json: {CHAR_TO_IDX_PATH}")
        return
    if not os.path.isfile(EMBEDDING_PATH):
        print(f"[ERROR] 缺少 embedding_matrix.npy: {EMBEDDING_PATH}")
        return
    if not os.path.isfile(BEST_CKPT):
        print(f"[WARNING] 未找到 best checkpoint: {BEST_CKPT}")
        print("  将使用随机初始化权重进行测试")
        checkpoint_path = None
    else:
        checkpoint_path = BEST_CKPT

    model, char_to_idx, idx_to_char = load_model(checkpoint_path)

    # 1. 参数统计
    total, trainable = model.param_count()
    print(f"\n模型参数量: {total:,} / 可训练 {trainable:,} (约 {total/1e6:.1f}M)")

    # 2. 文本生成
    print("\n" + "-" * 40)
    print("文本生成测试")
    print("-" * 40)

    prompts = [
        "今天天气",
        "人工智能技术正在快速发展",
        "在中国古代文化中",
        "据最新研究报告显示",
        "这个问题的关键在于",
    ]
    for prompt in prompts:
        generated = generate(model, idx_to_char, prompt, max_new_tokens=60)
        print(f"\n[Prompt] {prompt}")
        print(f"[生成]  {generated}")

    # 3. 形近字判别
    print("\n" + "-" * 40)
    print("形近字判别测试")
    print("-" * 40)
    glyph_discrimination_test(model, idx_to_char)

    print("\n[test_v10] 测试完成")


if __name__ == "__main__":
    main()
