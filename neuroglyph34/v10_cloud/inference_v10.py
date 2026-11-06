# -*- coding: utf-8 -*-
"""
inference_v10.py — V10 模型推理脚本
加载 best checkpoint，提供交互式和批量文本生成。
"""
import json
import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F

from model_v10 import NeuroGlyph34V10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "char_to_idx.json")
EMBEDDING_PATH = os.path.join(BASE_DIR, "embedding_matrix.npy")


def load_model(checkpoint_path: str):
    char_to_idx = json.load(open(CHAR_TO_IDX_PATH, encoding="utf-8"))
    idx_to_char = {int(v): k for k, v in char_to_idx.items()}

    model = NeuroGlyph34V10(
        vocab_size=7987,
        d_glyph=34,
        d_semantic=64,
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        dropout=0.0,
        max_seq_len=512,
    )

    embedding_matrix = np.load(EMBEDDING_PATH)
    model.load_glyph_matrix(embedding_matrix)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    total, trainable = model.param_count()
    print(f"模型加载完毕: {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch', '?')}, val_ppl={ckpt.get('val_ppl', '?'):.2f}")
    print(f"  参数量: {total:,} ({total/1e6:.1f}M)")

    return model, char_to_idx, idx_to_char


@torch.no_grad()
def generate(
    model,
    char_to_idx: dict,
    idx_to_char: dict,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    max_seq_len: int = 512,
):
    prompt_ids = [char_to_idx.get(ch, 0) for ch in prompt]
    input_ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]

        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature

        if top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, top_k)
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits[topk_idx] = topk_vals

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


def main():
    parser = argparse.ArgumentParser(description="NeuroGlyph34V10 推理脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(BASE_DIR, "checkpoints_v10", "best_model.pt"),
        help="checkpoint 路径",
    )
    parser.add_argument("--prompt", type=str, default="今天天气", help="生成提示词")
    parser.add_argument("--max-tokens", type=int, default=120, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K 采样")
    parser.add_argument("--top-p", type=float, default=0.92, help="Top-P 采样")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] checkpoint 不存在: {args.checkpoint}")
        return

    model, char_to_idx, idx_to_char = load_model(args.checkpoint)

    if args.interactive:
        print("\n交互生成模式 (输入 'quit' 退出)")
        print("-" * 50)
        while True:
            prompt = input("\n> ").strip()
            if prompt.lower() == "quit":
                break
            if not prompt:
                continue
            result = generate(
                model, char_to_idx, idx_to_char, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            print(f"\n{result}")
    else:
        result = generate(
            model, char_to_idx, idx_to_char, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\n[Prompt] {args.prompt}")
        print(f"[生成]  {result}")


if __name__ == "__main__":
    main()
