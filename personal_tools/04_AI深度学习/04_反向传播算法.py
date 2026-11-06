# 终极优化：按注意力权重关联词，避免重复生成
import math

# 1. 词嵌入（不变）
def word_embedding(word, vocab):
    idx = vocab[word]
    return [idx * 0.1, idx * 0.2]

# 2. LayerNorm归一化（不变）
def layer_norm(x):
    mean = sum(x) / len(x)
    var = sum((i - mean)**2 for i in x) / len(x)
    return [(i - mean) / math.sqrt(var + 1e-6) for i in x]

# 3. 自注意力（不变，保留权重归一化）
def self_attention(seq_emb):
    Q = K = V = seq_emb
    attn_weights = []
    for q in Q:
        weights = [q[0]*k[0] + q[1]*k[1] for k in K]
        weight_sum = sum(weights) + 1e-6
        weights = [w / weight_sum for w in weights]
        attn_weights.append(weights)
    attn_out = []
    for weights in attn_weights:
        out = [0.0, 0.0]
        for i, w in enumerate(weights):
            out[0] += w * V[i][0]
            out[1] += w * V[i][1]
        attn_out.append(out)
    return attn_out, attn_weights

# 4. 关键优化：按注意力权重选下一个词（不重复）
def predict_next_word(attn_weights, input_seq, vocab, reverse_vocab):
    # 取最后一个词的注意力权重（关注和前面词的关联）
    last_weights = attn_weights[-1]
    # 找权重最大的前2个词，取它们的下一个索引（避免重复）
    top2_indices = sorted(range(len(last_weights)), key=lambda i: last_weights[i], reverse=True)[:2]
    # 从这2个词的索引+1，选一个没出现过的
    candidate_indices = [idx + 1 for idx in top2_indices]
    for idx in candidate_indices:
        if idx not in [vocab[w] for w in input_seq] and idx in reverse_vocab:
            return reverse_vocab[idx]
    # 兜底：选第一个没出现过的词
    for idx in range(1, len(vocab)+1):
        if reverse_vocab[idx] not in input_seq:
            return reverse_vocab[idx]

# 主逻辑
if __name__ == "__main__":
    vocab = {"我":1, "爱":2, "豆":3, "包":4, "呀":5, "今":6, "天":7, "真":8, "好":9, "。":10}
    reverse_vocab = {v:k for k,v in vocab.items()}
    
    input_seq = ["我", "爱", "豆", "包"]
    print("初始输入：", "".join(input_seq))
    
    while len(input_seq) < 10:
        seq_emb = [word_embedding(w, vocab) for w in input_seq]
        attn_out, attn_weights = self_attention(seq_emb)  # 拿到注意力权重
        normed_out = layer_norm(attn_out[-1])
        next_word = predict_next_word(attn_weights, input_seq, vocab, reverse_vocab)  # 新预测逻辑
        input_seq.append(next_word)
        print(f"生成后：{''.join(input_seq)}")
    
    print("\n最终生成结果：", "".join(input_seq))
