#!/usr/bin/env python3
"""
NeuroGlyph-34 纯NumPy完整架构 (i5-13500H / 16GB)
架构: 5路MLP + 5路Embedding → 融合 → 2层MultiHead Attention → LM Head
"""
import os, json, time, math, random
import numpy as np

# ==================== 配置 ====================
DATA_DIR  = r"D:\Desktop\neuroglyph34_project"
CORPUS    = "corpus_training_combined.txt"  # 1.5M字符，小语料验证
SAVE_PATH = os.path.join(DATA_DIR, "neuroglyph34_np.npz")

SEQ_LEN   = 16      # 序列长度
BATCH     = 8       # 批次
D_MODEL   = 64      # Transformer维度
N_HEADS   = 2       # 注意力头
N_LAYERS  = 2       # Transformer层数
LR        = 0.003
STEPS     = 5000

# ==================== 工具 ====================
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

# ==================== 模块 ====================
class Linear:
    """仿 PyTorch 风格的 Linear 层"""
    def __init__(self, d_in, d_out):
        self.W = np.random.randn(d_in, d_out).astype(np.float32) * math.sqrt(2.0 / d_in)
        self.b = np.zeros(d_out, dtype=np.float32)
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
    
    def forward(self, x):
        self.x = x  # 缓存输入，用于反向传播
        return x @ self.W + self.b
    
    def backward(self, grad, lr, t):
        orig_shape = grad.shape
        grad_2d = grad.reshape(-1, orig_shape[-1])
        x_2d = self.x.reshape(-1, self.x.shape[-1])
        dW = x_2d.T @ grad_2d
        db = grad_2d.sum(axis=0)
        dx = (grad_2d @ self.W.T).reshape(orig_shape[:-1] + (self.W.shape[0],))
        # AdamW
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.m_W = b1 * self.m_W + (1 - b1) * dW
        self.v_W = b2 * self.v_W + (1 - b2) * dW**2
        m_hat = self.m_W / (1 - b1**t)
        v_hat = self.v_W / (1 - b2**t)
        self.W -= lr * (m_hat / (np.sqrt(v_hat) + eps) + 0.001 * self.W)
        
        self.m_b = b1 * self.m_b + (1 - b1) * db
        self.v_b = b2 * self.v_b + (1 - b2) * db**2
        m_hat_b = self.m_b / (1 - b1**t)
        v_hat_b = self.v_b / (1 - b2**t)
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
        
        return dx

class Embedding:
    def __init__(self, num, dim):
        self.W = np.random.randn(num, dim).astype(np.float32) * 0.02
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
    
    def forward(self, ids):
        """ids: (batch,), 返回 (batch, dim)"""
        self.ids = ids
        return self.W[ids]
    
    def backward(self, grad, lr, t):
        """grad: (batch, dim)"""
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.ids, grad)
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.m_W = b1 * self.m_W + (1 - b1) * dW
        self.v_W = b2 * self.v_W + (1 - b2) * dW**2
        m_hat = self.m_W / (1 - b1**t)
        v_hat = self.v_W / (1 - b2**t)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return None  # Embedding 不需要传回梯度

class MLPStream:
    """一路特征流的 2层 MLP"""
    def __init__(self, d_in, d_hidden):
        self.fc1 = Linear(d_in, d_hidden)
        self.fc2 = Linear(d_hidden, d_hidden)
    
    def forward(self, x):
        """x: (batch, d_in) → (batch, d_hidden)"""
        h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)
    
    def backward(self, grad, lr, t):
        dx = self.fc2.backward(grad, lr, t)
        # GELU 导数
        h = gelu(self.fc1.x @ self.fc1.W + self.fc1.b)
        d_gelu = dx * (0.5 * (1 + np.tanh(math.sqrt(2/math.pi) * (h + 0.044715 * h**3)))
                       + 0.5 * h * (1 - np.tanh(math.sqrt(2/math.pi) * (h + 0.044715 * h**3))**2)
                       * math.sqrt(2/math.pi) * (1 + 0.134145 * h**2))
        return self.fc1.backward(d_gelu, lr, t)

class MultiHeadAttention:
    def __init__(self):
        d, h = D_MODEL, N_HEADS
        self.d_k = d // h
        self.Wq = Linear(d, d)
        self.Wk = Linear(d, d)
        self.Wv = Linear(d, d)
        self.Wo = Linear(d, d)
        self.ln1_gamma = np.ones(d, dtype=np.float32)
        self.ln1_beta  = np.zeros(d, dtype=np.float32)
        self.m_ln_g = np.zeros(d)
        self.v_ln_g = np.zeros(d)
        self.m_ln_b = np.zeros(d)
        self.v_ln_b = np.zeros(d)
        self.N_HEADS = h
        self.D_MODEL = d
    
    def forward(self, x):
        """x: (batch, seq, d_model) → (batch, seq, d_model)"""
        B, S, D = x.shape
        h, dk = self.N_HEADS, self.d_k
        
        # 多头 QKV
        self.q = self.Wq.forward(x).reshape(B, S, h, dk).transpose(0,2,1,3)  # (B,h,S,dk)
        self.k = self.Wk.forward(x).reshape(B, S, h, dk).transpose(0,2,1,3)
        self.v = self.Wv.forward(x).reshape(B, S, h, dk).transpose(0,2,1,3)
        
        # 注意力
        scores = self.q @ self.k.transpose(0,1,3,2) / math.sqrt(dk)  # (B,h,S,S)
        # 因果掩码
        mask = np.triu(np.ones((S,S)), k=1) * -1e9
        self.attn = softmax(scores + mask, axis=-1)
        attn_out = (self.attn @ self.v).transpose(0,2,1,3).reshape(B, S, D)
        
        # 输出投影 + 残差 + LayerNorm
        self.wo_output = self.Wo.forward(attn_out)
        self.ln_input = x + self.wo_output  # 缓存残差+投影结果
        self.residual = x
        return layer_norm(self.ln_input, self.ln1_gamma, self.ln1_beta)
    
    def backward(self, grad, lr, t):
        """简化反向：注意力层仅传梯度，不更新QKV权重（FFN承担主要学习）"""
        B, S, D = grad.shape
        
        # LayerNorm 反向
        mean = self.ln_input.mean(axis=-1, keepdims=True)
        var  = self.ln_input.var(axis=-1, keepdims=True)
        eps = 1e-5
        x_hat = (self.ln_input - mean) / np.sqrt(var + eps)
        
        d_gamma = (grad * x_hat).sum(axis=(0,1))
        d_beta  = grad.sum(axis=(0,1))
        dx_hat  = grad * self.ln1_gamma
        
        b1, b2, ee = 0.9, 0.999, 1e-8
        self.m_ln_g = b1 * self.m_ln_g + (1-b1) * d_gamma
        self.v_ln_g = b2 * self.v_ln_g + (1-b2) * d_gamma**2
        self.ln1_gamma -= lr * (self.m_ln_g/(1-b1**t)) / (np.sqrt(self.v_ln_g/(1-b2**t)) + ee)
        
        self.m_ln_b = b1 * self.m_ln_b + (1-b1) * d_beta
        self.v_ln_b = b2 * self.v_ln_b + (1-b2) * d_beta**2
        self.ln1_beta -= lr * (self.m_ln_b/(1-b1**t)) / (np.sqrt(self.v_ln_b/(1-b2**t)) + ee)
        
        # 残差梯度直通（注意力层不更新权重，仅传梯度给下层）
        return dx_hat

class FeedForward:
    def __init__(self):
        d, ff = D_MODEL, D_MODEL * 4
        self.fc1 = Linear(d, ff)
        self.fc2 = Linear(ff, d)
        self.ln2_gamma = np.ones(d, dtype=np.float32)
        self.ln2_beta  = np.zeros(d, dtype=np.float32)
        self.m_ln_g = np.zeros(d)
        self.v_ln_g = np.zeros(d)
        self.m_ln_b = np.zeros(d)
        self.v_ln_b = np.zeros(d)
    
    def forward(self, x):
        self.residual = x
        self.fc1_out = gelu(self.fc1.forward(x))
        self.fc2_out = self.fc2.forward(self.fc1_out)
        self.ln_input = x + self.fc2_out
        return layer_norm(self.ln_input, self.ln2_gamma, self.ln2_beta)
    
    def backward(self, grad, lr, t):
        """用缓存的中间值，不重复计算"""
        # LayerNorm 反向
        mean = self.ln_input.mean(axis=-1, keepdims=True)
        var  = self.ln_input.var(axis=-1, keepdims=True)
        eps = 1e-5
        x_hat = (self.ln_input - mean) / np.sqrt(var + eps)
        dx_hat = grad * self.ln2_gamma
        
        b1, b2, ee = 0.9, 0.999, 1e-8
        d_gamma = (grad * x_hat).sum(axis=(0,1))
        d_beta  = grad.sum(axis=(0,1))
        self.m_ln_g = b1 * self.m_ln_g + (1-b1) * d_gamma
        self.v_ln_g = b2 * self.v_ln_g + (1-b2) * d_gamma**2
        self.ln2_gamma -= lr * (self.m_ln_g/(1-b1**t)) / (np.sqrt(self.v_ln_g/(1-b2**t)) + ee)
        
        self.m_ln_b = b1 * self.m_ln_b + (1-b1) * d_beta
        self.v_ln_b = b2 * self.v_ln_b + (1-b2) * d_beta**2
        self.ln2_beta -= lr * (self.m_ln_b/(1-b1**t)) / (np.sqrt(self.v_ln_b/(1-b2**t)) + ee)
        
        grad_out = dx_hat
        grad_h = self.fc2.backward(grad_out, lr, t)
        # GELU 导数: grad_wrt_pre = grad_h * gelu'(fc1_out)
        pre_gelu = self.residual @ self.fc1.W + self.fc1.b
        cdf = 0.5 * (1.0 + np.tanh(math.sqrt(2/math.pi) * (pre_gelu + 0.044715 * pre_gelu**3)))
        pdf = math.sqrt(2/math.pi) * np.exp(-0.5 * pre_gelu**2) / np.sqrt(2*math.pi) * 2
        # 简化: 使用 erf 近似
        d_gelu = grad_h * (0.5 * (1 + np.tanh(math.sqrt(2/math.pi) * (pre_gelu + 0.044715*pre_gelu**3)))
                          + 0.5 * pre_gelu * (1 - np.tanh(math.sqrt(2/math.pi) * (pre_gelu + 0.044715*pre_gelu**3))**2)
                          * math.sqrt(2/math.pi) * (1 + 0.134145 * pre_gelu**2))
        return self.fc1.backward(d_gelu, lr, t) + dx_hat

class TransformerBlock:
    def __init__(self):
        self.attn = MultiHeadAttention()
        self.ffn  = FeedForward()
    
    def forward(self, x):
        x = self.attn.forward(x)
        x = self.ffn.forward(x)
        return x
    
    def backward(self, grad, lr, t):
        grad = self.ffn.backward(grad, lr, t)
        grad = self.attn.backward(grad, lr, t)
        return grad

# ==================== 模型 ====================
class NeuroGlyph34:
    def __init__(self, embedding_matrix):
        self.vocab_size = embedding_matrix.shape[0]
        self.pretrained = embedding_matrix.astype(np.float32)
        
        # 5路 MLP (替代CNN，等价效果)
        self.stream_sijiao  = MLPStream(5,  32)   # 四角
        self.stream_wubi    = MLPStream(4,  32)   # 五笔
        self.stream_bushou  = MLPStream(3,  16)   # 部首索引
        self.stream_bishun  = MLPStream(10, 48)   # 笔顺
        self.stream_pinyin  = MLPStream(6,  32)   # 拼音
        
        # 5路 Embedding
        self.emb_strokes   = Embedding(37, 8)    # 总笔画 0-36
        self.emb_rad_extra = Embedding(27, 8)    # 部首外笔画
        self.emb_same_grp  = Embedding(57, 8)    # 同部同画
        self.emb_structure = Embedding(15, 8)    # 结构
        self.emb_symbol    = Embedding(256, 16)  # 符号
        
        # 融合层: 32+32+16+48+32+8+8+8+8+16 = 208
        FUSION_IN = 208
        self.fusion = Linear(FUSION_IN, D_MODEL)
        
        # 位置编码
        self.pos_enc = np.random.randn(SEQ_LEN, D_MODEL).astype(np.float32) * 0.02
        
        # Transformer
        self.transformer = [TransformerBlock() for _ in range(N_LAYERS)]
        
        # LM Head
        self.lm_head = Linear(D_MODEL, self.vocab_size)
        
        # 统一管理所有可训练模块
        self._modules = {
            'sijiao': self.stream_sijiao, 'wubi': self.stream_wubi,
            'bushou': self.stream_bushou, 'bishun': self.stream_bishun,
            'pinyin': self.stream_pinyin,
            'fusion': self.fusion, 'lm_head': self.lm_head,
            'pos_enc': self  # 特殊处理
        }
    
    def forward(self, token_ids):
        """token_ids: (batch, seq_len) → logits: (batch, seq_len, vocab_size)"""
        B, S = token_ids.shape
        vecs = self.pretrained[token_ids]  # (B, S, 34)
        
        # 拆分34维 → 各流
        fused_list = []
        for t in range(S):
            v = vecs[:, t, :]  # (B, 34)
            
            s1 = self.stream_sijiao.forward(v[:, 0:5])
            s2 = self.stream_wubi.forward(v[:, 5:9])
            s3 = self.stream_bushou.forward(v[:, 10:13])
            s4 = self.stream_bishun.forward(v[:, 16:26])
            s5 = self.stream_pinyin.forward(v[:, 26:32])
            
            e1 = self.emb_strokes.forward(v[:, 9].astype(np.int32).clip(0, 36))
            e2 = self.emb_rad_extra.forward((v[:, 13] + 1).astype(np.int32).clip(0, 26))
            e3 = self.emb_same_grp.forward(v[:, 14].astype(np.int32).clip(0, 56))
            e4 = self.emb_structure.forward(v[:, 15].astype(np.int32).clip(0, 14))
            sy = (v[:, 32] * 16 + v[:, 33]).astype(np.int32).clip(0, 255)
            e5 = self.emb_symbol.forward(sy)
            
            fused = np.concatenate([s1, s2, s3, s4, s5, e1, e2, e3, e4, e5], axis=1)
            fused_list.append(fused)
        
        fused_seq = np.stack(fused_list, axis=1)  # (B, S, 208)
        fused_seq = self.fusion.forward(fused_seq) + self.pos_enc[:S]  # (B, S, D_MODEL)
        
        # Transformer
        for blk in self.transformer:
            fused_seq = blk.forward(fused_seq)
        
        # 输出
        return self.lm_head.forward(fused_seq)
    
    def backward(self, grad_logits, lr, step):
        """反向传播"""
        grad = self.lm_head.backward(grad_logits, lr, step)
        
        for blk in reversed(self.transformer):
            grad = blk.backward(grad, lr, step)
        
        # 位置编码梯度
        S = grad.shape[1]
        self.pos_enc[:S] -= lr * 0.001 * grad.sum(axis=0)
        
        # 融合层反向
        # 这里简化处理，不反向传播到各流内部（仅更新融合层和Transformer）
        # 完整版需要，但为速度先训练上层
        grad_fusion = self.fusion.backward(grad, lr, step)
        # 暂不反向到子流（冻结底层，仅训练上层）

# ==================== 训练 ====================
def main():
    print(f"=== NeuroGlyph-34 NumPy版 ===")
    
    # 加载数据
    emb = np.load(os.path.join(DATA_DIR, "embedding_matrix.npy"))
    with open(os.path.join(DATA_DIR, "char_to_idx.json"), encoding='utf-8') as f:
        vocab = json.load(f)
    
    corpus_path = os.path.join(DATA_DIR, CORPUS)
    with open(corpus_path, encoding='utf-8') as f:
        text = ''.join([l.strip() for l in f if l.strip()])
    
    print(f"词表: {len(vocab)},  嵌入: {emb.shape}")
    print(f"语料: {len(text)} 字符 ({os.path.getsize(corpus_path)/1024:.0f}KB)")
    print(f"架构: 5MLP+5Emb → {D_MODEL}d, {N_LAYERS}层 {N_HEADS}头 Transformer")
    
    # 模型
    model = NeuroGlyph34(emb)
    
    # 训练
    print(f"\n开始训练 ({STEPS}步, batch={BATCH}, seq={SEQ_LEN})...")
    t0 = time.time()
    for step in range(1, STEPS + 1):
        # 采样
        x_batch, y_batch = [], []
        for _ in range(BATCH):
            idx = random.randint(0, len(text) - SEQ_LEN - 1)
            seq = text[idx:idx + SEQ_LEN + 1]
            x_batch.append([vocab.get(c, 1) for c in seq[:-1]])
            y_batch.append([vocab.get(c, 1) for c in seq[1:]])
        
        x = np.array(x_batch, dtype=np.int32)
        y = np.array(y_batch, dtype=np.int32)
        
        # 前向
        logits = model.forward(x)  # (B, S, vocab_size)
        
        # 交叉熵
        B, S, V = logits.shape
        logits_2d = logits.reshape(-1, V)
        y_flat = y.reshape(-1)
        logits_2d -= logits_2d.max(axis=1, keepdims=True)
        probs = softmax(logits_2d)
        loss = -np.log(probs[np.arange(B*S), y_flat] + 1e-8).mean()
        
        # 反向
        grad = probs
        grad[np.arange(B*S), y_flat] -= 1
        grad /= (B * S)
        grad = grad.reshape(B, S, V)
        
        model.backward(grad, LR, step)
        
        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"Step {step:5d} | Loss: {loss:.4f} | {elapsed:.0f}s")
        
        if step % 2500 == 0:
            np.savez(SAVE_PATH,
                     pos_enc=model.pos_enc,
                     fusion_W=model.fusion.W, fusion_b=model.fusion.b,
                     lm_W=model.lm_head.W, lm_b=model.lm_head.b)
            print(f"  保存检查点: {SAVE_PATH}")
    
    elapsed = time.time() - t0
    print(f"\n完成! 总耗时: {elapsed:.0f}s ({elapsed/STEPS:.2f}s/step)")
    
    # 最终保存
    save_data = {'pos_enc': model.pos_enc}
    for name in ['fusion', 'lm_head']:
        m = getattr(model, name)
        save_data[f'{name}_W'] = m.W
        save_data[f'{name}_b'] = m.b
    for i, blk in enumerate(model.transformer):
        for prefix, sub in [('attn', blk.attn), ('ffn', blk.ffn)]:
            for subname in ['Wq','Wk','Wv','Wo','fc1','fc2']:
                if hasattr(sub, subname):
                    save_data[f'tr{i}_{prefix}_{subname}_W'] = getattr(sub, subname).W
                    save_data[f'tr{i}_{prefix}_{subname}_b'] = getattr(sub, subname).b
    
    np.savez(SAVE_PATH, **save_data)
    print(f"模型保存至: {SAVE_PATH}")

if __name__ == "__main__":
    main()