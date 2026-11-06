#!/usr/bin/env python3
"""
NeuroGlyph-34 混合架构训练脚本
输入: 34维向量 + 大规模语料
架构: 5路CNN + 5路Embedding → 融合 → 4层Transformer → LM Head
"""
import os, sys, json, math, time, random, logging, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# 配置
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 参数
class Config:
    # 数据
    data_dir = r"D:\Desktop\neuroglyph34_project"
    embedding_file = "embedding_matrix.npy"
    vocab_file = "char_to_idx.json"
    corpus_file = "corpus_combined.txt"  # 285MB 混合语料
    # 模型
    d_input = 34
    vocab_size = 7987
    special_tokens = 4  # PAD/UNK/SOS/EOS 索引 0-3
    # CNN流
    cnn_channels = [32, 32, 16, 64, 32]  # 四角/五笔/部首/笔顺/拼音
    cnn_kernels = [2, 2, 2, 3, 3]
    # Embedding流
    embed_dims = [8, 8, 8, 8, 16]  # 总笔画/外笔画/同部同画/结构/符号
    # 融合后
    d_fusion = 128
    n_layers = 4
    n_heads = 4
    d_ff = 512
    dropout = 0.1
    # 训练
    batch_size = 64
    seq_len = 128
    lr = 3e-4
    warmup_steps = 2000
    total_steps = 100_000
    eval_interval = 1000
    save_interval = 5000
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 0 if sys.platform == 'win32' else 4

config = Config()

# ==================== 数据集 ====================
class CharCorpusDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len=128):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        self.text = ''.join(lines)
        self.vocab = vocab
        self.seq_len = seq_len
        self.length = len(self.text) - seq_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        seq = self.text[idx:idx+self.seq_len+1]
        ids = [self.vocab.get(ch, 1) for ch in seq]  # 1=UNK
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

# ==================== 模型架构 ====================
class MultiStreamCNN(nn.Module):
    """5路CNN处理不同维组"""
    def __init__(self, d_input=34):
        super().__init__()
        # 流1: 四角(0-4) 5维
        self.stream1 = nn.Sequential(
            nn.Conv1d(1, config.cnn_channels[0], kernel_size=config.cnn_kernels[0], padding=1),
            nn.ReLU(),
            nn.Conv1d(config.cnn_channels[0], config.cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 流2: 五笔(5-8) 4维
        self.stream2 = nn.Sequential(
            nn.Conv1d(1, config.cnn_channels[1], kernel_size=config.cnn_kernels[1], padding=1),
            nn.ReLU(),
            nn.Conv1d(config.cnn_channels[1], config.cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 流3: 部首索引(10-12) 3维
        self.stream3 = nn.Sequential(
            nn.Conv1d(1, config.cnn_channels[2], kernel_size=config.cnn_kernels[2], padding=1),
            nn.ReLU(),
        )
        # 流4: 笔顺(16-25) 10维
        self.stream4 = nn.Sequential(
            nn.Conv1d(1, config.cnn_channels[3], kernel_size=config.cnn_kernels[3], padding=2),
            nn.ReLU(),
            nn.Conv1d(config.cnn_channels[3], config.cnn_channels[3]//2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.cnn_channels[3]//2, config.cnn_channels[3], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 流5: 拼音(26-31) 6维
        self.stream5 = nn.Sequential(
            nn.Conv1d(1, config.cnn_channels[4], kernel_size=config.cnn_kernels[4], padding=1),
            nn.ReLU(),
        )
        
        # Embedding流
        self.embed_total_strokes = nn.Embedding(37, config.embed_dims[0])  # 0-36
        self.embed_radical_extra = nn.Embedding(27, config.embed_dims[1])  # -1~25 → 0-26
        self.embed_same_group = nn.Embedding(57, config.embed_dims[2])     # 0-56
        self.embed_structure = nn.Embedding(15, config.embed_dims[3])      # 0-14
        self.embed_symbol = nn.Embedding(256, config.embed_dims[4])        # 符号1*16+符号2
        
        # 融合层
        total_cnn = sum(config.cnn_channels)
        total_embed = sum(config.embed_dims)
        self.fusion = nn.Linear(total_cnn + total_embed, config.d_fusion)
        
    def forward(self, x):  # x: [batch, seq_len, 34]
        # 各流CNN: 输入已为 [B, C_in=1, feature_len]
        s1 = self.stream1(x[:, :, 0:5]).mean(dim=2)    # [B, 32]
        s2 = self.stream2(x[:, :, 5:9]).mean(dim=2)    # [B, 32]
        s3 = self.stream3(x[:, :, 10:13]).mean(dim=2)  # [B, 16]
        s4 = self.stream4(x[:, :, 16:26]).mean(dim=2)  # [B, 64]
        s5 = self.stream5(x[:, :, 26:32]).mean(dim=2)  # [B, 32]
        
        # Embedding流: 标量取每个序列位置均值后embed
        total_strokes = x[:, :, 9].long().clamp(0, 36).squeeze(1)
        radical_extra = (x[:, :, 13] + 1).long().clamp(0, 26).squeeze(1)
        same_group = x[:, :, 14].long().clamp(0, 56).squeeze(1)
        structure = x[:, :, 15].long().clamp(0, 14).squeeze(1)
        symbol_idx = (x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255).squeeze(1)
        
        e1 = self.embed_total_strokes(total_strokes)
        e2 = self.embed_radical_extra(radical_extra)
        e3 = self.embed_same_group(same_group)
        e4 = self.embed_structure(structure)
        e5 = self.embed_symbol(symbol_idx)
        
        # 融合: [B, 176] + [B, 48] → [B, 128]
        combined = torch.cat([s1, s2, s3, s4, s5, e1, e2, e3, e4, e5], dim=1)
        return self.fusion(combined)

class NeuroGlyph34(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_fusion = config.d_fusion
        
        # 冻结的嵌入表
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True,
            padding_idx=0
        )
        
        # 多流CNN
        self.multistream = MultiStreamCNN()
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, config.d_fusion) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_fusion,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # LM Head
        self.lm_head = nn.Linear(config.d_fusion, config.vocab_size)
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        # 1. 查表得34维向量
        x = self.embedding(input_ids)  # [B, S, 34]
        
        # 2. 多流CNN + Embedding → 每个token的融合特征
        batch, seq = x.shape[0], x.shape[1]
        fused_list = []
        for i in range(seq):
            token_vec = x[:, i, :]  # [B, 34]
            fused = self.multistream(token_vec.unsqueeze(1))  # [B, d_fusion]
            fused_list.append(fused)
        fused_seq = torch.stack(fused_list, dim=1)  # [B, S, d_fusion]
        
        # 3. 加位置编码
        fused_seq = fused_seq + self.pos_embed[:, :seq, :]
        
        # 4. Transformer
        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        transformer_out = self.transformer(fused_seq, mask=mask)
        
        # 5. LM Head
        logits = self.lm_head(transformer_out)
        return logits

# ==================== 训练循环 ====================
def train_epoch(model, loader, optimizer, scaler, scheduler, step):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(config.device), y.to(config.device)
        optimizer.zero_grad()
        
        with autocast(enabled=(config.device=='cuda')):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        step += 1
        
        if step % 100 == 0:
            logger.info(f"Step {step:6d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if step >= config.total_steps:
            break
    
    return total_loss / len(loader), step

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    logger.info("=== NeuroGlyph-34 训练开始 ===")
    logger.info(f"设备: {config.device}")
    
    # 加载数据
    embedding_path = os.path.join(config.data_dir, config.embedding_file)
    vocab_path = os.path.join(config.data_dir, config.vocab_file)
    corpus_path = os.path.join(config.data_dir, config.corpus_file)
    
    logger.info(f"加载嵌入矩阵: {embedding_path}")
    embedding_matrix = np.load(embedding_path).astype(np.float32)
    logger.info(f"嵌入矩阵形状: {embedding_matrix.shape}")
    
    logger.info(f"加载词表: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    logger.info(f"词表大小: {len(vocab)}")
    
    logger.info(f"加载语料: {corpus_path}")
    dataset = CharCorpusDataset(corpus_path, vocab, config.seq_len)
    logger.info(f"语料长度: {len(dataset.text)} 字符, 样本数: {len(dataset)}")
    
    # 数据加载器
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 模型
    model = NeuroGlyph34(embedding_matrix).to(config.device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # 学习率调度
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        return max(0.0, float(config.total_steps - current_step) / float(config.total_steps - config.warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=(config.device=='cuda'))
    
    # 训练
    step = 0
    best_loss = float('inf')
    checkpoint_dir = os.path.join(config.data_dir, "checkpoints_34")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("开始训练...")
    while step < config.total_steps:
        epoch_loss, step = train_epoch(model, loader, optimizer, scaler, scheduler, step)
        logger.info(f"Epoch 平均损失: {epoch_loss:.4f}")
        
        # 评估和保存
        if step % config.eval_interval == 0:
            eval_loss = evaluate(model, loader)
            logger.info(f"评估损失: {eval_loss:.4f}")
            if eval_loss < best_loss:
                best_loss = eval_loss
                ckpt_path = os.path.join(checkpoint_dir, f"neuroglyph34_best.pth")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, ckpt_path)
                logger.info(f"保存最佳模型: {ckpt_path}")
        
        if step % config.save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"neuroglyph34_step{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, ckpt_path)
            logger.info(f"保存检查点: {ckpt_path}")
    
    logger.info("训练完成!")
    
    # 最终保存
    final_path = os.path.join(config.data_dir, "neuroglyph34_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"最终模型保存至: {final_path}")

if __name__ == "__main__":
    main()