"""
统一嵌入层 - PyTorch版本
可直接加载并用于深度学习模型
"""

import torch
import torch.nn as nn
import json
import numpy as np


class UnifiedEmbedding(nn.Module):
    """
    统一字符嵌入层
    支持汉字和符号的嵌入查找
    
    使用方法:
        embedding = UnifiedEmbedding(
            vocab_file='merged_vocab.json',
            embedding_file='embedding_matrix.npy',
            padding_idx=0
        )
        
        # 输入字符索引
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
        embeddings = embedding(input_ids)  # shape: (1, 5, 18)
    """
    
    def __init__(
        self,
        vocab_file: str = None,
        embedding_file: str = None,
        embedding_dim: int = 18,
        padding_idx: int = 0,
        freeze: bool = False,
        max_norm: float = None,
        norm_type: float = 2.0
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # 如果提供了词汇表和嵌入文件，则加载
        if vocab_file and embedding_file:
            self.load_pretrained(vocab_file, embedding_file)
        else:
            # 初始化随机嵌入
            self.num_embeddings = 6172  # 默认大小
            self.embedding = nn.Embedding(
                self.num_embeddings,
                embedding_dim,
                padding_idx=padding_idx
            )
        
        # 设置是否冻结
        if freeze:
            self.embedding.weight.requires_grad = False
        
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def load_pretrained(self, vocab_file: str, embedding_file: str):
        """加载预训练的嵌入矩阵"""
        
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab_size = vocab_data['meta']['total_vocab_size']
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        
        # 加载嵌入矩阵
        if embedding_file.endswith('.npy'):
            embeddings = np.load(embedding_file)
        else:
            # 如果是JSON格式
            with open(embedding_file, 'r', encoding='utf-8') as f:
                embeddings = np.array(json.load(f))
        
        # 创建嵌入层
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=self.padding_idx
        )
        
        # 加载权重
        self.embedding.weight.data = torch.tensor(embeddings, dtype=torch.float32)
        
        # 冻结padding索引
        if self.padding_idx is not None:
            self.embedding.weight.data[self.padding_idx] = torch.zeros(self.embedding_dim)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: shape (batch_size, seq_len) 或 (seq_len,)
            
        Returns:
            embeddings: shape (batch_size, seq_len, embedding_dim) 或 (seq_len, embedding_dim)
        """
        return self.embedding(input_ids)
    
    def get_embedding(self, char: str) -> torch.Tensor:
        """获取单个字符的嵌入向量"""
        if char in self.char_to_idx:
            idx = self.char_to_idx[char]
            return self.embedding.weight[idx]
        else:
            # 返回UNK的嵌入
            return self.embedding.weight[self.char_to_idx.get('<UNK>', 1)]
    
    def encode(self, text: str) -> torch.Tensor:
        """将文本编码为索引序列"""
        indices = []
        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 1))
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """将索引序列解码为文本"""
        chars = []
        for idx in indices:
            idx = idx.item() if hasattr(idx, 'item') else idx
            char = self.idx_to_char.get(idx, '<UNK>')
            chars.append(char)
        return ''.join(chars)


def create_embedding_layer(
    vocab_file: str = 'merged_vocab.json',
    embedding_file: str = 'embedding_matrix.npy',
    freeze: bool = False
) -> nn.Embedding:
    """
    创建嵌入层的便捷函数
    
    Args:
        vocab_file: 词汇表文件路径
        embedding_file: 嵌入矩阵文件路径
        freeze: 是否冻结嵌入权重
    
    Returns:
        nn.Embedding: PyTorch嵌入层
    """
    embedding = UnifiedEmbedding(
        vocab_file=vocab_file,
        embedding_file=embedding_file,
        freeze=freeze
    )
    return embedding.embedding


# ============== 使用示例 ==============
if __name__ == "__main__":
    # 示例用法
    print("统一嵌入层 - PyTorch版本")
    print("=" * 50)
    
    # 方法1: 创建带权重的嵌入层
    try:
        emb = UnifiedEmbedding(
            vocab_file='merged_vocab.json',
            embedding_file='embedding_matrix.npy',
            padding_idx=0
        )
        
        # 测试编码
        text = "你好世界"
        indices = emb.encode(text)
        print(f"文本: {text}")
        print(f"编码: {indices}")
        
        # 获取嵌入
        embeddings = emb(indices.unsqueeze(0))  # 添加batch维度
        print(f"嵌入形状: {embeddings.shape}")
        
        # 测试解码
        decoded = emb.decode(indices)
        print(f"解码: {decoded}")
        
    except FileNotFoundError:
        print("请先运行 create_embedding_layer.py 生成嵌入矩阵文件")
    
    print("\n使用方法:")
    print("  from unified_embedding_pytorch import UnifiedEmbedding")
    print("  ")
    print("  # 创建嵌入层")
    print("  embedding = UnifiedEmbedding(")
    print("      vocab_file='merged_vocab.json',")
    print("      embedding_file='embedding_matrix.npy'")
    print("  )")
    print("  ")
    print("  # 编码文本")
    print("  input_ids = embedding.encode('你好世界')")
    print("  ")
    print("  # 获取嵌入")
    print("  embeddings = embedding(input_ids.unsqueeze(0))")
