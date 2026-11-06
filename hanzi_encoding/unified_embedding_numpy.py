"""
统一嵌入层 - 纯NumPy版本
不需要任何深度学习框架，直接使用NumPy
"""

import json
import numpy as np


class UnifiedEmbedding:
    """
    统一字符嵌入层 - 纯NumPy实现
    
    使用方法:
        embedding = UnifiedEmbedding(
            vocab_file='merged_vocab.json',
            embedding_file='embedding_matrix.npy'
        )
        
        # 编码文本
        indices = embedding.encode("你好世界")
        print(indices)  # [4, 5, 6, 7]
        
        # 获取嵌入向量
        embeddings = embedding.get_embeddings(indices)
        print(embeddings.shape)  # (4, 18)
    """
    
    def __init__(
        self,
        vocab_file: str = None,
        embedding_file: str = None,
        embedding_dim: int = 18
    ):
        self.embedding_dim = embedding_dim
        
        # 加载词汇表和嵌入矩阵
        if vocab_file and embedding_file:
            self.load(vocab_file, embedding_file)
        else:
            raise ValueError("需要提供词汇表和嵌入矩阵文件")
    
    def load(self, vocab_file: str, embedding_file: str):
        """加载词汇表和嵌入矩阵"""
        
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab_size = vocab_data['meta']['total_vocab_size']
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        
        # 加载嵌入矩阵
        if embedding_file.endswith('.npy'):
            self.embeddings = np.load(embedding_file)
        else:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                self.embeddings = np.array(json.load(f))
        
        print(f"加载完成:")
        print(f"  词汇量: {self.vocab_size}")
        print(f"  嵌入维度: {self.embedding_dim}")
        print(f"  矩阵形状: {self.embeddings.shape}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        将文本编码为索引序列
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 索引数组
        """
        indices = []
        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 1))
            indices.append(idx)
        return np.array(indices, dtype=np.int32)
    
    def decode(self, indices) -> str:
        """
        将索引序列解码为文本
        
        Args:
            indices: 索引数组或列表
            
        Returns:
            str: 解码后的文本
        """
        chars = []
        for idx in indices:
            if isinstance(idx, (np.integer, np.int32, np.int64)):
                idx = int(idx)
            char = self.idx_to_char.get(str(idx), '<UNK>')
            chars.append(char)
        return ''.join(chars)
    
    def get_embedding(self, char: str) -> np.ndarray:
        """
        获取单个字符的嵌入向量
        
        Args:
            char: 输入字符
            
        Returns:
            np.ndarray: 嵌入向量 (embedding_dim,)
        """
        idx = self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 1))
        return self.embeddings[idx]
    
    def get_embeddings(self, indices) -> np.ndarray:
        """
        获取多个索引的嵌入向量
        
        Args:
            indices: 索引数组或列表
            
        Returns:
            np.ndarray: 嵌入向量数组 (len(indices), embedding_dim)
        """
        indices = np.array(indices, dtype=np.int32)
        return self.embeddings[indices]
    
    def get_vector_dim0(self, idx: int) -> np.ndarray:
        """根据索引获取向量"""
        return self.embeddings[idx]
    
    def __call__(self, indices) -> np.ndarray:
        """使对象可调用，等同于get_embeddings"""
        return self.get_embeddings(indices)


# ============== 使用示例 ==============
if __name__ == "__main__":
    # 示例用法
    print("=" * 50)
    print("统一嵌入层 - 纯NumPy版本")
    print("=" * 50)
    
    # 创建嵌入层
    embedding = UnifiedEmbedding(
        vocab_file='merged_vocab.json',
        embedding_file='embedding_matrix.npy'
    )
    
    print("\n" + "=" * 50)
    print("测试编码/解码:")
    print("=" * 50)
    
    # 测试文本
    test_texts = [
        "你好世界",
        "Hello123",
        "甲乙丙丁",
        "<PAD><UNK>"
    ]
    
    for text in test_texts:
        indices = embedding.encode(text)
        decoded = embedding.decode(indices)
        print(f"原文: {text}")
        print(f"编码: {indices}")
        print(f"解码: {decoded}")
        print()
    
    print("=" * 50)
    print("测试获取嵌入向量:")
    print("=" * 50)
    
    # 获取单个字符嵌入
    char = "中"
    vec = embedding.get_embedding(char)
    print(f"字符 '{char}' 的嵌入向量:")
    print(f"  形状: {vec.shape}")
    print(f"  值: {vec}")
    
    print("\n" + "=" * 50)
    print("批量获取嵌入:")
    print("=" * 50)
    
    # 批量获取
    text = "你好"
    indices = embedding.encode(text)
    embeddings = embedding.get_embeddings(indices)
    print(f"文本 '{text}' 的嵌入:")
    print(f"  索引: {indices}")
    print(f"  嵌入形状: {embeddings.shape}")
    print(f"  嵌入:\n{embeddings}")
