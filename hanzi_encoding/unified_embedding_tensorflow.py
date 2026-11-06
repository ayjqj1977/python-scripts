"""
统一嵌入层 - TensorFlow/Keras版本
可直接加载并用于深度学习模型
"""

import tensorflow as tf
from tensorflow import keras
import json
import numpy as np


class UnifiedEmbedding(keras.layers.Layer):
    """
    统一字符嵌入层
    支持汉字和符号的嵌入查找
    
    使用方法:
        embedding = UnifiedEmbedding(
            vocab_file='merged_vocab.json',
            embedding_matrix='embedding_matrix.npy',
            mask_zero=True
        )
        
        # 输入字符索引
        input_ids = tf.constant([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5
        embeddings = embedding(input_ids)  # shape: (1, 5, 18)
    """
    
    def __init__(
        self,
        vocab_file: str = None,
        embedding_matrix: str = None,
        embedding_dim: int = 18,
        mask_zero: bool = False,
        trainable: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.embedding_dim = embedding_dim
        self.mask_zero = mask_zero
        
        # 如果提供了词汇表和嵌入文件，则加载
        if vocab_file and embedding_matrix:
            self.load_pretrained(vocab_file, embedding_matrix)
        else:
            # 初始化随机嵌入
            self.vocab_size = 6172  # 默认大小
            self.embedding = keras.layers.Embedding(
                self.vocab_size,
                self.embedding_dim,
                mask_zero=mask_zero,
                trainable=trainable
            )
    
    def load_pretrained(self, vocab_file: str, embedding_matrix: str):
        """加载预训练的嵌入矩阵"""
        
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab_size = vocab_data['meta']['total_vocab_size']
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        
        # 加载嵌入矩阵
        if embedding_matrix.endswith('.npy'):
            embeddings = np.load(embedding_matrix)
        else:
            # 如果是JSON格式
            with open(embedding_matrix, 'r', encoding='utf-8') as f:
                embeddings = np.array(json.load(f))
        
        # 创建嵌入层
        self.embedding = keras.layers.Embedding(
            self.vocab_size,
            self.embedding_dim,
            mask_zero=self.mask_zero,
            trainable=True,
            embeddings_initializer=keras.initializers.Constant(embeddings.astype(np.float32))
        )
    
    def call(self, inputs):
        return self.embedding(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape + (self.embedding_dim,)
    
    def get_embedding(self, char: str) -> tf.Tensor:
        """获取单个字符的嵌入向量"""
        if char in self.char_to_idx:
            idx = self.char_to_idx[char]
            idx_tensor = tf.constant([idx])
            return self.embedding(idx_tensor)[0]
        else:
            # 返回UNK的嵌入
            unk_idx = self.char_to_idx.get('<UNK>', 1)
            idx_tensor = tf.constant([unk_idx])
            return self.embedding(idx_tensor)[0]
    
    def encode(self, text: str) -> tf.Tensor:
        """将文本编码为索引序列"""
        indices = []
        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 1))
            indices.append(idx)
        return tf.constant(indices, dtype=tf.int32)
    
    def decode(self, indices) -> str:
        """将索引序列解码为文本"""
        chars = []
        for idx in indices:
            if hasattr(idx, 'numpy'):
                idx = idx.numpy()
            char = self.idx_to_char.get(idx, '<UNK>')
            chars.append(char)
        return ''.join(chars)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_file': None,
            'embedding_matrix': None,
            'embedding_dim': self.embedding_dim,
            'mask_zero': self.mask_zero,
        })
        return config


def create_embedding_layer(
    vocab_file: str = 'merged_vocab.json',
    embedding_matrix: str = 'embedding_matrix.npy',
    mask_zero: bool = True,
    trainable: bool = True
) -> keras.layers.Embedding:
    """
    创建嵌入层的便捷函数
    
    Args:
        vocab_file: 词汇表文件路径
        embedding_matrix: 嵌入矩阵文件路径
        mask_zero: 是否将0作为mask
        trainable: 是否可训练
    
    Returns:
        keras.layers.Embedding: Keras嵌入层
    """
    embedding = UnifiedEmbedding(
        vocab_file=vocab_file,
        embedding_matrix=embedding_matrix,
        mask_zero=mask_zero,
        trainable=trainable
    )
    return embedding.embedding


# ============== 使用示例 ==============
if __name__ == "__main__":
    # 示例用法
    print("统一嵌入层 - TensorFlow版本")
    print("=" * 50)
    
    # 方法1: 创建带权重的嵌入层
    try:
        emb = UnifiedEmbedding(
            vocab_file='merged_vocab.json',
            embedding_matrix='embedding_matrix.npy',
            mask_zero=True
        )
        
        # 测试编码
        text = "你好世界"
        indices = emb.encode(text)
        print(f"文本: {text}")
        print(f"编码: {indices.numpy()}")
        
        # 获取嵌入
        embeddings = emb(tf.expand_dims(indices, 0))  # 添加batch维度
        print(f"嵌入形状: {embeddings.shape}")
        
        # 测试解码
        decoded = emb.decode(indices.numpy())
        print(f"解码: {decoded}")
        
    except FileNotFoundError:
        print("请先运行 create_embedding_layer.py 生成嵌入矩阵文件")
    
    print("\n使用方法:")
    print("  from unified_embedding_tensorflow import UnifiedEmbedding")
    print("  ")
    print("  # 创建嵌入层")
    print("  embedding = UnifiedEmbedding(")
    print("      vocab_file='merged_vocab.json',")
    print("      embedding_matrix='embedding_matrix.npy'")
    print("  )")
    print("  ")
    print("  # 编码文本")
    print("  input_ids = embedding.encode('你好世界')")
    print("  ")
    print("  # 获取嵌入")
    print("  embeddings = embedding(tf.expand_dims(input_ids, 0))")
