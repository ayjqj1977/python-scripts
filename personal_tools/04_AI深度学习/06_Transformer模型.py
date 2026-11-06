import torch
import torch.nn as nn

class Your18DimSmartModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ================= 1. 你的18维嵌入层 =================
        self.embedding = nn.Embedding(
            num_embeddings=6166,
            embedding_dim=18,
            sparse=False
        )

        # ================= 2. 按你逻辑分组 =================
        # 分组1：四角(5) + 五笔(4) = 9维（字形）
        # 分组2：拼音(6) + 声调(1) = 7维（音形）
        # 分组3：符号(2) = 2维（标记）
        # 我们不拆参数，直接在 forward 里切片

        # ================= 3. 分组卷积（学习你说的规律） =================
        # 字形组(9维) 用卷积核=3 学习局部模式（四角+五笔关联）
        self.cnn_shape = nn.Conv1d(
            in_channels=9,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # 音系组(7维) 用卷积核=3 学习拼音+声调组合
        self.cnn_phone = nn.Conv1d(
            in_channels=7,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # 符号组(2维) 用卷积核=1 简单映射
        self.cnn_symbol = nn.Conv1d(
            in_channels=2,
            out_channels=32,
            kernel_size=1
        )

        # ================= 4. 融合 + Transformer =================
        # 融合后总通道 64+64+32 = 160
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=160,
                nhead=8,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        )

        # ================= 5. 输出层（全局向量） =================
        self.head = nn.Linear(160, 512)

    def forward(self, x):
        # x: [batch, seq_len]

        # 1. 嵌入
        embed = self.embedding(x)  # [B, L, 18]

        # 2. 按你的逻辑分组
        # 切分：0-5(四角), 5-9(五笔), 9-15(拼音), 15-16(声调), 16-18(符号)
        shape_part = embed[:, :, :9]      # 9维 (四角+五笔)
        phone_part = embed[:, :, 9:16]   # 7维 (拼音+声调)
        symbol_part = embed[:, :, 16:]    # 2维 (符号)

        # 3. 分组卷积 (Conv1d 需要 [B, C, L])
        shape_out = self.cnn_shape(shape_part.transpose(1,2)) # [B,64,L]
        phone_out = self.cnn_phone(phone_part.transpose(1,2)) # [B,64,L]
        symbol_out = self.cnn_symbol(symbol_part.transpose(1,2)) # [B,32,L]

        # 4. 拼接融合
        fused = torch.cat([shape_out, phone_out, symbol_out], dim=1) # [B, 160, L]
        fused = fused.transpose(1,2) # [B, L, 160]

        # 5. Transformer + 输出
        trans_out = self.transformer(fused)
        global_feat = self.head(trans_out.mean(dim=1)) # [B,512]

        return global_feat

# ================= 测试运行 =================
model = Your18DimSmartModel()
x = torch.randint(0, 6166, (4, 20))
print(model(x))