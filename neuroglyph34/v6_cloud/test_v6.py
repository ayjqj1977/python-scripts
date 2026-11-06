# -*- coding: utf-8 -*-
"""
V6 纯34模型测试脚本
====================
- 加载 V6 best checkpoint
- 8条现代文提示生成采样
- 18对形近字判别（内部表征余弦相似度）
- PPL 估算（100篇短文）
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 路径 ====================
V6_DIR = os.path.dirname(os.path.abspath(__file__))
V10_DIR = r"D:\desktop\学习结果小文件\neuroglyph34_cloud_v10"
CKPT_PATH = os.path.join(V6_DIR, "checkpoints_v6", "v6_best_ppl13.55_step33480.pt")
CHAR_TO_IDX_PATH = os.path.join(V6_DIR, "char_to_idx.json")
EMB_MATRIX_PATH = os.path.join(V6_DIR, "embedding_matrix.npy")

# 现代文采样提示
MODERN_PROMPTS = [
    "人工智能技术的快速发展正在深刻改变",
    "在当今全球经济格局中中国的角色",
    "量子计算作为一种前沿技术有望",
    "环境保护与可持续发展已成为全球",
    "深度学习模型在自然语言处理领域",
    "随着5G通信技术的普及物联网应用",
    "中国传统文化中的书法艺术体现了",
    "数字化转型已经成为企业提升竞争力",
]

# 18对形近字
SIMILAR_PAIRS = [
    ("人", "入"), ("已", "己"), ("未", "末"), ("干", "千"),
    ("土", "士"), ("目", "且"), ("甲", "由"), ("王", "玉"),
    ("午", "牛"), ("外", "处"), ("大", "太"), ("本", "木"),
    ("白", "百"), ("田", "由"), ("申", "电"), ("休", "体"),
    ("晴", "睛"), ("辨", "辩"),
]

# ==================== 模型定义 (从 train_neuroglyph34_cloud_v6.py 复制) ====================
class Config:
    project_dir = V6_DIR
    vocab_size = 7987  # 训练脚本默认值, 实际从 checkpoint 推断
    special_tokens = 4
    cnn_channels = [64, 64, 64, 64, 64]
    cnn_kernels = [2, 2, 2, 3, 3]
    embed_dims = [16, 16, 16, 16, 32]
    total_cnn = sum(cnn_channels)
    total_emb = sum(embed_dims)
    fusion_input = total_cnn + total_emb
    d_fusion = 516
    n_layers = 12
    n_heads = 12
    d_ff = 2048
    dropout = 0.15
    max_seq_len = 256

class StreamCNN(nn.Module):
    def __init__(self, in_dim, ch1, ch2, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, ch1, kernel_size=kernel_size, padding=kernel_size - 1),
            nn.GELU(),
            nn.Conv1d(ch1, ch2, kernel_size=kernel_size, padding=kernel_size - 1),
            nn.GELU(),
        )
    def forward(self, x, T):
        out = self.conv(x)
        return out[:, :, :T]

class NeuroGlyph34V6(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        cnn_slices = [(0, 5), (5, 9), (10, 13), (16, 26), (26, 32)]
        self.cnn_streams = nn.ModuleList()
        for (s, e), ch2, ks in zip(cnn_slices, cfg.cnn_channels, cfg.cnn_kernels):
            in_dim = e - s
            self.cnn_streams.append(StreamCNN(in_dim, in_dim * 2, ch2, ks))
        self.cnn_slices = cnn_slices
        self.emb_total_strokes = nn.Embedding(37, cfg.embed_dims[0])
        self.emb_radical_extra = nn.Embedding(27, cfg.embed_dims[1])
        self.emb_same_group = nn.Embedding(57, cfg.embed_dims[2])
        self.emb_structure = nn.Embedding(15, cfg.embed_dims[3])
        self.emb_symbol = nn.Embedding(256, cfg.embed_dims[4])
        self.fusion = nn.Sequential(
            nn.Linear(cfg.fusion_input, cfg.d_fusion),
            nn.LayerNorm(cfg.d_fusion),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.pos_encoding = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_fusion))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_fusion, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.ln_final = nn.LayerNorm(cfg.d_fusion)
        self.lm_head = nn.Linear(cfg.d_fusion, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, _ = x.shape
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T).permute(0, 2, 1)
            cnn_outputs.append(feat)
        emb_outputs = [
            self.emb_total_strokes(x[:, :, 9].long().clamp(0, 36)),
            self.emb_radical_extra((x[:, :, 13] + 1).long().clamp(0, 26)),
            self.emb_same_group(x[:, :, 14].long().clamp(0, 56)),
            self.emb_structure(x[:, :, 15].long().clamp(0, 14)),
            self.emb_symbol((x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255)),
        ]
        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)
        fused = self.fusion(merged) + self.pos_encoding[:, :T, :]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(fused, mask=causal_mask)
        hidden = self.ln_final(hidden)
        return self.lm_head(hidden)

    def get_hidden(self, x):
        B, T, _ = x.shape
        cnn_outputs = []
        for (s, e), stream in zip(self.cnn_slices, self.cnn_streams):
            feat = x[:, :, s:e].permute(0, 2, 1)
            feat = stream(feat, T).permute(0, 2, 1)
            cnn_outputs.append(feat)
        emb_outputs = [
            self.emb_total_strokes(x[:, :, 9].long().clamp(0, 36)),
            self.emb_radical_extra((x[:, :, 13] + 1).long().clamp(0, 26)),
            self.emb_same_group(x[:, :, 14].long().clamp(0, 56)),
            self.emb_structure(x[:, :, 15].long().clamp(0, 14)),
            self.emb_symbol((x[:, :, 32] * 16 + x[:, :, 33]).long().clamp(0, 255)),
        ]
        merged = torch.cat(cnn_outputs + emb_outputs, dim=-1)
        fused = self.fusion(merged) + self.pos_encoding[:, :T, :]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(fused, mask=causal_mask)
        return self.ln_final(hidden)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# ==================== 加载资源 ====================
print("=" * 70)
print("NeuroGlyph-34 Cloud V6 测试")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 1. 加载词表
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)
vocab_size = len(char_to_idx)
idx_to_char = {v: k for k, v in char_to_idx.items()}
print(f"词表大小: {vocab_size}")

# 2. 加载嵌入矩阵
embedding_matrix = np.load(EMB_MATRIX_PATH)
print(f"嵌入矩阵: {embedding_matrix.shape}")

# 3. 推断 checkpoint 实际 vocab_size
print(f"\n--- 加载 Checkpoint ---")
ckpt_raw = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
actual_vocab = ckpt_raw["model_state"]["lm_head.weight"].shape[0]
print(f"Checkpoint 实际 vocab_size: {actual_vocab}")
print(f"Checkpoint step: {ckpt_raw['step']}")
print(f"best_loss: {ckpt_raw.get('best_loss', 'N/A')}")

# 4. 用实际 vocab 建 Config 并初始化模型
config = Config()
config.vocab_size = actual_vocab
print(f"\nConfig: d_fusion={config.d_fusion}, layers={config.n_layers}, heads={config.n_heads}")
print(f"Config: d_ff={config.d_ff}, dropout={config.dropout}, max_seq_len={config.max_seq_len}")

model = NeuroGlyph34V6(config).to(device)
total_p, trainable_p = model.param_count()
print(f"模型参数: {total_p:,} 总, {trainable_p:,} 可训练")
model.load_state_dict(ckpt_raw["model_state"])
model.eval()
print("Checkpoint 加载成功!\n")

# ==================== 辅助函数 ====================
def tokens_to_features(token_ids, emb_matrix):
    """将 token_id 序列转为 34 维字形特征向量序列"""
    feat_seq = emb_matrix[token_ids]
    return torch.FloatTensor(feat_seq).unsqueeze(0).to(device)

@torch.no_grad()
def generate(prompt, max_new=40, temperature=0.8, top_k=40):
    """自回归生成"""
    # 将 prompt 转为 token_ids
    tokens = []
    for ch in prompt:
        if ch in char_to_idx:
            tokens.append(char_to_idx[ch])
    if len(tokens) < 2:
        return prompt + " (prompt too short)"
    input_ids = tokens[:min(len(tokens), config.max_seq_len - max_new)]
    generated = list(prompt[:len(input_ids)])
    for _ in range(max_new):
        feat = torch.FloatTensor(embedding_matrix[input_ids]).unsqueeze(0).to(device)
        logits = model(feat)
        next_logits = logits[0, -1, :] / temperature
        tk_vals, tk_idx = torch.topk(next_logits, min(top_k, next_logits.size(0)))
        probs = F.softmax(tk_vals, dim=0)
        chosen = torch.multinomial(probs, 1).item()
        next_token = tk_idx[chosen].item()
        if next_token == 0:
            break
        gen_char = idx_to_char.get(next_token, "?")
        generated.append(gen_char)
        input_ids.append(next_token)
        input_ids = input_ids[-config.max_seq_len:]
    return "".join(generated)

@torch.no_grad()
def get_char_hidden(ch):
    """获取字符的内部表征向量 (Transformer 输出层归一化后)"""
    idx = char_to_idx[ch]
    feat = torch.FloatTensor(embedding_matrix[[idx, idx]]).unsqueeze(0).to(device)
    hidden = model.get_hidden(feat)
    return hidden[0, -1, :]

@torch.no_grad()
def compute_ppl(text, seq_len=256):
    """计算单篇文本的 PPL"""
    tokens = []
    for ch in text:
        if ch in char_to_idx:
            tokens.append(char_to_idx[ch])
    if len(tokens) < seq_len + 1:
        return None
    total_loss = 0.0
    total_tokens = 0
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    for start in range(0, len(tokens) - seq_len, seq_len // 2):
        end = start + seq_len
        if end + 1 > len(tokens):
            break
        input_ids = tokens[start:end]
        target_ids = tokens[start + 1:end + 1]
        feat = torch.FloatTensor(embedding_matrix[input_ids]).unsqueeze(0).to(device)
        target = torch.LongTensor(target_ids).unsqueeze(0).to(device)
        logits = model(feat)
        loss = crit(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        valid = (target != 0).sum().item()
        total_loss += loss.item()
        total_tokens += valid
    if total_tokens == 0:
        return None
    return math.exp(total_loss / total_tokens)

# ==================== 测试 1: 现代文生成采样 ====================
print("=" * 70)
print("测试 1: 现代文生成采样 (8 条)")
print("=" * 70)
for i, prompt in enumerate(MODERN_PROMPTS):
    t0 = time.time()
    result = generate(prompt, max_new=40, temperature=0.8, top_k=40)
    elapsed = time.time() - t0
    print(f"\n[{i+1}] Prompt: {prompt}")
    print(f"    生成: {result}")
    print(f"    耗时: {elapsed:.2f}s")

# ==================== 测试 2: 形近字判别 ====================
print("\n" + "=" * 70)
print("测试 2: 形近字判别 (18 对)")
print("=" * 70)

# 收集所有需要测试的字符并预计算 hidden
all_chars = set()
for a, b in SIMILAR_PAIRS:
    if a in char_to_idx and b in char_to_idx:
        all_chars.add(a)
        all_chars.add(b)

char_hidden_cache = {}
for ch in all_chars:
    char_hidden_cache[ch] = get_char_hidden(ch)

results = []
for a, b in SIMILAR_PAIRS:
    if a not in char_hidden_cache or b not in char_hidden_cache:
        print(f"  {a}-{b}: 字符不在词表中, 跳过")
        continue
    sim = F.cosine_similarity(
        char_hidden_cache[a].unsqueeze(0), char_hidden_cache[b].unsqueeze(0)
    ).item()
    results.append((a, b, sim))

results.sort(key=lambda x: x[2], reverse=True)
print(f"\n{'字符对':<10} {'余弦相似度':<12} {'判定'}")
print("-" * 40)
good = 0  # cosine < 0.5 表示模型能有效区分
bad = 0   # cosine > 0.85 表示难以区分
mid = 0
for a, b, sim in results:
    if sim < 0.5:
        status = "✓ 易区分"
        good += 1
    elif sim > 0.85:
        status = "⚠ 难区分"
        bad += 1
    else:
        status = "~ 一般"
        mid += 1
    print(f"{a}-{b:<8} {sim:.4f}        {status}")

print(f"\n统计: 易区分 {good} 对, 一般 {mid} 对, 难区分 {bad} 对")
print(f"平均余弦相似度: {np.mean([r[2] for r in results]):.4f}")

# ==================== 测试 3: PPL 估算 ====================
print("\n" + "=" * 70)
print("测试 3: PPL 估算 (100 篇短文)")
print("=" * 70)

# 构建 100 篇测试短文 (不同主题的现代文)
test_texts = [
    # 科技类 (20篇)
    "人工智能技术近年来取得了突破性进展深度学习模型在图像识别语音识别和自然语言处理等领域的表现已经超过人类水平",
    "量子计算利用量子比特的叠加和纠缠特性能够以指数级速度解决某些传统计算机难以处理的问题如大数分解和组合优化",
    "区块链技术通过去中心化的分布式账本实现了数据不可篡改和透明可追溯为金融供应链和版权保护等领域带来革命性变化",
    "云计算通过虚拟化技术将计算资源池化按需分配给用户大幅降低了企业信息化成本并提升了业务灵活性和可扩展性",
    "物联网通过传感器和通信技术将物理世界数字化实现了设备之间的互联互通在智能家居智慧城市和工业自动化中广泛应用",
    "大数据分析技术能够从海量数据中挖掘有价值的信息和知识帮助企业优化决策提升运营效率并发现新的商业机会",
    "自动驾驶技术融合了计算机视觉雷达感知和高精度地图等多种技术正在逐步从辅助驾驶向完全无人驾驶方向发展",
    "智能制造通过引入工业机器人物联网和大数据分析实现了生产过程的自动化和智能化大幅提升了制造业效率和质量",
    "第五代移动通信技术提供了超高带宽超低延迟和海量连接能力为增强现实虚拟现实和远程医疗等应用奠定了基础",
    "基因编辑技术的突破使得科学家能够精准修改生物体的遗传信息为疾病治疗作物改良和生物制造开辟了新途径",
    "神经网络是一种模拟人脑结构和功能的计算模型通过大量神经元的连接和权重调整实现了从数据中学习复杂模式的能力",
    "信息安全技术包括加密认证访问控制和安全审计等多个方面旨在保护信息系统免受未授权访问和恶意攻击的威胁",
    "边缘计算将数据处理能力下沉到网络边缘减少了数据传输延迟和带宽消耗特别适用于实时分析和低延迟响应场景",
    "机器人技术集成了机械电子控制计算机和人工智能等多个学科正在从工业制造向服务医疗和教育等领域拓展应用",
    "虚拟现实技术通过头戴式显示设备和交互手柄营造出沉浸式的三维虚拟环境在游戏培训和仿真领域有广泛应用前景",
    "自然语言处理使计算机能够理解生成和分析人类语言包括情感分析机器翻译文本摘要和对话系统等核心任务",
    "可再生能源技术包括太阳能风能水能和生物质能等正在逐步替代化石燃料成为全球能源结构的核心组成部分",
    "三维打印技术通过逐层堆积材料的方式制造物体可以实现复杂结构的快速成型在航空航天医疗和建筑领域应用广泛",
    "增强现实技术将虚拟信息叠加到现实世界中通过智能眼镜或手机屏幕呈现为工业维护教育培训和娱乐提供了新体验",
    "脑机接口技术建立了大脑与外部设备之间的直接通信通道使得瘫痪患者可以通过思维控制假肢或计算机光标移动",
    # 经济商业类 (20篇)
    "数字经济正在成为全球经济增长的新引擎电子商务移动支付和在线教育等新业态深刻改变了人们的生活和消费方式",
    "供给侧结构性改革旨在通过优化生产要素配置提升供给体系质量和效率推动中国经济从高速增长向高质量发展转变",
    "中小企业是国民经济的重要组成部分在促进就业推动创新和增加税收方面发挥着不可替代的作用需要更多政策支持",
    "国际贸易格局正在经历深刻调整区域全面经济伙伴关系协定的签署为亚太地区经济一体化提供了新的制度框架",
    "金融市场在资源配置中发挥着核心作用通过股票债券和基金等工具引导资金流向高效率和创新能力强的企业和行业",
    "品牌建设是企业核心竞争力的重要组成部分优秀的品牌能够赢得消费者信任和市场溢价为企业带来持续的竞争优势",
    "创新驱动发展战略强调科技创新在经济社会发展中的核心地位通过加大研发投入和培育创新人才推动产业升级",
    "消费升级反映了居民收入水平提高和消费观念变化从追求数量转向追求品质个性化定制和体验式消费成为趋势",
    "产业链供应链的稳定性对于经济安全至关重要需要加强关键核心技术攻关降低对单一来源的过度依赖风险",
    "营商环境优化通过简政放权降低市场准入门槛和加强知识产权保护激发了市场主体活力和创新创业热情",
    "货币政策调控需要兼顾经济增长物价稳定和金融风险防范等多重目标灵活运用多种工具保持流动性合理充裕",
    "乡村振兴战略通过发展现代农业改善农村基础设施和提升农民收入推动城乡融合发展和农业农村现代化进程",
    "风险投资在培育科技创新企业中发挥着关键作用为初创企业提供资金支持和管理经验助力科技成果转化和产业化",
    "共享经济通过互联网平台将闲置资源高效匹配给需求方提高了资源利用效率但也面临监管和劳动者权益保护挑战",
    "产业政策在引导资源配置和促进产业升级中具有重要作用但需要与市场竞争机制有机结合避免扭曲市场信号",
    "绿色发展理念强调经济发展与生态环境保护的协调统一通过节能减排和循环经济实现可持续发展目标",
    "跨境电子商务打破了传统贸易的地域限制使得中小企业也能够直接参与全球市场拓展了国际业务发展空间",
    "人力资源管理在现代企业中日益重要包括人才招聘培训发展绩效评估和薪酬激励等多个关键环节",
    "供应链管理通过优化从原材料采购到产品交付的全流程提高了企业运营效率降低了库存和物流成本",
    "企业数字化转型不仅仅是技术升级更是组织架构业务流程和企业文化的全面变革需要顶层设计和全员参与",
    # 文化教育类 (20篇)
    "中华优秀传统文化博大精深蕴含丰富的哲学思想人文精神和道德规范是中华民族的精神命脉和文化根基",
    "书法艺术是中国传统美学的精髓通过笔墨纸砚的巧妙运用将汉字的形态美与书写者的情感意境完美融合",
    "教育改革需要注重培养学生的创新思维批判性思维和解决实际问题的能力而不仅仅是知识的灌输和应试技能",
    "古诗词是中国文学宝库中的璀璨明珠唐诗宋词以其精炼的语言深远的意境和丰富的情感感染了一代又一代读者",
    "高等教育的使命不仅是传授专业知识更要培养学生的社会责任感和家国情怀为国家发展和社会进步贡献力量",
    "非物质文化遗产是民族文化记忆的重要载体通过保护和传承传统技艺民俗和表演艺术维系着民族认同和文化多样性",
    "阅读是获取知识和开阔视野的重要途径良好的阅读习惯能够培养独立思考能力丰富精神世界提升人文素养",
    "互联网技术极大改变了知识传播的方式在线教育平台使得优质教育资源能够跨越地域限制惠及更多学习者",
    "中国画以笔墨为主要表现手段追求意境和气韵的生动与西方绘画的写实风格形成鲜明对比体现了独特的东方美学",
    "终身学习理念强调学习是贯穿人生的持续过程在知识更新加速的时代保持学习能力是个人发展的核心竞争力",
    "博物馆是连接过去与未来的桥梁通过收藏研究和展示文物和艺术品传递历史文化知识提升公众审美和文化素养",
    "戏曲艺术融合了文学音乐舞蹈和美术等多种表现形式京剧昆曲和黄梅戏等剧种各具特色深受群众喜爱",
    "科技与人文的融合发展是当代教育的重要方向既要培养学生的科学素养也要注重人文关怀和审美能力的培育",
    "传统节日承载着中华民族的文化记忆春节中秋节端午节等节日蕴含着团圆感恩和孝道等核心价值观念",
    "语言是文化的载体和思维方式的表现学习外语不仅是掌握一种沟通工具更是了解不同文化的窗口和途径",
    "素质教育旨在促进学生德智体美劳全面发展培养具有健全人格创新精神和实践能力的社会主义建设者和接班人",
    "茶文化在中国有数千年的历史从种植加工到品鉴冲泡形成了一套完整的文化体系体现了中国人对生活品质的追求",
    "校园文化是学校精神风貌的集中体现良好的校园文化氛围能够潜移默化地影响学生的价值观行为习惯和人格塑造",
    "民间艺术来源于劳动人民的生产生活实践剪纸泥塑刺绣等艺术形式质朴自然富有浓郁的生活气息和地域特色",
    "家庭是人生的第一个课堂家庭教育在培养孩子的品德行为习惯和价值观念方面具有学校教育不可替代的作用",
    # 社会民生类 (20篇)
    "医疗保障体系的不断完善为人民群众提供了更加公平可及和高效的医疗服务分级诊疗制度的推进优化了医疗资源配置",
    "城市化进程中需要注重生态环境保护和公共服务配套建设智慧城市通过信息技术提升了城市管理和居民生活质量",
    "人口老龄化是全球面临的共同挑战需要完善养老服务体系发展银发经济和探索长期护理保险制度等多措并举应对",
    "就业是民生之本在经济结构调整和产业升级过程中需要加强职业技能培训和公共就业服务促进更充分更高质量就业",
    "食品安全关系千家万户需要从源头治理过程监管和风险防控等方面构建全链条食品安全保障体系",
    "住房问题是重要的民生问题坚持房子是用来住的不是用来炒的定位完善多主体供给多渠道保障的住房制度",
    "公共交通体系建设对于缓解城市交通拥堵减少碳排放和方便市民出行具有重要意义需要优先发展公交和轨道交通",
    "社区是基层治理的基础单元通过网格化管理居民自治和专业服务相结合提升社区治理效能和居民满意度",
    "生态文明建设关系人民福祉和民族未来需要牢固树立绿水青山就是金山银山的理念推动绿色低碳发展",
    "社会保障制度是社会的安全网通过养老保险医疗保险和最低生活保障等制度安排为人民群众提供基本生活保障",
    "脱贫攻坚取得胜利后需要巩固拓展脱贫攻坚成果与乡村振兴有效衔接防止返贫和建立长效帮扶机制",
    "网络空间治理面临新挑战需要加强个人信息保护打击网络诈骗和整治网络谣言营造清朗的网络空间环境",
    "志愿者服务是社会文明进步的重要标志越来越多的市民参与志愿活动用实际行动传递爱心和传播文明",
    "公共文化服务体系建设满足了人民群众日益增长的精神文化需求图书馆博物馆和基层文化服务中心遍布城乡",
    "心理健康问题受到社会越来越多的关注加强心理健康教育和咨询服务有助于提升全民心理健康水平和幸福感",
    "应急管理体系的完善对于应对自然灾害和突发事件至关重要需要提升监测预警应急响应和救援保障能力",
    "慈善事业是社会保障体系的有益补充通过扶贫济困助学助医和救灾救援等方式帮助困难群体改善生活状况",
    "全民健身运动的推广促进了人民群众身心健康城市步道体育公园和社区健身设施为市民提供了便利的锻炼条件",
    "消费者权益保护涉及产品质量价格欺诈和售后服务等多个方面需要完善法律法规和加强市场监管力度",
    "交通出行方式的多元化给市民提供了更多选择共享单车网约车和轨道交通的融合发展便利了日常出行",
    # 科学研究类 (20篇)
    "生物学研究发现细胞通过复杂的信号传导网络调控基因表达细胞周期和凋亡等基本生命过程的精确机制值得深入研究",
    "物理学的前沿研究正在探索物质的微观结构和宇宙的宏观演化量子场论和弦理论为理解自然界基本力提供了理论框架",
    "气候变化科学研究表明人类活动排放的温室气体是全球变暖的主要原因需要全球合作采取减排和适应措施应对挑战",
    "材料科学的进步推动了新材料的开发和应用纳米材料石墨烯和超导材料等具有优异的物理化学性能和广阔应用前景",
    "天文学观测技术的进步使人类能够探测到更遥远的天体引力波探测和黑洞成像等成就开启了多信使天文学新时代",
    "脑科学是二十一世纪最前沿的研究领域之一通过研究神经元网络连接和脑区功能揭示思维记忆和意识产生的神经基础",
    "数学作为基础科学为其他学科提供了强有力的分析工具从数论代数到拓扑学和概率论在现代科技中发挥着关键作用",
    "化学研究在分子层面揭示了物质的结构性质和转化规律催生了新材料药物和高分子化合物等改变人类生活的技术",
    "海洋科学研究涉及海洋物理化学和生物学等多个方面对于开发利用海洋资源和保护海洋生态环境具有重要指导意义",
    "农业科技通过选育优良品种改良种植技术和推广智能农业装备大幅提高了农作物产量和品质保障了粮食安全",
    "航天技术的快速发展使人类活动范围扩展到太空载人航天月球探测和火星探测任务不断刷新人类探索的新高度",
    "地理信息系统技术整合了遥感测绘和计算机技术为城市规划自然灾害评估和环境监测提供了空间数据分析决策支持",
    "生命科学的发展深化了人类对生命本质的认识基因组学蛋白质组学和代谢组学为精准医疗和药物研发带来了新机遇",
    "环境科学研究揭示了污染物在环境中的迁移转化规律和生态效应为制定环境标准和治理技术提供了科学依据",
    "统计学方法是科学研究中不可或缺的工具通过实验设计数据分析和统计推断帮助研究者从数据中得出可靠结论",
    "古生物研究通过化石记录揭示了生命演化历史和古环境变迁为理解生物多样性的形成和当代气候变化提供了参考",
    "宇宙学研究通过观测宇宙微波背景辐射和星系分布研究宇宙的起源演化以及暗物质和暗能量等重大基本问题",
    "药学研究关注药物在体内的吸收分布代谢和排泄过程为新药研发临床合理用药和个体化治疗提供了理论基础",
    "声学研究涵盖声音的产生传播和接收等物理现象超声波技术在医学诊断工业检测和水下通信等领域有重要应用",
    "生态学研究生物与环境的相互关系维护生物多样性对于生态平衡和人类可持续发展具有不可替代的重要意义",
]

# 确保有 100 篇, 拼接成更长文本以支持 seq_len=64
test_texts = test_texts[:100]
# 每5篇拼接为1段确保长度足够 seq_len=64
merged_texts = []
for g in range(0, len(test_texts), 5):
    merged_texts.append("".join(test_texts[g:g+5]))
print(f"测试文本: {len(test_texts)} 篇 -> 合并为 {len(merged_texts)} 段")
print(f"每段长度: {[len(t) for t in merged_texts[:5]]} ...")

ppls = []
valid_count = 0
t0 = time.time()
for i, text in enumerate(merged_texts):
    ppl = compute_ppl(text, seq_len=64)
    if ppl is not None:
        ppls.append(ppl)
        valid_count += 1
    if (i + 1) % 5 == 0:
        print(f"  已处理 {i+1}/{len(merged_texts)} 段, 当前PPL={ppls[-1] if ppls else 'N/A':.2f}")

elapsed = time.time() - t0
print(f"\n有效 PPL 估算: {valid_count}/{len(merged_texts)} 段")
print(f"总耗时: {elapsed:.2f}s")

ppls_arr = np.array(ppls) if ppls else np.array([])
if len(ppls) > 0:
    print(f"\nPPL 统计:")
    print(f"  均值: {ppls_arr.mean():.2f}")
    print(f"  中位数: {np.median(ppls_arr):.2f}")
    print(f"  最小: {ppls_arr.min():.2f}")
    print(f"  最大: {ppls_arr.max():.2f}")
    print(f"  标准差: {ppls_arr.std():.2f}")

# ==================== 汇总 ====================
print("\n" + "=" * 70)
print("测试汇总")
print("=" * 70)
print(f"模型: V6 (纯34, 5路CNN, 无语义嵌入, d_fusion=384)")
print(f"Checkpoint: v6_best_ppl13.55_step33480.pt")
print(f"实际 vocab_size: {actual_vocab}")
print(f"参数: {total_p:,} 总, {trainable_p:,} 可训练")

print(f"\n生成采样: 8/8 条完成")
print(f"形近字判别: {len(results)} 对, 易区分 {good}, 一般 {mid}, 难区分 {bad}")
if len(ppls) > 0:
    print(f"PPL 估算: {ppls_arr.mean():.2f} ± {ppls_arr.std():.2f} ({valid_count} 段)")
else:
    print("PPL 估算: 无有效结果")

print("\n测试完成!")
