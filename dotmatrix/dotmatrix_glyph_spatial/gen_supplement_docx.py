# -*- coding: utf-8 -*-
"""
论文体「补充实验」文档生成器 —— 收尾 Q1(跨类信号) 与 Q2(浅模型是否埋没几何) 的新发现。
读取 train_multi_mid_result.json 与 dt_cmp_result.json，产出独立补充实验 docx。
"""
import json, os
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = r"D:/desktop/学习结果小文件/dotmatrix_glyph_spatial"
multi = json.load(open(os.path.join(BASE, "train_multi_mid_result.json"), encoding="utf-8"))
dt = json.load(open(os.path.join(BASE, "dt_cmp_result.json"), encoding="utf-8"))

doc = Document()
# 基础字体
st = doc.styles["Normal"]; st.font.name = "宋体"; st.font.size = Pt(11)

def H(txt, lvl=1):
    h = doc.add_heading(txt, level=lvl)
    for r in h.runs: r.font.name = "黑体"
    return h

def P(txt, bold=False, italic=False):
    p = doc.add_paragraph(); r = p.add_run(txt)
    r.bold = bold; r.italic = italic; return p

def TABLE(rows, header):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = h
        for pp in c.paragraphs:
            for rr in pp.runs: rr.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = str(v)
    return t

# ---------- 标题 ----------
ti = doc.add_heading("补充实验：跨大类细粒度信号\n与「浅模型是否埋没几何特征」的验证", level=0)
for r in ti.runs: r.font.name = "黑体"
P("汉字结构化编码 · 可训练判别器系列（二）  |  2026-07-30", italic=True)
P("接前篇《从字形判别到语义距离》实验 D 之后，针对三个顺手想到的方向所做的延伸验证。"
  "方法学与前作一致：输入仍是用户手工设计的可解读字形维度，判别头为本地 CPU 训练的多类"
  "softmax 逻辑回归（零 torch、零云端），按字 80/20 划分以保证真·泛化。")

# ---------- 1. 问题 ----------
H("1. 问题：形旁信号是「实体类专属」吗？浅模型是否埋没了几何特征？", 1)
P("前作实验 D 只在「物(B)」类内部做中类判别，发现字形（主要是部首/形旁）能把测试准确率"
  "从基线 14.9% 拉到约 45%（+30pp），证实了「形旁表义」。这带来两个自然的追问：")
P("（一）其他大类（人、抽象、特征、活动…）内部是否也存在类似细粒度信号？还是说形旁只在"
  "实体类有效？", italic=True)
P("（二）几何特征（s32 矢量排位、det 树向量）贡献甚微，会不会是因为逻辑回归太浅、捕捉不到"
  "几何特征的非线性组合？", italic=True)
P("第二个追问的取舍本身是合理的——我们刻意选浅线性模型以保可解读性；但这里要做的，是用"
  "证据确认这个取舍没有悄悄让结论偏掉。")

# ---------- 2. 跨大类结果 ----------
H("2. 跨大类内部中类判别：信号强弱取决于「中类切分是否对齐形旁语义场」", 1)
P("把实验 D 推广到类内中类充足且相对平衡的大类（A 人 / D 抽象 / E 特征 / H 活动，外加 B 物"
  "作参照）。对每个大类内部，仍用 C_full（283 维）做多类 softmax LR，5-seed 按字 80/20，"
  "报告测试准确率、基线（猜最多中类）与增益，并统计 top-15 特征里「部首 vs 几何」的占比。")

rows = []
for lab in multi["targets"]:
    r = multi["results"].get(lab)
    if not r: continue
    rows.append([lab, r["name"], r["n"], r["n_mid"],
                 f"{r['test_acc']*100:.1f}%", f"{r['baseline_acc']*100:.1f}%",
                 f"{r['gain_pp']:+.1f}pp", f"{r['top_rad_count']}/15"])
TABLE(rows, ["类", "名", "字数", "中类", "测试", "基线", "增益", "top15部首数"])
P("结果出乎「形旁只在实体类有效」的直觉：")
P("• 物(B) 仍是最强信号（+31pp），且 top-15 特征 14 个是部首——形旁表义最干净的样板。", bold=True)
P("• 人(A) 反而**没有信号**（-0.6pp），尽管部首占了 top-15 的 11 个。原因在于「人」类的内部"
  "中类（人体 / 人辈 / 人的品质…）是按语义而非按形旁划分的——「亻」在这些中类里均匀分布，"
  "部首虽是「最重要特征」却区分不开中类。")
P("• 抽象(D, +6.6pp) 与 活动(H, +4.2pp) 有**弱但真实**的信号；特征(E, +0.6pp) 基本无信号。")
P("结论：形旁信号并非「实体类专属」。真正决定有没有信号的是——该大类的内部中类切分，"
  "是否对齐了形旁实际编码的语义场。物类的「木/犭/氵」天然就是语义场，所以信号强；"
  "人/特征类的内部中类不按形旁聚，所以即便部首权重最大也白搭。这是一个比「实体 vs 非实体」"
  "更诚实、也更细粒度的解释。")

# ---------- 3. 浅模型验证 ----------
H("3. 浅模型验证：决策树并未让几何特征「说话」", 1)
P("针对「几何特征是否因逻辑回归太浅被埋没」，在物(B)类内部 18 中类任务上，把浅线性 LR 与"
  "深度受限、可解释的浅决策树（max_depth=4/6）直接对比，并比较两者特征重要性里「部首块」"
  "与「几何块」（结构+笔画+s32+det）的占比。")
dtrows = [
    ["基线(猜最多中类)", f"{dt['baseline']*100:.1f}%", "—", "—"],
    ["LR(线性, 浅)", f"{dt['LR']['acc']*100:.1f}%",
     f"{dt['LR']['rad_share']*100:.1f}%", f"{dt['LR']['geom_share']*100:.1f}%"],
    ["DT(max_depth=4)", f"{dt['DT4']['acc']*100:.1f}%",
     f"{dt['DT4']['rad_share']*100:.1f}%", f"{dt['DT4']['geom_share']*100:.1f}%"],
    ["DT(max_depth=6)", f"{dt['DT6']['acc']*100:.1f}%", "—", "—"],
]
TABLE(dtrows, ["模型", "测试准确率", "部首块占比", "几何块占比"])
P("关键发现：浅决策树（能抓特征交互与非线性）**没有胜过**线性 LR（DT4 反而低 13pp，"
  "在小样本 18 类中浅树更易过拟合）；而几何块重要性占比在 LR 下 19.7%、在 DT 下 24.0%，"
  "几乎没变。", bold=True)
P("这说明：几何特征在「字形→语义类属」这件事上确实弱，**不是模型太浅的锅**。我们刻意选"
  "浅线性模型保可解读性，这个取舍经住了检验——它没有让「几何贡献小」的结论偏掉。")
P("那么几何特征到底该在哪说话？答案是另一个任务：**形近字判别**（判断两个字在视觉上是否"
  "相近）。在那里，s32/det 这类几何形状向量恰恰是对的特征，而部首标签帮不上忙——这将是下"
  "一个值得做的、能真正凸显「预结构化几何特征」价值的判别实验（伪标签可从高 G2-R2 相似字"
  "对或现成形近字表取得）。")

# ---------- 4. 开放数据集 ----------
H("4. 开放数据集：权重即 JSON，免训练可复现", 1)
P("与前作一脉相承的工程取向——把训练好的判别头权重导出为自包含的 `model_for_web.json`："
  "内含标准化统计量、逐字字形组件、大类与物类中类权重，任何人拿到即可零训练复现预测。"
  "配套 `reproduce.py`（`python reproduce.py 树`）与离线网页 `glyph_semantic_predictor.html`，"
  "以及可重跑的训练脚本 `train_discriminator.py` / `train_multi_mid.py` / `dt_cmp.py`，"
  "构成一份「方法透明、可独立验证」的小型开放数据集。")

doc.save(os.path.join(BASE, "汉字结构化编码_补充实验_跨类信号与浅模型验证_2026-07-30.docx"))
print("[OK] 补充实验 docx 已生成")
