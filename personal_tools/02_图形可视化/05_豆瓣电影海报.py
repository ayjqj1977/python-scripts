import pandas as pd
import matplotlib.pyplot as plt
import re

# 设置中文字体，避免图表中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取美化后的Excel数据
excel_path = "豆瓣TOP250_美化版.xlsx"
df = pd.read_excel(excel_path)

# ========== 1. 数据预处理 ==========
# 提取评分并转换为数值类型
df["评分"] = pd.to_numeric(df["评分"], errors="coerce")
# 过滤无效评分数据
df = df[df["评分"].notna()]

# 区分中外作者（通过作者字段中的国家标识）
def is_foreign_author(author):
    # 匹配带[国家]标识的作者，判定为外国作者
    if re.search(r"\[.*\]", str(author)):
        return "外国作者"
    else:
        return "中国作者"

df["作者类型"] = df["作者"].apply(is_foreign_author)

# ========== 2. 绘制评分分布柱状图 ==========
plt.figure(figsize=(12, 6))
# 按评分区间分组统计
score_bins = [9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7]
score_labels = ["9.0-9.1", "9.1-9.2", "9.2-9.3", "9.3-9.4", "9.4-9.5", "9.5-9.6", "9.6-9.7"]
df["评分区间"] = pd.cut(df["评分"], bins=score_bins, labels=score_labels, right=False)
score_count = df["评分区间"].value_counts().sort_index()

# 绘制柱状图
ax1 = plt.subplot(1, 2, 1)
score_count.plot(kind="bar", color="#FF6B6B", ax=ax1)
ax1.set_title("豆瓣TOP250书籍评分分布", fontsize=14, pad=20)
ax1.set_xlabel("评分区间", fontsize=12)
ax1.set_ylabel("书籍数量", fontsize=12)
# 给柱子添加数值标签
for i, v in enumerate(score_count):
    ax1.text(i, v + 0.5, str(v), ha="center", fontsize=10)

# ========== 3. 绘制中外作者占比饼图 ==========
ax2 = plt.subplot(1, 2, 2)
author_count = df["作者类型"].value_counts()
colors = ["#4ECDC4", "#45B7D1"]
wedges, texts, autotexts = ax2.pie(
    author_count, labels=author_count.index, autopct="%1.1f%%",
    colors=colors, startangle=90, textprops={"fontsize": 12}
)
ax2.set_title("豆瓣TOP250书籍中外作者占比", fontsize=14, pad=20)
# 美化饼图数值标签
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("bold")

# 调整子图间距，保存图表
plt.tight_layout()
plt.savefig("豆瓣TOP250书籍分析图.png", dpi=300, bbox_inches="tight")
plt.show()

print("可视化图表已保存为：豆瓣TOP250书籍分析图.png")