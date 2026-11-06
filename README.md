# python-scripts

作者个人原创 Python 脚本仓库：汉字结构化编码研究、NeuroGlyph 向量模型训练管线、点阵字形空间实验与个人工具脚本。

## 目录结构

| 目录 | 内容 | 文件数 |
|------|------|--------|
| hanzi_encoding | 汉字 18/32 维向量统一嵌入层（PyTorch / TensorFlow / NumPy）及语料处理管线 | 19 |
| neuroglyph18 | NeuroGlyph18 多尺度 CNN+Transformer 训练管线（4_train / 5_test / 云端训练系列） | 53 |
| neuroglyph34 | NeuroGlyph-34 混合架构 v1~v12（本地项目 + 云端迭代系列） | 30 |
| dotmatrix | 点阵字形与空间结构实验：HZK32 点阵解析、PointNet、空间自编码器、字形探针系列 | 86 |
| personal_tools | 个人工具脚本：数学计算、图形可视化、生活工具、AI 学习、游戏、姓名处理、文件整理、个人归档 | 95 |

合计 283 个 Python 脚本，均为个人原创。

## 说明

- 仅搬运源码（.py），语料数据、模型权重（npy/ckpt）等大文件不入库。
- 已排除过程版本（train_extended 系列、cleanup_bookmarks 早期版等）。
- 下载类脚本数据源为公开语料库，脚本本身为自写。
- 漫剧 / 短剧策划类文件暂不上传。
