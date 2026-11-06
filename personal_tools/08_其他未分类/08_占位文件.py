
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid  # 加载标准图数据集
from torch_geometric.nn import GCNConv        # GCN核心层

# ====================== 1. 加载数据集（Cora论文分类数据集） ======================
# 自动下载Cora数据集，存放在./data目录，返回数据集对象
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]  # 数据集仅含1张图，直接取第0个元素

# 查看数据集基本信息（课上可对照理解图的结构）
print("数据集信息：")
print(f"图数量：{len(dataset)}")
print(f"节点数：{data.num_nodes}")  # Cora共2708个论文节点
print(f"边数：{data.num_edges}")    # 论文间引用关系边
print(f"特征维度：{data.num_node_features}")  # 每个节点（论文）的特征维度（1433）
print(f"类别数：{dataset.num_classes}")  # 论文分类数（7类）
print(f"训练集节点数：{data.train_mask.sum()}")  # 训练用节点（140个）
print(f"测试集节点数：{data.test_mask.sum()}")   # 测试用节点（1000个）

# ====================== 2. 搭建GCN模型（2层简单架构，入门首选） ======================
class GCN(torch.nn.Module):
    def __init__(self, hidden_dim=16):  # hidden_dim：隐藏层维度，可调整
        super(GCN, self).__init__()
        # 第一层GCN：输入特征维度→隐藏层维度
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim)
        # 第二层GCN：隐藏层维度→输出类别维度（7类）
        self.conv2 = GCNConv(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        # x：节点特征矩阵（shape：[节点数, 特征维度]）
        # edge_index：图的边索引（PyG专用格式，替代邻接矩阵）
        
        # 第一层GCN+激活函数（ReLU避免梯度消失）
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)  # Dropout防止过拟合（训练时生效）
        
        # 第二层GCN（输出类别得分，无需激活）
        x = self.conv2(x, edge_index)
        return x

# ====================== 3. 初始化模型、优化器、损失函数 ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动适配GPU/CPU
model = GCN(hidden_dim=16).to(device)  # 模型实例化并移到设备
data = data.to(device)  # 数据也移到同设备（GPU/CPU）

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # 优化器（Adam常用）
criterion = torch.nn.CrossEntropyLoss()  # 损失函数（分类任务用交叉熵）

# ====================== 4. 模型训练（核心流程，课上重点理解） ======================
def train():
    model.train()  # 模型切换到训练模式（启用Dropout）
    optimizer.zero_grad()  # 清空上一轮梯度
    out = model(data.x, data.edge_index)  # 前向传播：输入特征+边索引→输出类别得分
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 仅计算训练集损失
    loss.backward()  # 反向传播求梯度
    optimizer.step()  # 优化器更新模型参数
    return loss.item()  # 返回本轮损失值

# ====================== 5. 模型测试（评估分类准确率） ======================
def test():
    model.eval()  # 模型切换到测试模式（关闭Dropout）
    out = model(data.x, data.edge_index)  # 前向传播得到所有节点的类别得分
    pred = out.argmax(dim=1)  # 得分最高的类别作为预测结果（argmax取索引）
    
    # 计算训练集、测试集准确率（mask筛选对应节点）
    train_acc = int((pred[data.train_mask] == data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
    test_acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())
    return train_acc, test_acc

# ====================== 6. 执行训练+测试（可视化进度） ======================
if __name__ == "__main__":
    epochs = 200  # 训练轮数（可调整，200轮足够收敛）
    for epoch in range(1, epochs+1):
        loss = train()
        # 每20轮打印一次进度（避免输出过多）
        if epoch % 20 == 0:
            train_acc, test_acc = test()
            print(f"Epoch: {epoch:3d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # 训练结束后打印最终准确率
    final_train_acc, final_test_acc = test()
    print("\n训练完成！")
    print(f"最终训练准确率：{final_train_acc:.4f}")
    print(f"最终测试准确率：{final_test_acc:.4f}")  # 正常收敛可达到80%+