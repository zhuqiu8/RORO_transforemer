import pandas as pd
import numpy as np

# 原始数据示例
import pandas as pd
import json

# Excel 文件路径
file_path = r"C:\Users\zhuqiu\Desktop\RORO训练数据集\R3布局.xlsx"

# 读取所有工作表的名称
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# 初始化堆场数据结构
raw_data = []

# 遍历每个工作表
for sheet_name in sheet_names:
    # 读取当前工作表的数据
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 按堆场 ID 分组
    for yard_id, group in df.groupby('yard_id'):
        # 提取堆场信息
        yard_info = {
            "yard_id": yard_id,
            "length": 270.0,  # 假设堆场长度为 270 米，保留一位小数
            "width": 14.5,  # 假设堆场宽度为 14.5 米
            "vehicles": [],
            "layout": []
        }

        # 统计车辆信息
        vehicle_counts = group.groupby(['brand', 'model']).size().reset_index(name='count')
        for _, row in vehicle_counts.iterrows():
            vehicle_info = {
                "brand": row['brand'],
                "model": row['model'],
                "length": round(group[(group['brand'] == row['brand']) & (group['model'] == row['model'])]['Height'].iloc[0], 1),  # 保留一位小数
                "width": round(group[(group['brand'] == row['brand']) & (group['model'] == row['model'])]['Width'].iloc[0], 1),  # 保留一位小数
                "count": int(row['count'])  # 转换为 Python int
            }
            yard_info['vehicles'].append(vehicle_info)

        # 提取布局信息
        for _, row in group.iterrows():
            layout_info = {
                "x": round(float(row['X']), 1),  # 保留一位小数
                "y": round(float(row['Y']), 1),  # 保留一位小数
                "rotated": bool(row['Rotated']),  # 转换为 Python bool
                "brand": row['brand'],
                "model": row['model']
            }
            yard_info['layout'].append(layout_info)

        raw_data.append(yard_info)

# 数据清洗函数
def clean_data(data):
    print(f"Original data length: {len(data)}")
    valid_records = []
    for record in data:
        # 有效性检查
        if record['length'] <= 0 or record['width'] <= 0:
            print(f"Skipping record due to invalid dimensions: length={record['length']}, width={record['width']}")
            continue



        # 移除无效布局
        total_area = sum(v['length'] * v['width'] * v['count'] for v in record['vehicles'])
        yard_area = record['length'] * record['width']
        
        # 打印详细的面积信息
        print(f"\nYard {record['yard_id']} area analysis:")
        print(f"Yard dimensions: {record['length']}m x {record['width']}m")
        print(f"Yard area: {yard_area} m²")
        print(f"Total vehicle area: {total_area} m²")
        print(f"Area ratio: {total_area/yard_area:.2f}")
        
        # 放宽面积限制到50%误差
        if total_area > yard_area * 1.5:  # 允许50%误差
            print(f"Skipping record due to area mismatch: total_area={total_area}, yard_area={yard_area}")
            continue

        valid_records.append(record)
    
    print(f"Cleaned data length: {len(valid_records)}")
    return valid_records


cleaned_data = clean_data(raw_data)


# 特征编码示例
def feature_engineering(record):
    features = []

    # 堆场基础特征
    yard_area = record['length'] * record['width']
    features += [
        record['length'] / 450,  # 归一化到0-1
        record['width'] / 30,
        yard_area / (450 * 30)
    ]

    # 车辆组合特征
    for vehicle in record['vehicles']:
        vehicle_area = vehicle['length'] * vehicle['width']
        features += [
            vehicle['count'] / 200,
            vehicle['length'] / 5,
            vehicle['width'] / 3,
            vehicle_area / yard_area,
            #brand_priority_map.get(vehicle['brand'], 1.0)
        ]

    # 填充至固定维度
    return features + [0] * (256 - len(features))


# 标签生成（热力图）
def generate_heatmap(layout, vehicles, yard_size=(450, 32), grid_size=0.1):  # 保持0.1的grid_size
    grid_x = int(yard_size[0] / grid_size)
    grid_y = int(yard_size[1] / grid_size)
    heatmap = np.zeros((grid_x, grid_y))

    # 创建车辆尺寸查找字典
    vehicle_dimensions = {}
    for vehicle in vehicles:
        key = (vehicle['brand'], vehicle['model'])
        vehicle_dimensions[key] = (vehicle['length'], vehicle['width'])

    for car in layout:
        # 获取车辆尺寸
        key = (car['brand'], car['model'])
        length, width = vehicle_dimensions[key]

        # 获取车辆实际占用区域
        if car['rotated']:
            length, width = width, length

        # 计算占用网格范围（考虑车辆实际尺寸）
        x_start = int(car['x'] / grid_size)
        y_start = int(car['y'] / grid_size)
        x_end = min(x_start + int(np.ceil(length / grid_size)), grid_x)
        y_end = min(y_start + int(np.ceil(width / grid_size)), grid_y)

        # 填充更精确的占用区域
        heatmap[x_start:x_end, y_start:y_end] = 1

    return heatmap.flatten()

class DataAugmenter:
    def __init__(self):
        self.prob = 0.5  # 增强概率

    def augment(self, record):
        # 随机镜像
        if np.random.rand() < self.prob:
            record['length'], record['width'] = record['width'], record['length']
            for car in record['layout']:
                car['x'], car['y'] = car['y'], car['x']
                car['length'], car['width'] = car['width'], car['length']

        # 尺寸扰动
        scale = np.random.uniform(0.9, 1.1)
        record['length'] *= scale
        record['width'] *= scale
        for car in record['vehicles']:
            car['length'] *= scale
            car['width'] *= scale

        return record


class DataAugmenter:
    def __init__(self):
        self.prob = 0.5  # 增强概率

    def augment(self, record):
        # 随机镜像
        if np.random.rand() < self.prob:
            record['length'], record['width'] = record['width'], record['length']
            for car in record['layout']:
                car['x'], car['y'] = car['y'], car['x']
                car['length'], car['width'] = car['width'], car['length']

        # 尺寸扰动
        scale = np.random.uniform(0.9, 1.1)
        record['length'] *= scale
        record['width'] *= scale
        for car in record['vehicles']:
            car['length'] *= scale
            car['width'] *= scale

        return record


import torch
import torch.nn as nn

grid_size = 0.1  # 修改为更精细的分辨率
class LayoutTransformer(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=4500*320):  # 修改输出维度以匹配0.1的grid_size
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=6
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x.unsqueeze(0)).squeeze(0)
        return self.decoder(x)


# 数据集划分
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(cleaned_data, test_size=0.2, random_state=42)


# 数据加载器
class LayoutDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        features = feature_engineering(record)
        heatmap = generate_heatmap(record['layout'], record['vehicles'])
        return {
            'features': torch.FloatTensor(features),
            'heatmap': torch.FloatTensor(heatmap)
        }


train_loader = torch.utils.data.DataLoader(
    LayoutDataset(train_data),
    batch_size=32,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    LayoutDataset(val_data),
    batch_size=32,
    shuffle=False
)

# 初始化
model = LayoutTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.BCEWithLogitsLoss()

# 训练循环
for epoch in range(50):
    model.train()
    total_loss = 0

    for batch in train_loader:
        features = batch['features']
        targets = batch['heatmap']

        # 前向传播
        outputs = model(features)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features']
            targets = batch['heatmap']
            outputs = model(features)
            val_loss += criterion(outputs, targets).item()

    # 学习率调整
    scheduler.step()

    print(
        f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")


    def evaluate_model(model, loader):
        model.eval()
        metrics = {
            'iou': [],
            'accuracy': [],
            'utilization_error': []
        }

        with torch.no_grad():
            for batch in loader:
                features = batch['features']
                targets = batch['heatmap']

                outputs = torch.sigmoid(model(features))
                preds = (outputs > 0.5).float()

                # 计算IoU
                intersection = (preds * targets).sum(dim=1)
                union = (preds + targets).sum(dim=1) - intersection
                metrics['iou'].append((intersection / (union + 1e-6)).mean().item())

                # 计算准确率
                correct = (preds == targets).float().mean()
                metrics['accuracy'].append(correct.item())

                # 计算利用率误差
                pred_util = preds.mean()
                true_util = targets.mean()
                metrics['utilization_error'].append(abs(pred_util - true_util).item())

        return {k: np.mean(v) for k, v in metrics.items()}


# 训练完成后保存模型
torch.save(model.state_dict(), 'layout_transformer_weights.pth')