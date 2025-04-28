from main import *
# 加载模型参数（使用更安全的方式）
model = LayoutTransformer()
model.load_state_dict(torch.load('layout_transformer_weights.pth', weights_only=True))
model.eval()

# 假设有一个新的堆场记录
new_yard_record = {
    "yard_id": "new_yard_001",
    "length": 450,
    "width": 32,
    "vehicles": [
        {
            "brand": "KIA",
            "model": "NIRO",
            "length": 4.7,
            "width": 2.2,
            "count": 0
        },
        {
            "brand": "KIA",
            "model": "STONIC",
            "length": 4.5,
            "width": 2.3,
            "count": 100
        }
    ],
    "layout": []  # 空布局，因为我们希望模型生成它
}

# 对数据进行相同的特征工程处理
features = feature_engineering(new_yard_record)
features_tensor = torch.FloatTensor(features).unsqueeze(0)  # 添加batch维度

with torch.no_grad():
    # 获取模型输出
    output = model(features_tensor)
    # 应用sigmoid并二值化
    heatmap = torch.sigmoid(output).squeeze(0) > 0.5
    heatmap = heatmap.numpy().reshape(450, 32)  # 重塑为热力图形状


def heatmap_to_layout(heatmap, vehicle_types, grid_size=1.0):
    """
    将热力图转换为实际的车辆布局

    参数:
        heatmap: 二维numpy数组，表示占用情况
        vehicle_types: 车辆类型列表，包含品牌、型号和数量
        grid_size: 每个网格单元的实际大小(米)

    返回:
        布局列表，包含每个车辆的位置和方向
    """
    layout = []

    # 首先计算每种车辆需要放置的数量
    vehicles_to_place = []
    for vehicle in vehicle_types:
        vehicles_to_place.extend([(vehicle['brand'], vehicle['model'],
                                   vehicle['length'], vehicle['width'])]
                                 * vehicle['count'])

    # 找到所有被占用的网格
    occupied = np.argwhere(heatmap)

    placed_vehicles = []  # 保存已放置车辆的位置和尺寸

    for vehicle in vehicles_to_place:
        brand, model, length, width = vehicle
        placed = False

        # 尝试多次随机位置
        for _ in range(100):  # 最多尝试100次
            # 随机选择一个被占用的网格
            if len(occupied) == 0:
                break

            idx = np.random.randint(0, len(occupied))
            grid_x, grid_y = occupied[idx]
            x, y = grid_x * grid_size, grid_y * grid_size

            # 随机决定是否旋转
            rotated = np.random.rand() > 0.5
            actual_length = width if rotated else length
            actual_width = length if rotated else width

            # 检查碰撞
            collision = False
            for placed_vehicle in placed_vehicles:
                if check_collision(
                        (x, y, actual_length, actual_width),
                        placed_vehicle
                ):
                    collision = True
                    break

            if not collision:
                placed_vehicles.append((x, y, actual_length, actual_width))
                layout.append({
                    "x": float(x),
                    "y": float(y),
                    "rotated": rotated,
                    "brand": brand,
                    "model": model
                })
                placed = True
                break

        if not placed:
            print(f"警告：无法为 {brand} {model} 找到合适位置")

    return layout


def check_collision(rect1, rect2):
    """检查两个矩形是否重叠"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1)

# 使用函数转换热力图
predicted_layout = heatmap_to_layout(
    heatmap,
    new_yard_record['vehicles'],
    grid_size=1.0
)

print(f"生成的布局包含 {len(predicted_layout)} 辆车")

import matplotlib.pyplot as plt


def debug_visualization(heatmap, layout, yard_size=(450, 32)):
    plt.figure(figsize=(15, 5))

    # 原始热力图
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap.reshape(yard_size).T, cmap='hot')
    plt.title("模型预测的热力图")

    # 实际生成的布局
    plt.subplot(1, 2, 2)
    visualize_layout(yard_size[0], yard_size[1], layout)

    plt.show()

def visualize_layout(yard_length, yard_width, layout):
    """可视化车辆布局"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制堆场边界
    ax.add_patch(plt.Rectangle((0, 0), yard_length, yard_width,
                               fill=False, edgecolor='black', linewidth=2))

    # 绘制每辆车
    for car in layout:
        x, y = car['x'], car['y']
        if car['rotated']:
            length, width = car['width'], car['length']
        else:
            length, width = car['length'], car['width']

        ax.add_patch(plt.Rectangle((x, y), length, width,
                                   fill=True, alpha=0.5, edgecolor='blue'))
        ax.text(x + length / 2, y + width / 2, f"{car['brand']}\n{car['model']}",
                ha='center', va='center', fontsize=8)

    ax.set_xlim(0, yard_length)
    ax.set_ylim(0, yard_width)
    ax.set_aspect('equal')
    ax.set_title("预测的车辆布局")
    plt.xlabel("长度 (米)")
    plt.ylabel("宽度 (米)")
    plt.grid(True)
    plt.show()


# 为可视化准备车辆尺寸信息
vehicle_dimensions = {}
for vehicle in new_yard_record['vehicles']:
    brand = vehicle['brand']
    model = vehicle['model']
    vehicle_dimensions[(brand, model)] = (vehicle['length'], vehicle['width'])

# 为布局中的每辆车添加尺寸信息
for car in predicted_layout:
    car['length'], car['width'] = vehicle_dimensions[(car['brand'], car['model'])]

# 可视化
visualize_layout(new_yard_record['length'], new_yard_record['width'], predicted_layout)
# 在预测后调用
heatmap = torch.sigmoid(model(features_tensor)).squeeze(0).numpy()
predicted_layout = heatmap_to_layout(heatmap > 0.5, new_yard_record['vehicles'])
debug_visualization(heatmap, predicted_layout)
# 将预测结果保存回原始数据结构
new_yard_record['layout'] = predicted_layout

# 保存为JSON文件
import json
with open('predicted_layout.json', 'w') as f:
    json.dump(new_yard_record, f, indent=2)

print("预测结果已保存为 predicted_layout.json")