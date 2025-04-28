import pandas as pd
import json

# Excel 文件路径
file_path = "R3布局.xlsx"

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
            "length": 270,  # 假设堆场长度为 270 米
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
                "length": int(group[(group['brand'] == row['brand']) & (group['model'] == row['model'])]['Height'].iloc[0]),
                "width": int(group[(group['brand'] == row['brand']) & (group['model'] == row['model'])]['Width'].iloc[0]),
                "count": int(row['count'])  # 转换为 Python int
            }
            yard_info['vehicles'].append(vehicle_info)

        # 提取布局信息
        for _, row in group.iterrows():
            layout_info = {
                "x": float(row['X']),  # 转换为 Python float
                "y": float(row['Y']),  # 转换为 Python float
                "rotated": bool(row['Rotated']),  # 转换为 Python bool
                "brand": row['brand'],
                "model": row['model']
            }
            yard_info['layout'].append(layout_info)

        raw_data.append(yard_info)

# 打印生成的 raw_data
print(json.dumps(raw_data, indent=4))