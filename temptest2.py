
''''''
# 临时验证脚本 - 可以在Python控制台中运行
import torch
import os

# 检查文件是否存在
file_path = './data/QM9_enhanced/processed/data_v3.pt'
print(f"文件是否存在: {os.path.exists(file_path)}")

if os.path.exists(file_path):
    # 加载数据查看格式
    try:
        data, slices = torch.load(file_path)
        print(f"数据类型: data={type(data)}, slices={type(slices)}")

        # 如果有x属性，检查其形状
        if hasattr(data, 'x'):
            print(f"x形状: {data.x.shape}")
            print(f"x的前几个值: {data.x[:5] if data.x.numel() >= 5 else data.x}")

        # 检查slices中是否有x
        if isinstance(slices, dict) and 'x' in slices:
            x_slices = slices['x']
            print(f"x_slices: {x_slices[:5]}")  # 前几个切片索引

            # 计算第一个样本的x
            if len(x_slices) > 1:
                first_x = data.x[x_slices[0]:x_slices[1]]
                print(f"第一个样本的x形状: {first_x.shape}")

    except Exception as e:
        print(f"加载数据时出错: {e}")
        # 尝试其他加载方式
        try:
            raw_data = torch.load(file_path)
            print(f"原始数据类型: {type(raw_data)}")
            if isinstance(raw_data, (list, tuple)):
                print(f"数据长度: {len(raw_data)}")
                if len(raw_data) > 0:
                    print(f"第一个元素类型: {type(raw_data[0])}")
        except Exception as e2:
            print(f"二次加载也失败: {e2}")


'''
# 查看qm9_target_dict内容的脚本
try:
    from torch_geometric.nn.models.schnet import qm9_target_dict

    print("=== QM9 Target Dictionary ===")
    print("键(索引) -> 值(属性名):")
    for idx, name in qm9_target_dict.items():
        print(f"  {idx}: '{name}'")

    print(f"\n总共有 {len(qm9_target_dict)} 个目标属性")
    print(f"\n所有可用的属性名: {list(qm9_target_dict.values())}")

except ImportError as e:
    print(f"无法导入qm9_target_dict: {e}")
    print("可能的原因：")
    print("1. torch_geometric版本不兼容")
    print("2. 模块结构发生变化")

    # 尝试其他可能的导入路径
    try:
        from torch_geometric.datasets import QM9

        print("\n尝试通过QM9数据集获取目标信息...")

        # 创建一个临时QM9实例来查看可用属性
        temp_dataset = QM9(root='./temp_qm9_check')
        if hasattr(temp_dataset, 'available_properties'):
            print(f"可用属性: {temp_dataset.available_properties}")

    except Exception as e2:
        print(f"备选方案也失败: {e2}")
'''

