"""
pt_test.py - 比较qm9_enhanced_data_1000.pt和data_v3.pt的数据格式差异
"""

import torch
import os
import sys
from torch_geometric.data import Data
from pprint import pprint
import numpy as np

from torch_geometric.datasets import QM9
from collections import defaultdict


def create_atomref_defaultdict(target_idx=7):  # 默认为energy_U0
    # 创建一个临时QM9数据集实例以获取atomref
    dataset = QM9(root="./data/QM9")
    # 获取指定目标属性的atomref
    atomref = dataset.atomref(target_idx)

    # 创建defaultdict并添加atomref
    extra_info = defaultdict(list)
    if atomref is not None:
        extra_info['atomref'] = atomref

    return extra_info

def load_pt_file(file_path):
    """加载PT文件并返回数据"""
    print(f"尝试加载文件: {file_path}")
    try:
        data = torch.load(file_path)
        print(f"成功加载: {file_path}")
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None


def analyze_data(data, name):
    """分析数据的结构和格式"""
    print(f"\n===== {name} 数据分析 =====")

    # 检查数据类型
    print(f"数据类型: {type(data)}")

    # 检查是否为列表
    if isinstance(data, list):
        print(f"列表长度: {len(data)}")

        # 分析第一个元素
        if len(data) > 0:
            first_item = data[0]
            print(f"\n第一个元素类型: {type(first_item)}")

            # 如果是PyTorch Geometric数据对象
            if isinstance(first_item, Data):
                print("\n属性:")
                for key in first_item.keys:
                    attr = getattr(first_item, key)
                    if torch.is_tensor(attr):
                        print(f"  - {key}: Tensor类型, 形状 {attr.shape}, 数据类型 {attr.dtype}")
                    else:
                        print(f"  - {key}: {type(attr)}")

                # 检查一些常见属性
                if hasattr(first_item, 'x'):
                    print(f"\n节点特征 (x) 示例:\n{first_item.x[:3, :5]}")

                if hasattr(first_item, 'pos'):
                    print(f"\n节点位置 (pos) 示例:\n{first_item.pos[:3]}")

                if hasattr(first_item, 'edge_index'):
                    print(f"\n边索引 (edge_index) 示例:\n{first_item.edge_index[:, :5]}")
                    print(f"边数量: {first_item.edge_index.shape[1]}")

                if hasattr(first_item, 'y'):
                    print(f"\n标签 (y):\n{first_item.y}")

            # 如果是字典
            elif isinstance(first_item, dict):
                print("\n字典键:")
                for key, value in first_item.items():
                    if torch.is_tensor(value):
                        print(f"  - {key}: Tensor类型, 形状 {value.shape}, 数据类型 {value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        print(f"  - {key}: {type(value)} 长度 {len(value)}")
                    else:
                        print(f"  - {key}: {type(value)}")


def compare_data_formats(data1, data2, name1, name2):
    """比较两个数据集的格式差异"""
    print("\n===== 数据格式比较 =====")

    # 检查是否都是列表
    if isinstance(data1, list) and isinstance(data2, list):
        print(f"{name1} 长度: {len(data1)}, {name2} 长度: {len(data2)}")

        # 获取第一个元素
        if len(data1) > 0 and len(data2) > 0:
            item1 = data1[0]
            item2 = data2[0]

            # 检查元素类型
            print(f"{name1} 元素类型: {type(item1)}")
            print(f"{name2} 元素类型: {type(item2)}")

            # 比较属性
            if isinstance(item1, Data) and isinstance(item2, Data):
                keys1 = set(item1.keys)
                keys2 = set(item2.keys)

                common_keys = keys1.intersection(keys2)
                only_in_1 = keys1 - keys2
                only_in_2 = keys2 - keys1

                print(f"\n共同属性: {common_keys}")
                print(f"仅在 {name1} 中的属性: {only_in_1}")
                print(f"仅在 {name2} 中的属性: {only_in_2}")

                # 比较共同属性的形状
                print("\n共同属性的比较:")
                for key in common_keys:
                    attr1 = getattr(item1, key)
                    attr2 = getattr(item2, key)

                    if torch.is_tensor(attr1) and torch.is_tensor(attr2):
                        print(f"  - {key}: {name1} 形状 {attr1.shape}, {name2} 形状 {attr2.shape}")

                        # 形状不同时显示警告
                        if attr1.shape != attr2.shape:
                            print(f"    [警告] {key} 形状不匹配!")

                    # 检查数据类型
                    if torch.is_tensor(attr1) and torch.is_tensor(attr2):
                        if attr1.dtype != attr2.dtype:
                            print(f"    [警告] {key} 数据类型不匹配: {attr1.dtype} vs {attr2.dtype}")

            # 如果是字典，则比较键
            elif isinstance(item1, dict) and isinstance(item2, dict):
                keys1 = set(item1.keys())
                keys2 = set(item2.keys())

                common_keys = keys1.intersection(keys2)
                only_in_1 = keys1 - keys2
                only_in_2 = keys2 - keys1

                print(f"\n共同键: {common_keys}")
                print(f"仅在 {name1} 中的键: {only_in_1}")
                print(f"仅在 {name2} 中的键: {only_in_2}")


# 在pt_test.py中添加以下代码

def analyze_tuple_data(data, name):
    """分析元组类型的数据结构"""
    print(f"\n===== {name} 数据分析 =====")
    print(f"数据类型: {type(data)}")
    print(f"元组长度: {len(data)}")

    # 分析元组中的每个元素
    for i, item in enumerate(data):
        print(f"\n元素 {i} 类型: {type(item)}")

        # 如果是张量
        if torch.is_tensor(item):
            print(f"  形状: {item.shape}, 数据类型: {item.dtype}")
            if i < 3:  # 只显示前几个元素的样本值
                print(f"  样本值: {item[:3] if item.numel() > 3 else item}")

        # 如果是列表
        elif isinstance(item, list):
            print(f"  列表长度: {len(item)}")
            if len(item) > 0:
                print(f"  第一个元素类型: {type(item[0])}")

                # 如果列表中的元素是Data对象
                if isinstance(item[0], Data):
                    sample = item[0]
                    print("\n  第一个元素的属性:")
                    for key in sample.keys:
                        attr = getattr(sample, key)
                        if torch.is_tensor(attr):
                            print(f"    - {key}: 形状 {attr.shape}, 类型 {attr.dtype}")
                        else:
                            print(f"    - {key}: {type(attr)}")


def improved_compare_data_formats(data1, data2, name1, name2):
    """比较不同类型数据集的格式差异"""
    print("\n===== 数据格式比较 =====")
    print(f"{name1} 类型: {type(data1)}, {name2} 类型: {type(data2)}")

    # 处理列表类型的数据集
    if isinstance(data1, list) and len(data1) > 0:
        print(f"\n{name1} 第一个元素类型: {type(data1[0])}")
        if isinstance(data1[0], dict):
            print(f"{name1} 字典键: {list(data1[0].keys())}")
            # 显示各个键对应的数组形状
            for key, value in data1[0].items():
                if isinstance(value, np.ndarray):
                    print(f"  - {key}: 形状 {value.shape}, 类型 {value.dtype}")

    # 处理元组类型的数据集
    if isinstance(data2, tuple):
        print(f"\n{name2} 元组长度: {len(data2)}")
        for i, item in enumerate(data2):
            print(f"元素 {i} 类型: {type(item)}")
            if isinstance(item, list) and len(item) > 0:
                print(f"  列表长度: {len(item)}")
                if len(item) > 0:
                    print(f"  第一个元素类型: {type(item[0])}")

                    # 如果是PyG数据对象，检查其属性
                    if isinstance(item[0], Data):
                        print(f"  属性: {item[0].keys}")

    # 尝试找到共同的数据结构进行比较
    # 这部分需要根据实际数据结构来调整
    print("\n尝试进行数据结构映射和比较...")


def detailed_tuple_analysis(data, name):
    """对元组类型数据进行更详细的分析"""
    print(f"\n===== {name} 详细分析 =====")

    # 分析第一个元素 (Data对象)
    if len(data) > 0 and isinstance(data[0], Data):
        data_obj = data[0]
        print("\n[元素0] Data对象详细信息:")

        # 修正：调用keys方法而不是作为属性访问
        print(f"  Data对象概览: {data_obj}")

        # 直接访问常见属性而不依赖于keys()方法
        common_attrs = ['x', 'edge_index', 'edge_attr', 'y', 'pos', 'z', 'smiles', 'name', 'idx']
        print("\n  属性详细信息:")

        for key in common_attrs:
            if hasattr(data_obj, key):
                attr = getattr(data_obj, key)
                if torch.is_tensor(attr):
                    print(f"  - {key}: Tensor类型, 形状 {attr.shape}, 数据类型 {attr.dtype}")
                    # 对于非常大的张量，只显示尺寸信息
                    if attr.numel() > 0 and attr.numel() < 10:
                        print(f"    样本数据: {attr}")
                elif isinstance(attr, list):
                    print(f"  - {key}: List类型, 长度 {len(attr)}")
                    if len(attr) > 0:
                        print(f"    第一个元素类型: {type(attr[0])}")
                else:
                    print(f"  - {key}: {type(attr)}")

        # 显示图的基本统计信息
        if hasattr(data_obj, 'x') and hasattr(data_obj, 'edge_index'):
            num_nodes = data_obj.x.shape[0]
            num_edges = data_obj.edge_index.shape[1]
            print(f"\n  图统计: {num_nodes}个节点, {num_edges}条边")
            print(f"  平均度数: {2 * num_edges / num_nodes:.2f}")

            # 估计这是多少个分子
            if hasattr(data_obj, 'y'):
                num_graphs = data_obj.y.shape[0]
                print(f"  估计包含 {num_graphs} 个分子")
                print(f"  每个分子平均有 {num_nodes / num_graphs:.1f} 个节点")

    # 分析第二个元素 (defaultdict)
    if len(data) > 1 and isinstance(data[1], dict):
        dict_obj = data[1]
        print("\n[元素1] defaultdict详细信息:")
        print(f"  键数量: {len(dict_obj)}")
        print(f"  键列表: {list(dict_obj.keys())}")

        # 显示每个键的值类型和基本信息
        for key, value in dict_obj.items():
            if torch.is_tensor(value):
                print(f"  - {key}: Tensor类型, 形状 {value.shape}, 数据类型 {value.dtype}")
                if value.numel() > 0 and value.numel() < 10:
                    print(f"    样本数据: {value}")
            elif isinstance(value, (list, tuple)):
                print(f"  - {key}: {type(value)}类型, 长度 {len(value)}")
                if len(value) > 0:
                    print(f"    第一个元素类型: {type(value[0])}")
            else:
                print(f"  - {key}: {type(value)}")


def convert_enhanced_to_data_batch(enhanced_data_list):
    """将增强数据列表转换为PyG Data批次对象"""
    from torch_geometric.data import Batch

    # 首先将每个字典转换为Data对象
    data_objects = []
    for enhanced_dict in enhanced_data_list:
        # 将NumPy数组转换为PyTorch张量
        x = torch.tensor(enhanced_dict['x'], dtype=torch.float)
        edge_index = torch.tensor(enhanced_dict['edge_index'], dtype=torch.long)
        pos = torch.tensor(enhanced_dict['pos'], dtype=torch.float)
        z = torch.tensor(enhanced_dict['z'], dtype=torch.long)

        # 创建数据对象
        data = Data(x=x, edge_index=edge_index, pos=pos, z=z)

        # 添加边特征
        if 'edge_attr' in enhanced_dict and enhanced_dict['edge_attr'] is not None:
            data.edge_attr = torch.tensor(enhanced_dict['edge_attr'], dtype=torch.float)

        # 添加标签
        if 'y' in enhanced_dict and enhanced_dict['y'] is not None:
            data.y = torch.tensor(enhanced_dict['y'], dtype=torch.float)

        # 添加其他属性
        if 'smiles' in enhanced_dict:
            data.smiles = enhanced_dict['smiles']

        if 'name' in enhanced_dict:
            data.name = enhanced_dict['name']

        if 'idx' in enhanced_dict:
            data.idx = enhanced_dict['idx']

        data_objects.append(data)

    # 创建一个批次
    print(f"创建批次: {len(data_objects)}个分子")
    return Batch.from_data_list(data_objects)


def compare_batch_with_data_v3(batch_data, data_v3_tuple):
    """比较批次数据和data_v3的结构和内容"""
    print("\n===== 批次转换后比较 =====")

    # 获取data_v3中的Data对象
    data_v3_obj = data_v3_tuple[0]

    print(f"批次数据: {batch_data}")
    print(f"data_v3: {data_v3_obj}")

    # 比较基本属性
    batch_attrs = {attr for attr in dir(batch_data) if
                   not attr.startswith('_') and not callable(getattr(batch_data, attr))}
    data_v3_attrs = {attr for attr in dir(data_v3_obj) if
                     not attr.startswith('_') and not callable(getattr(data_v3_obj, attr))}

    common_attrs = batch_attrs.intersection(data_v3_attrs)
    print(f"\n共同属性: {common_attrs}")

    # 比较关键张量的形状
    for attr in ['x', 'edge_index', 'edge_attr', 'y', 'pos', 'z']:
        if hasattr(batch_data, attr) and hasattr(data_v3_obj, attr):
            batch_shape = getattr(batch_data, attr).shape
            data_v3_shape = getattr(data_v3_obj, attr).shape
            print(f"{attr} 形状比较: 批次 {batch_shape}, data_v3 {data_v3_shape}")

    # 输出结构差异总结
    print("\n=== 结构差异总结 ===")
    print("1. data_v3.pt是一个元组，包含一个大型Data对象和一个defaultdict")
    print("2. 大型Data对象似乎是多个分子的连接表示，而不是单独的分子集合")
    print("3. qm9_enhanced_data_1000.pt是一个字典列表，每个字典表示一个分子")
    print("4. 将字典列表转换为Data对象批次后，结构上更接近data_v3.pt的形式")

    # 提供使用建议
    print("\n=== 使用建议 ===")
    print("要用qm9_enhanced_data_1000.pt替代data_v3.pt，需要考虑：")
    print("1. data_v3.pt可能包含预先计算的批处理或连接形式的数据")
    print("2. 在训练时，需要将字典列表转换为PyG Data对象")
    print("3. 可能需要调整批处理逻辑，以适应不同的数据结构")
    print("4. 两者的特征维度可能不同，需要确保模型能处理变化后的维度")


def convert_enhanced_to_data_v3_format(enhanced_data_list):
    """将增强数据列表转化为data_v3.pt格式（元组形式）"""
    from collections import defaultdict
    from torch_geometric.data import Data
    import torch

    print("开始转换数据格式...")
    print(f"处理 {len(enhanced_data_list)} 个分子数据")

    # 第一步：计算数据总大小和起始索引
    total_nodes = 0
    total_edges = 0
    total_molecules = len(enhanced_data_list)

    # 获取第一个样本以确定特征维度
    first_sample = enhanced_data_list[0]
    x_dim = first_sample['x'].shape[1]
    edge_attr_dim = first_sample['edge_attr'].shape[1]
    y_dim = first_sample['y'].shape[1]

    # 计算所有分子的节点数和边数
    for molecule in enhanced_data_list:
        total_nodes += molecule['x'].shape[0]
        total_edges += molecule['edge_index'].shape[1]

    print(f"总分子数: {total_molecules}, 总节点数: {total_nodes}, 总边数: {total_edges}")

    # 第二步：创建存储所有分子数据的大张量
    x = torch.zeros((total_nodes, x_dim), dtype=torch.float32)
    edge_index = torch.zeros((2, total_edges), dtype=torch.int64)
    edge_attr = torch.zeros((total_edges, edge_attr_dim), dtype=torch.float32)
    y = torch.zeros((total_molecules, y_dim), dtype=torch.float32)
    pos = torch.zeros((total_nodes, 3), dtype=torch.float32)
    z = torch.zeros(total_nodes, dtype=torch.int64)

    smiles_list = []
    name_list = []
    idx_list = []

    # 第三步：填充数据并记录索引
    node_offset = 0
    edge_offset = 0
    mol_offset = 0

    # 创建索引张量来记录每个属性的起始位置
    x_indices = torch.zeros(total_molecules + 1, dtype=torch.int64)
    edge_indices = torch.zeros(total_molecules + 1, dtype=torch.int64)
    pos_indices = torch.zeros(total_molecules + 1, dtype=torch.int64)
    z_indices = torch.zeros(total_molecules + 1, dtype=torch.int64)
    y_indices = torch.arange(total_molecules + 1, dtype=torch.int64)
    smiles_indices = torch.arange(total_molecules + 1, dtype=torch.int64)
    name_indices = torch.arange(total_molecules + 1, dtype=torch.int64)
    idx_indices = torch.arange(total_molecules + 1, dtype=torch.int64)

    for i, molecule in enumerate(enhanced_data_list):
        num_nodes = molecule['x'].shape[0]
        num_edges = molecule['edge_index'].shape[1]

        # 填充数据
        x[node_offset:node_offset + num_nodes] = torch.tensor(molecule['x'], dtype=torch.float32)

        # 边索引需要调整以反映节点的全局索引
        molecule_edge_index = torch.tensor(molecule['edge_index'], dtype=torch.int64)
        molecule_edge_index += node_offset  # 将局部节点索引转换为全局索引
        edge_index[:, edge_offset:edge_offset + num_edges] = molecule_edge_index

        edge_attr[edge_offset:edge_offset + num_edges] = torch.tensor(molecule['edge_attr'], dtype=torch.float32)
        y[mol_offset] = torch.tensor(molecule['y'], dtype=torch.float32)
        pos[node_offset:node_offset + num_nodes] = torch.tensor(molecule['pos'], dtype=torch.float32)
        z[node_offset:node_offset + num_nodes] = torch.tensor(molecule['z'], dtype=torch.int64)

        smiles_list.append(molecule['smiles'])
        name_list.append(molecule['name'])
        idx_list.append(molecule['idx'])

        # 记录索引
        x_indices[i + 1] = x_indices[i] + num_nodes
        edge_indices[i + 1] = edge_indices[i] + num_edges
        pos_indices[i + 1] = pos_indices[i] + num_nodes
        z_indices[i + 1] = z_indices[i] + num_nodes

        # 更新偏移量
        node_offset += num_nodes
        edge_offset += num_edges
        mol_offset += 1

    # 第四步：创建单个Data对象
    data_obj = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos,
        z=z,
        smiles=smiles_list,
        name=name_list,
        idx=idx_list
    )

    # 第五步：创建defaultdict并填充与原始data_v3.pt相同的结构
    slices = defaultdict(list)
    slices['x'] = x_indices
    slices['edge_index'] = edge_indices
    slices['edge_attr'] = edge_indices  # 边属性与边索引使用相同的切片
    slices['y'] = y_indices
    slices['pos'] = pos_indices
    slices['z'] = z_indices
    slices['smiles'] = smiles_indices
    slices['name'] = name_indices
    slices['idx'] = idx_indices

    # 返回与data_v3.pt相同格式的元组
    return (data_obj, slices)


def analyze_defaultdict(data_v3_tuple):
    """分析data_v3.pt中的defaultdict（元组的第二个元素）"""
    print("\n===== defaultdict (元组元素1) 详细信息 =====")

    if len(data_v3_tuple) > 1 and isinstance(data_v3_tuple[1], dict):
        defaultdict_obj = data_v3_tuple[1]
        print(f"defaultdict类型: {type(defaultdict_obj)}")
        print(f"键数量: {len(defaultdict_obj)}")

        if len(defaultdict_obj) > 0:
            print(f"键列表: {list(defaultdict_obj.keys())}")

            # 显示每个键的值类型和基本信息
            for key, value in defaultdict_obj.items():
                if torch.is_tensor(value):
                    print(f"  - {key}: Tensor类型, 形状 {value.shape}, 数据类型 {value.dtype}")
                    print(f"    值: {value}")
                elif isinstance(value, (list, tuple)):
                    print(f"  - {key}: {type(value)}类型, 长度 {len(value)}")
                    if len(value) > 0:
                        print(f"    第一个元素: {value[0]}")
                else:
                    print(f"  - {key}: {type(value)}")
                    print(f"    值: {value}")
        else:
            print("defaultdict为空")
    else:
        print("元组中不存在第二个元素或第二个元素不是字典类型")

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 2:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        # 默认文件路径
        file1 = "./data/QM9_enhanced/processed/qm9_enhanced_test.pt"
        file2 = "./data/QM9_enhanced/processed/data_v3_mark.pt"

    # 加载数据
    data1 = load_pt_file(file1)
    data2 = load_pt_file(file2)

    # 分析数据
    if data1 is not None:
        analyze_data(data1, "qm9_enhanced_data_1000.pt")

    if data2 is not None:
        # 使用专门处理元组的函数
        analyze_tuple_data(data2, "data_v3.pt")
        # 添加详细分析
        detailed_tuple_analysis(data2, "data_v3.pt")

    if data1 is not None and data2 is not None:
        # 使用改进的比较函数
        improved_compare_data_formats(data1, data2, "qm9_enhanced_data_1000.pt", "data_v3.pt")

        # 转换增强数据为批次并比较
        try:
            batch_data = convert_enhanced_to_data_batch(data1[:10])  # 只使用前10个样本以节省资源
            compare_batch_with_data_v3(batch_data, data2)
        except Exception as e:
            print(f"\n批次转换出错: {e}")
            import traceback
            traceback.print_exc()

    if data1 is not None:
        # 转换为data_v3格式
        converted_data = convert_enhanced_to_data_v3_format(data1)

        # 保存转换后的数据
        output_path = "./data/QM9_enhanced/processed/qm9_enhanced_data_1000_v3_format.pt"
        torch.save(converted_data, output_path)
        print(f"转换后的数据已保存至: {output_path}")

        # 验证转换后的数据
        print("\n==== 转换后数据验证 ====")
        print(f"数据类型: {type(converted_data)}")
        print(f"元组长度: {len(converted_data)}")

        # 验证第一个元素（Data对象）
        data_obj = converted_data[0]
        print(f"\n[元素0] Data对象信息:")
        print(f"  - x: 形状 {data_obj.x.shape}, 类型 {data_obj.x.dtype}")
        print(f"  - edge_index: 形状 {data_obj.edge_index.shape}, 类型 {data_obj.edge_index.dtype}")
        if hasattr(data_obj, 'edge_attr') and data_obj.edge_attr is not None:
            print(f"  - edge_attr: 形状 {data_obj.edge_attr.shape}, 类型 {data_obj.edge_attr.dtype}")
        print(f"  - pos: 形状 {data_obj.pos.shape}, 类型 {data_obj.pos.dtype}")
        print(f"  - z: 形状 {data_obj.z.shape}, 类型 {data_obj.z.dtype}")
        if hasattr(data_obj, 'y') and data_obj.y is not None:
            print(f"  - y: 形状 {data_obj.y.shape}, 类型 {data_obj.y.dtype}")

        # 在main函数中调用
        if data2 is not None:
            # 分析原始data_v3.pt中的defaultdict
            analyze_defaultdict(data2)

        # 在转换和保存后调用
        if converted_data is not None:
            # 分析转换后的defaultdict
            analyze_defaultdict(converted_data)


if __name__ == "__main__":
    main()