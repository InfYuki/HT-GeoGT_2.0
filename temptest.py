'''
====================================================
    Dataset test
====================================================
'''
import random
import numpy as np
import torch
import pytorch_lightning as pl
from argparse import Namespace
from ht_geogt.data import DataModule

# 设置随机种子确保结果可复现
pl.seed_everything(42)

# 配置数据集参数
args = Namespace(
    dataset="QM9Edge",
    dataset_arg="energy_U0",
    dataset_root="./data/QM9",
    max_nodes=None,
    mean=None,
    std=None,
    batch_size=32,
    inference_batch_size=32,
    standardize=False,
    splits=None,
    split_mode=None,
    train_size=2000,
    val_size=200,
    test_size=200,
    num_workers=0,
    reload=0,
    seed=42,
    log_dir="./logs",
    prior_model="Atomref"
)

print("=== QM9数据集基本信息 ===")

# 初始化数据模块
data_module = DataModule(args)
data_module.prepare_dataset()

# 打印数据集大小
print(f"数据集总大小: {len(data_module.dataset)}")
print(f"训练集大小: {len(data_module.train_dataset)}")
print(f"验证集大小: {len(data_module.val_dataset)}")
print(f"测试集大小: {len(data_module.test_dataset)}")

# 获取第一个分子图
first_mol = data_module.dataset[0]

print("\n=== 第一个分子图的详细信息 ===")
print(f"数据类型: {type(first_mol)}")
print(f"可用属性: {first_mol.keys}")

# 打印原子信息
if hasattr(first_mol, 'z'):
    print(f"\n原子类型 (z): {first_mol.z}")
    print(f"原子数量: {len(first_mol.z)}")

# 打印坐标信息
if hasattr(first_mol, 'pos'):
    print(f"\n原子坐标 (pos): 形状 {first_mol.pos.shape}")
    print(first_mol.pos)

# 打印节点特征
if hasattr(first_mol, 'x'):
    print(f"\n节点特征 (x): 形状 {first_mol.x.shape}")
    print(first_mol.x)

# 打印边信息
if hasattr(first_mol, 'edge_index'):
    print(f"\n边连接 (edge_index): 形状 {first_mol.edge_index.shape}")
    print(first_mol.edge_index)
    print(f"边数量: {first_mol.edge_index.shape[1]}")

if hasattr(first_mol, 'edge_attr'):
    print(f"\n边特征 (edge_attr): 形状 {first_mol.edge_attr.shape}")
    print(first_mol.edge_attr[:5])  # 只打印前5个边的特征

# 打印标签信息
if hasattr(first_mol, 'y'):
    print(f"\n标签 (y): {first_mol.y}")

# 获取批次数据并打印
print("\n=== 批次数据信息 ===")
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

print(f"批次类型: {type(batch)}")
if isinstance(batch, dict):
    print(f"批次键: {list(batch.keys())}")

    # 打印批次中每个键的形状
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"'{key}' 形状: {value.shape}")

    # 打印第一个图的信息
    if 'z' in batch:
        print(f"\n第一个图的原子类型: {batch['z'][0]}")
    if 'pos' in batch:
        print(f"\n第一个图的原子坐标 (前3个): \n{batch['pos'][0][:3]}")
    if 'x' in batch:
        print(f"\n第一个图的节点特征 (前3个): \n{batch['x'][0][:3]}")

'''
====================================================
    PathFinder 单分子测试 - 使用原始边信息
====================================================
'''
from ht_geogt.modules.PathFinder import FindCenterPath_Node

print("\n=== PathFinder单分子测试 - 使用原始边信息 ===")

# 确保GPU可用并设置随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 设置PathFinder参数
max_num_path = 6  # 最大路径数
max_num_hop = 4  # 最大跳数
k = 2  # 最小度数阈值

# 使用第一个分子 (而非批次数据)
sample_mol = data_module.dataset[5]
print(f"分子名称: {sample_mol.name}, SMILES: {sample_mol.smiles}")

# 将数据移到设备上
z = sample_mol.z.to(device)
pos = sample_mol.pos.to(device)
x = sample_mol.x.to(device) if hasattr(sample_mol, 'x') else None
edge_index = sample_mol.edge_index.to(device)  # 直接使用原始边信息

print(f"原子数量: {len(z)}")
print(f"原子类型: {z.tolist()}")
print(f"原始边数量: {edge_index.size(1)}")
print(f"原始边索引: \n{edge_index.t()}")

# 将分子数据转换为图字典格式
graph = {
    'edge_index': edge_index,
    'pos': {i: pos[i].cpu().numpy() for i in range(len(z))},
    'attr': {i: x[i] for i in range(len(z))}
}

# 存储该分子所有节点的centerpath结果
mol_centerpaths = {}

# 为每个节点计算centerpath
for node_idx in range(len(z)):
    try:
        result = FindCenterPath_Node(max_num_path, max_num_hop, node_idx, graph, k)
        if result['centerpathlist']:
            mol_centerpaths[node_idx] = result
            print(f"\n节点 {node_idx} 找到 {len(result['centerpathlist'])} 条centerpath")

            # 打印该节点的centerpath详细信息
            print(f"节点 {node_idx} 的centerpath示例:")
            for path_idx, path in enumerate(result['centerpathlist']):
                # 过滤掉-1填充
                actual_path = [p for p in path if p != -1]
                print(f"  路径 {path_idx + 1}:")
                print(f"    节点序列: {actual_path}")

                # 打印路径上的原子类型
                path_atom_types = [z[p].item() for p in actual_path]
                print(f"    原子类型: {path_atom_types}")
                print(f"    实际路径长度: {len(actual_path)}")
    except Exception as e:
        print(f"节点 {node_idx} 生成centerpath出错: {str(e)}")


'''
====================================================
    encode_tokens单分子测试 - 保留原始路径
====================================================
'''
print("\n=== encode_tokens单分子测试 - 保留原始路径 ===")
from ht_geogt.modules.encode_tokens import TokenEncoder, CenterPathEncoder_one

if mol_centerpaths:
    token_dim = 32
    inv_dim = 12
    equ_dim = 12
    feature_dim = x.size(1)

    # 选择第一个有centerpath的节点
    node_idx = list(mol_centerpaths.keys())[0]
    result = mol_centerpaths[node_idx]

    print(f"选择节点 {node_idx} 进行编码测试，该节点有 {len(result['centerpathlist'])} 条centerpath")

    # 初始化TokenEncoder
    token_encoder = TokenEncoder(feature_dim, token_dim, inv_dim, equ_dim).to(device)

    # 为每条centerpath生成tokens
    all_inv_tokens = []
    all_equ_tokens = []

    for path_idx, path in enumerate(result['centerpathlist']):
        # 使用原始路径，不过滤-1
        print(f"\n编码路径 {path_idx + 1}...")
        print(f"  路径节点序列: {path}")

        # 构建centerpath_one输入格式
        centerpath_one = (node_idx, path)

        try:
            # 调用CenterPathEncoder_one进行编码
            inv_tokens, equ_tokens = CenterPathEncoder_one(
                graph, centerpath_one, max_num_hop, token_dim, token_encoder, inv_dim, equ_dim
            )

            all_inv_tokens.append(inv_tokens)
            all_equ_tokens.append(equ_tokens)

            print(f"  编码完成:")
            print(f"  不变特征tokens数量: {len(inv_tokens)}")
            print(f"  等变特征tokens数量: {len(equ_tokens)}")

            # 计算路径上的三元组数量
            center_idx = len(path) // 2
            # 计算可能的跳数 (不包括中心节点自身)
            max_possible_hops = min(center_idx, len(path) - center_idx - 1)
            expected_tokens = max_possible_hops + 1  # 加1是因为中心节点自身也会产生一个token
            print(f"  预期token数量 (基于三元组): {expected_tokens}")

            # 打印每个token的形状
            for i, (inv_t, equ_t) in enumerate(zip(inv_tokens, equ_tokens)):
                print(f"  Token {i} - 不变特征形状: {inv_t.shape}, 等变特征形状: {equ_t.shape}")

                # 如果不是第一个token，打印生成此token的三元组节点
                if i > 0 and i <= max_possible_hops:
                    left_node = path[center_idx - i]
                    right_node = path[center_idx + i]
                    print(f"    对应三元组: [{left_node}, {node_idx}, {right_node}]")
                    # 打印三元组中节点的原子类型
                    if left_node != -1 and right_node != -1:
                        atom_types = [z[left_node].item(), z[node_idx].item(), z[right_node].item()]
                        print(f"    三元组原子类型: {atom_types}")

            # 打印第一个token的值
            if inv_tokens:
                print(f"\n  第一个token的不变特征前5个值: {inv_tokens[0][:5]}")
                print(f"  第一个token的等变特征前5个值: {equ_tokens[0][:5]}")
                print(f"\n  第二个token的不变特征前5个值: {inv_tokens[1][:5]}")
                print(f"  第二个token的等变特征前5个值: {equ_tokens[1][:5]}")

        except Exception as e:
            print(f"  编码路径出错: {str(e)}")
            import traceback

            traceback.print_exc()

    # 统计编码结果
    print(f"\n共成功编码 {len(all_inv_tokens)} 条路径")
else:
    print("没有找到合适的中心路径进行编码测试")

print("\n=== 测试完成 ===")

'''
====================================================
    tokenlevel_attn和centerlevel_attn测试
====================================================
'''
print("\n=== tokenlevel_attn和centerlevel_attn测试 ===")
import torch.nn as nn
from ht_geogt.modules.tokenlevel_attn import TokenLevelAttn
from ht_geogt.modules.centerpathlevel_attn import CenterPathLevelAttn

if mol_centerpaths:
    # 设置参数
    token_dim = 32
    inv_dim = 12
    equ_dim = 12
    cptoken_dim = 64  # centerpath token维度
    node_dim = 128  # 节点嵌入维度
    feature_dim = x.size(1)

    # 初始化模型
    token_encoder = TokenEncoder(feature_dim, token_dim, inv_dim, equ_dim).to(device)
    token_level_attn = TokenLevelAttn(token_dim, cptoken_dim, num_heads=4, num_layers=3, dropout=0.1).to(device)
    cp_level_attn = CenterPathLevelAttn(cptoken_dim, node_dim, num_heads=4, num_layers=2, max_num_path=max_num_path,
                                        dropout=0.1).to(device)

    print(f"模型初始化完成:")
    print(f"  TokenEncoder: 输入维度={feature_dim}, token维度={token_dim}")
    print(f"  TokenLevelAttn: 输入维度={token_dim}, 输出维度={cptoken_dim}")
    print(f"  CenterPathLevelAttn: 输入维度={cptoken_dim}, 输出维度={node_dim}")

    # 处理每个中心节点
    node_embeddings = {}

    for node_idx, result in mol_centerpaths.items():
        print(f"\n处理节点 {node_idx}:")

        # 1. 为每条centerpath生成tokens
        path_inv_tokens = []
        path_equ_tokens = []

        for path_idx, path in enumerate(result['centerpathlist']):
            print(f"  编码路径 {path_idx + 1}...")
            centerpath_one = (node_idx, path)

            try:
                # 生成tokens
                inv_tokens, equ_tokens = CenterPathEncoder_one(
                    graph, centerpath_one, max_num_hop, token_dim, token_encoder, inv_dim, equ_dim
                )

                path_inv_tokens.append(inv_tokens)
                path_equ_tokens.append(equ_tokens)

                print(f"    生成了 {len(inv_tokens)} 个tokens")

            except Exception as e:
                print(f"    编码路径出错: {str(e)}")
                continue

        # 2. 使用TokenLevelAttn处理每条路径的tokens
        inv_cptokens = []
        equ_cptokens = []

        print("\n  应用TokenLevelAttn...")
        for path_idx, (inv_tokens, equ_tokens) in enumerate(zip(path_inv_tokens, path_equ_tokens)):
            try:
                # 准备tokens为批处理格式
                inv_tokens_batch = torch.stack(inv_tokens).unsqueeze(0)  # [1, num_tokens, token_dim]
                equ_tokens_batch = torch.stack(equ_tokens).unsqueeze(0)  # [1, num_tokens, token_dim]

                # 应用TokenLevelAttn
                inv_cptoken = token_level_attn(inv_tokens_batch)  # [1, cptoken_dim]
                equ_cptoken = token_level_attn(equ_tokens_batch)  # [1, cptoken_dim]

                inv_cptokens.append(inv_cptoken.squeeze(0))
                equ_cptokens.append(equ_cptoken.squeeze(0))

                print(f"    路径 {path_idx + 1}: tokens聚合为cptoken, 形状: {inv_cptoken.shape}")

            except Exception as e:
                print(f"    处理路径 {path_idx + 1} tokens出错: {str(e)}")
                import traceback

                traceback.print_exc()

        # 3. 使用CenterPathLevelAttn聚合所有路径的cptokens
        if inv_cptokens:
            print("\n  应用CenterPathLevelAttn...")
            try:
                # 组合所有路径的cptokens
                num_paths = len(inv_cptokens)

                # 如果路径数少于max_num_path，需要填充
                if num_paths < max_num_path:
                    # 创建填充cptoken
                    pad_inv = torch.zeros((max_num_path - num_paths, cptoken_dim), device=device)
                    pad_equ = torch.zeros((max_num_path - num_paths, cptoken_dim), device=device)

                    # 合并实际cptokens和填充
                    inv_cptokens_tensor = torch.cat([torch.stack(inv_cptokens), pad_inv], dim=0)
                    equ_cptokens_tensor = torch.cat([torch.stack(equ_cptokens), pad_equ], dim=0)
                else:
                    # 如果路径数超过max_num_path，只取前max_num_path个
                    inv_cptokens_tensor = torch.stack(inv_cptokens[:max_num_path])
                    equ_cptokens_tensor = torch.stack(equ_cptokens[:max_num_path])

                # 批量处理多个中心节点时需要添加批次维度
                inv_cptokens_batch = inv_cptokens_tensor.unsqueeze(0)  # [1, max_num_path, cptoken_dim]
                equ_cptokens_batch = equ_cptokens_tensor.unsqueeze(0)  # [1, max_num_path, cptoken_dim]

                # 创建路径数量掩码
                num_valid_paths = torch.tensor([min(num_paths, max_num_path)], device=device)

                # 应用CenterPathLevelAttn
                with torch.no_grad():  # 测试模式
                    inv_node_embedding = cp_level_attn(inv_cptokens_batch, num_valid_paths)  # [1, node_dim]
                    equ_node_embedding = cp_level_attn(equ_cptokens_batch, num_valid_paths)  # [1, node_dim]

                # 合并不变特征和等变特征
                node_embedding = torch.cat([inv_node_embedding, equ_node_embedding], dim=1)  # [1, 2*node_dim]
                node_embeddings[node_idx] = node_embedding.squeeze(0)  # [2*node_dim]

                print(f"    生成节点嵌入，形状: {node_embedding.shape}")
                print(f"    不变特征嵌入前5个值: {inv_node_embedding[0, :5]}")
                print(f"    等变特征嵌入前5个值: {equ_node_embedding[0, :5]}")

            except Exception as e:
                print(f"    生成节点嵌入出错: {str(e)}")
                import traceback

                traceback.print_exc()

    # 4. 更新原始节点特征
    if node_embeddings:
        print("\n更新节点特征:")

        # 定义线性层将node_embedding映射回原始特征空间
        output_dim = x.size(1)  # 原始特征维度
        input_dim = 2 * node_dim  # 合并后的node_embedding维度

        # 创建映射层
        mapping_layer = nn.Linear(input_dim, output_dim).to(device)

        # 更新每个有centerpath的节点特征
        updated_nodes = {}
        for node_idx, embedding in node_embeddings.items():
            # 通过映射层将嵌入转换为原始特征尺寸
            updated_feature = mapping_layer(embedding)

            # 获取原始特征
            original_feature = x[node_idx]

            # 残差连接: 原始特征 + 更新特征
            final_feature = original_feature + updated_feature
            updated_nodes[node_idx] = final_feature

            print(f"  节点 {node_idx}:")
            print(f"    原始特征前5个值: {original_feature[:5]}")
            print(f"    更新特征前5个值: {updated_feature[:5]}")
            print(f"    最终特征前5个值: {final_feature[:5]}")

    print(f"\n共更新了 {len(node_embeddings)} 个节点的特征")

else:
    print("没有找到合适的中心路径进行测试")

print("\n=== tokenlevel_attn和centerlevel_attn测试完成 ===")




