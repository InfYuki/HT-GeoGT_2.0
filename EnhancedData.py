'''
EnhancedData.py - 分子图数据增强模块
功能: 通过中心路径信息增强分子图节点特征
'''

import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import copy
from torch_geometric.data import Data, Dataset
from ht_geogt.modules.PathFinder import FindCenterPath_Node
from ht_geogt.modules.encode_tokens import TokenEncoder, CenterPathEncoder_one
from ht_geogt.modules.tokenlevel_attn import TokenLevelAttn
from ht_geogt.modules.centerpathlevel_attn import CenterPathLevelAttn



class EnhancedDataProcessor:
    def __init__(self, max_num_path=6, max_num_hop=4, k=2,
                 token_dim=32, inv_dim=12, equ_dim=12, cptoken_dim=64, node_dim=128,
                 device=None):
        """初始化分子图增强处理器

        参数:
            max_num_path (int): 最大路径数
            max_num_hop (int): 最大跳数
            k (int): 节点最小度数阈值
            token_dim (int): token向量维度
            inv_dim (int): 不变特征维度
            equ_dim (int): 等变特征维度
            cptoken_dim (int): centerpath token维度
            node_dim (int): 节点嵌入维度
            device: 计算设备(CPU/GPU)
        """
        """初始化分子图增强处理器 - GPU优化版本"""
        self.max_num_path = max_num_path
        self.max_num_hop = max_num_hop
        self.k = k
        self.token_dim = token_dim
        self.inv_dim = inv_dim
        self.equ_dim = equ_dim
        self.cptoken_dim = cptoken_dim
        self.node_dim = node_dim

        # 使用指定设备或自动选择
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"EnhancedDataProcessor初始化完成，使用设备: {self.device}")
        print(f"参数配置: max_num_path={max_num_path}, max_num_hop={max_num_hop}, k={k}")
        print(f"特征维度: token_dim={token_dim}, inv_dim={inv_dim}, equ_dim={equ_dim}")
        print(f"聚合维度: cptoken_dim={cptoken_dim}, node_dim={node_dim}")

        # 初始化模型并移到设备上
        self.token_encoder = TokenEncoder(11, self.token_dim, self.inv_dim, self.equ_dim).to(self.device)
        self.token_level_attn = TokenLevelAttn(self.token_dim, self.cptoken_dim, num_heads=4, num_layers=3, dropout=0.1).to(self.device)
        self.cp_level_attn = CenterPathLevelAttn(self.cptoken_dim, 11, num_heads=4, num_layers=2, max_num_path=self.max_num_path, dropout=0.1).to(self.device)

        # 跟踪更新的节点数量
        self.total_nodes = 0
        self.updated_nodes = 0

    def _prepare_graph_dict(self, mol_data):
        """将PyG数据转换为图字典格式 - GPU优化版本"""
        # 将数据移到目标设备上
        z = mol_data.z.to(self.device) if hasattr(mol_data, 'z') else None
        pos = mol_data.pos.to(self.device) if hasattr(mol_data, 'pos') else None
        x = mol_data.x.to(self.device) if hasattr(mol_data, 'x') else None
        edge_index = mol_data.edge_index.to(self.device) if hasattr(mol_data, 'edge_index') else None

        # 检查必要属性
        if z is None or pos is None or edge_index is None:
            raise ValueError("分子数据缺少必要属性(z, pos, edge_index)")

        # 如果没有节点特征x，则基于z生成
        if x is None:
            unique_z = torch.unique(z)
            feature_dim = len(unique_z)
            x = torch.zeros((z.size(0), feature_dim), device=self.device)
            for i, atom_z in enumerate(z):
                x[i, (atom_z == unique_z).nonzero(as_tuple=True)[0]] = 1.0

        # 构建图字典 - 保持所有数据在GPU上
        graph = {
            'edge_index': edge_index,
            'pos': {i: pos[i] for i in range(len(z))},  # 保持在GPU上
            'attr': {i: x[i] for i in range(len(z))}
        }

        return graph, x, z


    def _find_centerpath(self, graph, num_nodes):
        """为图中的节点找到中心路径

        参数:
            graph (dict): 图字典
            num_nodes (int): 节点数量

        返回:
            dict: 节点及其中心路径字典
        """
        # 修改FindCenterPath_Node函数以支持GPU上的pos
        mol_centerpaths = {}

        # 为每个节点计算中心路径
        for node_idx in range(num_nodes):
            try:
                result = FindCenterPath_Node(self.max_num_path, self.max_num_hop, node_idx, graph, self.k)
                if result['centerpathlist']:
                    mol_centerpaths[node_idx] = result
            except Exception as e:
                continue  # 跳过处理失败的节点

        return mol_centerpaths

    def process_molecule(self, mol_data, verbose=False):
        """处理单个分子数据"""
        # 准备图字典和其他数据
        graph, x, z = self._prepare_graph_dict(mol_data)
        num_nodes = len(z)
        feature_dim = x.size(1)

        self.total_nodes += num_nodes

        # 寻找中心路径
        mol_centerpaths = self._find_centerpath(graph, num_nodes)

        if not mol_centerpaths:
            if verbose:
                print(f"警告: 未找到任何中心路径，使用原始特征重复")
            # 为没有centerpath的节点创建22维特征：前11维和后11维都是原始特征
            enhanced_data = copy.deepcopy(mol_data)
            original_x = enhanced_data.x.to(self.device)  # 确保在GPU上
            # 拼接原始特征：[原始x, 原始x]
            x_enhanced = torch.cat([original_x, original_x], dim=1)  # (N, 22)
            enhanced_data.x = x_enhanced
            if verbose:
                print(f"  创建了 {num_nodes} 个节点的重复特征，形状: {x_enhanced.shape}")
            return enhanced_data, {} if verbose else None

        if verbose:
            print(f"找到 {len(mol_centerpaths)} 个节点的中心路径")

        # 节点嵌入结果
        node_embeddings = {}

        # 使用批处理方式处理所有节点
        # 收集所有节点的所有路径的tokens
        all_node_paths = []
        all_node_ids = []

        # 处理每个中心节点
        for node_idx, result in mol_centerpaths.items():
            # 1. 为每条centerpath生成tokens
            path_inv_tokens = []
            path_equ_tokens = []

            for path in result['centerpathlist']:
                try:
                    # 生成tokens
                    inv_tokens, equ_tokens = CenterPathEncoder_one(
                        graph, (node_idx, path), self.max_num_hop, self.token_dim,
                        self.token_encoder, self.inv_dim, self.equ_dim
                    )

                    path_inv_tokens.append(inv_tokens)
                    path_equ_tokens.append(equ_tokens)

                except Exception as e:
                    if verbose:
                        print(f"生成tokens出错: {str(e)}")
                    continue

            if not path_inv_tokens:
                continue

            # 2. 使用TokenLevelAttn处理每条路径的tokens
            inv_cptokens = []
            equ_cptokens = []

            for inv_tokens, equ_tokens in zip(path_inv_tokens, path_equ_tokens):
                try:
                    # 准备tokens为批处理格式
                    inv_tokens_batch = torch.stack(inv_tokens).unsqueeze(0)
                    equ_tokens_batch = torch.stack(equ_tokens).unsqueeze(0)

                    # 应用TokenLevelAttn
                    inv_cptoken = self.token_level_attn(inv_tokens_batch)
                    equ_cptoken = self.token_level_attn(equ_tokens_batch)

                    inv_cptokens.append(inv_cptoken.squeeze(0))
                    equ_cptokens.append(equ_cptoken.squeeze(0))

                except Exception as e:
                    if verbose:
                        print(f"生成cptoken出错: {str(e)}")
                    continue

            # 3. 使用CenterPathLevelAttn聚合所有路径的cptokens
            if inv_cptokens:
                try:
                    # 组合所有路径的cptokens
                    num_paths = len(inv_cptokens)

                    # 处理路径数量
                    if num_paths < self.max_num_path:
                        # 创建填充cptoken
                        pad_inv = torch.zeros((self.max_num_path - num_paths, self.cptoken_dim), device=self.device)
                        pad_equ = torch.zeros((self.max_num_path - num_paths, self.cptoken_dim), device=self.device)

                        # 合并实际cptokens和填充
                        inv_cptokens_tensor = torch.cat([torch.stack(inv_cptokens), pad_inv], dim=0)
                        equ_cptokens_tensor = torch.cat([torch.stack(equ_cptokens), pad_equ], dim=0)
                    else:
                        # 如果路径数超过max_num_path，只取前max_num_path个
                        inv_cptokens_tensor = torch.stack(inv_cptokens[:self.max_num_path])
                        equ_cptokens_tensor = torch.stack(equ_cptokens[:self.max_num_path])

                    # 批量处理
                    inv_cptokens_batch = inv_cptokens_tensor.unsqueeze(0)
                    equ_cptokens_batch = equ_cptokens_tensor.unsqueeze(0)

                    # 创建路径数量掩码
                    num_valid_paths = torch.tensor([min(num_paths, self.max_num_path)], device=self.device)

                    # 应用CenterPathLevelAttn - 现在输出11维特征
                    with torch.no_grad():
                        inv_node_embedding = self.cp_level_attn(inv_cptokens_batch, num_valid_paths)  # (1, 11)
                        equ_node_embedding = self.cp_level_attn(equ_cptokens_batch, num_valid_paths)  # (1, 11)

                    # 保存inv和equ特征分别 - 保持在GPU上
                    node_embeddings[node_idx] = {
                        'inv': inv_node_embedding.squeeze(0),  # (11,)
                        'equ': equ_node_embedding.squeeze(0)  # (11,)
                    }

                except Exception as e:
                    if verbose:
                        print(f"生成节点嵌入出错: {str(e)}")
                    continue

        # 如果没有成功生成节点嵌入，返回22维零向量
        if not node_embeddings:
            if verbose:
                print("没有成功生成节点嵌入")
            enhanced_data = copy.deepcopy(mol_data)
            x_enhanced = torch.zeros((num_nodes, 22), dtype=torch.float, device=self.device)
            enhanced_data.x = x_enhanced
            return enhanced_data, {} if verbose else None

        if verbose:
            print(f"成功生成 {len(node_embeddings)} 个节点的嵌入")

        # 4. 直接构造22维特征向量
        # 创建增强数据的深拷贝
        enhanced_data = copy.deepcopy(mol_data)

        # 初始化22维特征矩阵：对于没有centerpath的节点，使用原始特征重复
        original_x = enhanced_data.x.to(self.device)  # 确保在GPU上
        x_enhanced = torch.cat([original_x, original_x], dim=1)  # (N, 22) - 默认重复原始特征

        # 保存特征变化信息
        feature_changes = {}

        # 更新每个有centerpath的节点特征 - 保持在GPU上直到最后
        for node_idx, embeddings in node_embeddings.items():
            inv_features = embeddings['inv']  # 保持在GPU上
            equ_features = embeddings['equ']  # 保持在GPU上

            # 拼接inv和equ特征：前11维是inv，后11维是equ
            combined_features = torch.cat([inv_features, equ_features], dim=0)  # (22,)

            # 更新节点特征
            x_enhanced[node_idx] = combined_features

            # 更新节点计数
            self.updated_nodes += 1

            # 记录特征变化 - 只在需要时移到CPU
            if verbose:
                feature_changes[node_idx] = {
                    'original_features': original_x[node_idx].detach().cpu().numpy(),
                    'inv_features': inv_features.detach().cpu().numpy(),
                    'equ_features': equ_features.detach().cpu().numpy(),
                    'combined_features': combined_features.detach().cpu().numpy()
                }

        # 为没有centerpath的节点记录特征信息
        if verbose:
            no_centerpath_nodes = set(range(num_nodes)) - set(node_embeddings.keys())
            for node_idx in list(no_centerpath_nodes)[:3]:  # 只记录前3个作为示例
                feature_changes[f'no_centerpath_{node_idx}'] = {
                    'original_features': original_x[node_idx].detach().cpu().numpy(),
                    'duplicated_features': x_enhanced[node_idx].detach().cpu().numpy(),
                    'note': 'No centerpath - used duplicated original features'
                }

        # 更新PyG数据对象的节点特征
        enhanced_data.x = x_enhanced

        return enhanced_data, feature_changes if verbose else None



    def process_dataset(self, dataset, output_path=None, start_idx=0, end_idx=None, verbose=True):
        """处理整个数据集

        参数:
            dataset (Dataset): 输入数据集
            output_path (str, optional): 输出文件路径，如果指定则保存处理结果
            start_idx (int): 开始处理的索引
            end_idx (int): 结束处理的索引
            verbose (bool): 是否显示详细信息

        返回:
            list: 增强后的数据列表
        """
        import gc

        end_idx = len(dataset) if end_idx is None else min(end_idx, len(dataset))
        enhanced_dataset = []

        # 创建进度条
        iterator = tqdm(range(start_idx, end_idx), desc="处理分子数据") if verbose else range(start_idx, end_idx)

        # 处理每个分子
        for i in iterator:
            try:
                mol_data = dataset[i]
                enhanced_mol, _ = self.process_molecule(mol_data)
                enhanced_dataset.append(enhanced_mol)
            except Exception as e:
                if verbose:
                    print(f"处理分子 {i} 时出错: {str(e)}")
                enhanced_dataset.append(dataset[i])  # 出错时使用原始数据

        # 保存结果
        #if output_path:
        #    torch.save(enhanced_dataset, output_path)
        #    if verbose:
        #        print(f"增强数据已保存至: {output_path}")
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()

        return enhanced_dataset


def save_enhanced_dataset(enhanced_dataset, output_path):
    """使用原始数据结构和压缩方式保存数据"""
    import pickle
    import gzip
    import os
    import gc

    print(f"开始保存数据集，包含 {len(enhanced_dataset)} 个分子")

    # 创建原始数据列表
    raw_dataset = []
    for data in enhanced_dataset:
        # 将PyTorch张量转换为NumPy数组
        raw_data = {
            'x': data.x.detach().cpu().numpy(),
            'edge_index': data.edge_index.detach().cpu().numpy(),
            'edge_attr': data.edge_attr.detach().cpu().numpy() if hasattr(data, 'edge_attr') else None,
            'y': data.y.detach().cpu().numpy() if hasattr(data, 'y') else None,
            'pos': data.pos.detach().cpu().numpy(),
            'z': data.z.detach().cpu().numpy()
        }

        # 添加字符串元数据
        if hasattr(data, 'smiles'):
            raw_data['smiles'] = data.smiles
        if hasattr(data, 'name'):
            raw_data['name'] = data.name
        if hasattr(data, 'idx'):
            raw_data['idx'] = data.idx.item() if torch.is_tensor(data.idx) else data.idx

        raw_dataset.append(raw_data)

        # 每添加50个分子就进行一次垃圾回收
        if len(raw_dataset) % 50 == 0:
            gc.collect()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为gzip压缩的pickle文件
    file_path = output_path  # 保持原文件路径
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(raw_dataset, f)

    # 清理内存
    del raw_dataset
    gc.collect()

    # 计算文件大小
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"保存完成，文件大小: {file_size_mb:.2f} MB")

    return file_size_mb

def load_enhanced_dataset(file_path):
    """加载增强数据集并转换回PyTorch Geometric数据格式"""
    import pickle
    import gzip
    import torch
    from torch_geometric.data import Data

    print(f"加载增强数据集: {file_path}")

    try:
        # 尝试作为gzip文件加载
        with gzip.open(file_path, 'rb') as f:
            raw_dataset = pickle.load(f)
    except Exception as e:
        print(f"警告: 无法作为gzip文件加载 ({e})，尝试作为普通pickle或PyTorch文件加载...")
        try:
            # 尝试作为普通PyTorch文件加载
            raw_dataset = torch.load(file_path)
        except:
            print("错误: 无法加载数据文件")
            raise

    # 如果加载的是原始列表格式
    if isinstance(raw_dataset, list) and isinstance(raw_dataset[0], dict):
        # 转换回PyG数据对象
        dataset = []
        for raw_data in raw_dataset:
            data = Data(
                x=torch.tensor(raw_data['x'], dtype=torch.float),
                edge_index=torch.tensor(raw_data['edge_index'], dtype=torch.long),
                pos=torch.tensor(raw_data['pos'], dtype=torch.float),
                z=torch.tensor(raw_data['z'], dtype=torch.long)
            )

            if raw_data['edge_attr'] is not None:
                data.edge_attr = torch.tensor(raw_data['edge_attr'], dtype=torch.float)

            if raw_data['y'] is not None:
                data.y = torch.tensor(raw_data['y'], dtype=torch.float)

            if 'smiles' in raw_data:
                data.smiles = raw_data['smiles']

            if 'name' in raw_data:
                data.name = raw_data['name']

            if 'idx' in raw_data:
                data.idx = raw_data['idx']

            dataset.append(data)
    else:
        # 已经是PyG数据对象的列表
        dataset = raw_dataset

    print(f"加载了 {len(dataset)} 个分子数据")
    return dataset


class EnhancedDataset(Dataset):
    def __init__(self, original_dataset, processor=None, transform=None, pre_transform=None,
                 max_num_path=6, max_num_hop=4, k=2, node_dim=11, batch_processing=True, cache_file=None):
        """初始化增强数据集

        参数:
            original_dataset (Dataset): 原始数据集
            processor (EnhancedDataProcessor, optional): 数据处理器
            transform (callable, optional): 数据变换函数
            pre_transform (callable, optional): 数据预处理函数
            max_num_path (int): 最大路径数
            max_num_hop (int): 最大跳数
            k (int): 节点最小度数阈值
            batch_processing (bool): 是否批量处理整个数据集
            cache_file (str, optional): 缓存文件路径
        """
        super(EnhancedDataset, self).__init__(transform=transform, pre_transform=pre_transform)
        self.original_dataset = original_dataset
        self.batch_processing = batch_processing
        self.cache_file = cache_file
        self.node_dim = node_dim  # 现在是11
        self.enhanced_data = None

        # 创建或加载处理器
        if processor is None:
            self.processor = EnhancedDataProcessor(
                max_num_path=max_num_path,
                max_num_hop=max_num_hop,
                k=k
            )
        else:
            self.processor = processor

        # 如果指定了缓存文件，尝试加载
        if cache_file:
            try:
                self.enhanced_data = torch.load(cache_file)
                print(f"已从 {cache_file} 加载缓存数据")
                return
            except:
                print(f"无法加载缓存文件 {cache_file}，将进行实时处理")

        # 批量处理数据集
        if batch_processing:
            print("批量处理整个数据集...")
            self.enhanced_data = self.processor.process_dataset(original_dataset)

            # 保存处理结果
            if cache_file:
                torch.save(self.enhanced_data, cache_file)
                print(f"增强数据已缓存至: {cache_file}")

    def len(self):
        return len(self.original_dataset)

    def get(self, idx):
        # 如果已经批量处理过，返回处理结果
        if self.enhanced_data is not None:
            return self.enhanced_data[idx]

        # 否则实时处理单个样本
        mol_data = self.original_dataset[idx]
        enhanced_mol, _ = self.processor.process_molecule(mol_data)
        return enhanced_mol


def enhance_dataset(input_dataset, output_path=None, max_num_path=6, max_num_hop=4, k=2,
                    token_dim=32, inv_dim=12, equ_dim=12, cptoken_dim=64, node_dim=128,
                    batch_size=None, device=None, verbose=True):
    """增强分子数据集的便捷函数，添加内存管理"""
    import gc
    import os
    import time
    import torch
    import psutil  # 用于监控内存使用情况

    # 初始化处理器
    processor = EnhancedDataProcessor(
        max_num_path=max_num_path,
        max_num_hop=max_num_hop,
        k=k,
        token_dim=token_dim,
        inv_dim=inv_dim,
        equ_dim=equ_dim,
        cptoken_dim=cptoken_dim,
        node_dim=node_dim,
        device=device
    )

    # 处理数据集
    if batch_size is None:
        # 处理整个数据集
        enhanced_data = processor.process_dataset(input_dataset, output_path=None, verbose=verbose)
        return enhanced_data
    else:
        # 分批处理数据集
        dataset_size = len(input_dataset)
        processed_count = 0

        # 确保输出目录存在
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 获取输出目录
        processed_dir = os.path.dirname(output_path) if output_path else "./data/QM9_enhanced"
        os.makedirs(processed_dir, exist_ok=True)

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_num = end_idx // batch_size

            # 显示内存使用情况
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # 转换为MB

            print(f"处理批次: {start_idx}~{end_idx - 1} / {dataset_size} (内存使用: {mem_before:.1f} MB)")

            try:
                # 处理当前批次
                batch_data = processor.process_dataset(
                    input_dataset,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    verbose=verbose
                )

                # 保存当前批次
                batch_output_path = os.path.join(processed_dir, f"qm9_enhanced_batch_{batch_num}.pt")
                print(f'开始保存预处理信息：批次{batch_num}')
                save_enhanced_dataset(batch_data, batch_output_path)
                print(f'预处理信息保存完成：批次{batch_num}')

                # 更新计数
                processed_count += len(batch_data)

                # 清理内存
                del batch_data
                torch.cuda.empty_cache()  # 清理GPU缓存
                gc.collect()  # 强制垃圾回收

                # 显示清理后的内存使用情况
                mem_after = process.memory_info().rss / 1024 / 1024
                print(f"批次处理完成，内存使用: {mem_after:.1f} MB (变化: {mem_after - mem_before:.1f} MB)")

            except Exception as e:
                print(f"处理批次 {batch_num} 时出错: {str(e)}")
                # 继续处理下一批次
                continue

        # 输出处理统计信息
        if verbose:
            print(f"处理统计: 总节点数 {processor.total_nodes}, 更新节点数 {processor.updated_nodes}")
            update_rate = processor.updated_nodes / processor.total_nodes * 100 if processor.total_nodes > 0 else 0
            print(f"节点更新率: {update_rate:.2f}%")
            print(f"成功处理的分子数量: {processed_count}/{dataset_size}")

        return None  # 分批处理模式下不返回数据，只保存到文件


def compare_features(original_dataset, enhanced_data, sample_indices=[0], num_nodes=3):
    """比较原始特征和增强后特征的差异"""
    print("\n=== 特征变化详细比较 ===")

    for idx in sample_indices:
        if idx >= len(original_dataset):
            continue

        print(f"\n分子 {idx}:")
        orig_mol = original_dataset[idx]

        # 处理增强数据可能是列表的情况
        if isinstance(enhanced_data, list):
            enh_mol = enhanced_data[idx]  # 字典形式
            enh_x = enh_mol['x']  # NumPy数组
            orig_x = orig_mol.x.detach().cpu().numpy()
        else:
            enh_mol = enhanced_data[idx]  # PyG Data对象
            enh_x = enh_mol.x.detach().cpu().numpy()
            orig_x = orig_mol.x.detach().cpu().numpy()

        # 比较前n个节点的特征
        for node_idx in range(min(num_nodes, len(orig_x))):
            print(f"\n节点 {node_idx}:")
            print(f"  原始特征: {orig_x[node_idx][:5]}")
            print(f"  增强特征: {enh_x[node_idx][:5]}")


def merge_pt_files(input_files, output_path):
    """合并多个.pt文件到一个文件，保持原来的pt格式

    参数:
        input_files: 输入文件路径列表
        output_path: 输出文件路径
    """
    import os
    import torch
    from tqdm import tqdm
    import gzip
    import pickle

    print(f"合并 {len(input_files)} 个文件到 {output_path}")

    # 验证所有输入文件存在
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

    # 合并所有数据
    all_data = []
    total_molecules = 0

    for file_path in tqdm(input_files, desc="合并文件"):
        try:
            # 尝试不同的加载方式
            try:
                # 尝试作为gzip文件加载
                with gzip.open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
            except:
                # 尝试作为普通PyTorch文件加载
                batch_data = torch.load(file_path)

            batch_size = len(batch_data)
            all_data.extend(batch_data)
            total_molecules += batch_size
            print(f"  已加载 {file_path}: {batch_size} 个分子")
        except Exception as e:
            print(f"警告: 加载文件 {file_path} 失败: {e}")

    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 使用torch.save保存合并后的数据，保持.pt格式
    torch.save(all_data, output_path)

    # 计算文件大小
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"合并完成! 总计 {total_molecules} 个分子")
    print(f"输出文件: {output_path}, 大小: {file_size_mb:.2f} MB")

    return total_molecules, file_size_mb


if __name__ == "__main__":
    # 示例用法
    from torch_geometric.datasets import QM9
    import os
    import random
    import glob
    import gc


    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)

    # 设置参数
    root = "./data"
    raw_dir = os.path.join(root, "QM9")
    processed_dir = os.path.join(root, "QM9_enhanced")
    ''''''
    # 确保目录存在
    os.makedirs(processed_dir, exist_ok=True)

    # 加载QM9数据集
    print("加载QM9数据集...")
    dataset = QM9(root=raw_dir)
    print(f"数据集大小: {len(dataset)}")

    # 增强小批量数据进行测试
    test_size = 130000
    print(f"测试处理前 {test_size} 个分子...")

    enhanced_data = enhance_dataset(
        dataset[:test_size],
        output_path=os.path.join(processed_dir, "qm9_enhanced_test.pt"),
        max_num_path=6,
        max_num_hop=4,
        k=2,
        batch_size=50,
        verbose=True
    )

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()

    # 查找所有批次文件
    batch_files = sorted(glob.glob("./data/QM9_enhanced/qm9_enhanced_batch_*.pt"))

    if batch_files:
        # 合并所有批次文件
        output_file = "./data/QM9_enhanced/qm9_enhanced_test.pt"
        merge_pt_files(batch_files, output_file)
    else:
        print("未找到批次文件!")

    output_path = os.path.join(processed_dir, "qm9_enhanced_test.pt")
    #print('开始保存预处理信息！')
    #save_enhanced_dataset(enhanced_data,output_path)
    #print('预处理信息保存完成！')

    # 检查基本结果
    #print(f"增强前第一个分子节点特征: {dataset[0].x.shape}")
    #print(f"增强后第一个分子节点特征: {enhanced_data[0].x.shape}")

    # 详细比较几个样本的特征变化
    print("\n=== 特征变化详细比较 ===")
    sample_indices = [0, 5, 10]  # 选择几个样本进行比较

    enhanced_data_pt = load_enhanced_dataset(output_path)
    compare_features(dataset[:test_size], enhanced_data_pt, sample_indices=sample_indices, num_nodes=3)
