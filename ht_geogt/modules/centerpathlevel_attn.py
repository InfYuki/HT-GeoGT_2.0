import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CenterPathLevelAttn(nn.Module):
    """Centerpath级的自注意力层

    功能：对一个节点的所有cptoken做自注意力机制，并聚合成一个node_dim维的node_embedding

    参数：
    cptoken_dim: int - 输入cptoken的维度
    node_dim: int - 输出node_embedding的维度
    num_heads: int - 注意力头数
    num_layers: int - 自注意力层数
    max_num_path: int - 最大centerpath数量
    dropout: float - dropout率
    """

    def __init__(self, cptoken_dim, node_dim, num_heads=4, num_layers=2, max_num_path=2, dropout=0.1):
        super(CenterPathLevelAttn, self).__init__()

        self.cptoken_dim = cptoken_dim
        self.node_dim = node_dim
        self.max_num_path = max_num_path

        # 多层自注意力
        self.layers = nn.ModuleList([
            CPAttnLayer(cptoken_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影层
        self.out_proj = nn.Linear(cptoken_dim, node_dim)
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, cptokens, num_valid_paths):
        """
        参数：
        cptokens: tensor of shape (batch_size, max_num_path, cptoken_dim)
            一个节点的所有cptoken
        num_valid_paths: tensor of shape (batch_size,)
            每个节点实际的centerpath数量

        返回：
        node_embedding: tensor of shape (batch_size, node_dim)
            节点嵌入
        """
        batch_size = cptokens.size(0)
        device = cptokens.device  # 获取输入张量的设备

        # 创建mask：标记有效的cptoken位置
        mask = torch.arange(self.max_num_path, device=device).unsqueeze(0) < num_valid_paths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, max_num_path]

        # 多层自注意力处理
        x = cptokens
        for layer in self.layers:
            x = layer(x, mask)

        # 只对有效的cptoken进行平均
        valid_tokens = mask.squeeze(1).squeeze(1)  # [batch_size, max_num_path]
        x = x * valid_tokens.unsqueeze(-1)
        output = x.sum(dim=1) / num_valid_paths.unsqueeze(1).clamp(min=1)

        # 通过输出投影层和层归一化
        node_embedding = self.out_proj(output)
        node_embedding = self.layer_norm(node_embedding)

        return node_embedding


class CPAttnLayer(nn.Module):
    """单层centerpath自注意力模块"""

    def __init__(self, cptoken_dim, num_heads, dropout=0.1):
        super(CPAttnLayer, self).__init__()

        self.cptoken_dim = cptoken_dim
        self.num_heads = num_heads
        self.head_dim = cptoken_dim // num_heads

        # Q,K,V的线性变换层
        self.q_linear = nn.Linear(cptoken_dim, cptoken_dim)
        self.k_linear = nn.Linear(cptoken_dim, cptoken_dim)
        self.v_linear = nn.Linear(cptoken_dim, cptoken_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(cptoken_dim, cptoken_dim * 4),
            nn.ReLU(),
            nn.Linear(cptoken_dim * 4, cptoken_dim)
        )

        self.norm1 = nn.LayerNorm(cptoken_dim)
        self.norm2 = nn.LayerNorm(cptoken_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 多头自注意力计算
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # 分头
        batch_size, num_paths, _ = x.size()
        q = q.view(batch_size, num_paths, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_paths, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_paths, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合
        attn_output = torch.matmul(attn_weights, v)

        # 重组多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_paths, self.cptoken_dim)

        # 残差连接和层归一化
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x


def test_centerpath_level_attn_with_qm9():
    """测试CenterPathLevelAttn模块与QM9数据集"""
    import os
    import numpy as np
    import random
    from torch_geometric.datasets import QM9
    from ht_geogt.modules.PathFinder import FindCenterPath_Node
    from ht_geogt.modules.encode_tokens import CenterPathEncoder_one, TokenEncoder
    from ht_geogt.modules.tokenlevel_attn import TokenLevelAttn

    # 设置随机种子以保证结果可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置数据集路径
    dataset_root = './data/QM9'

    # 加载QM9数据集
    dataset = QM9(root=dataset_root)
    print(f"数据集大小: {len(dataset)} 分子")

    # 选择几个分子样本进行测试，形成一个小批次
    batch_size = 4
    sample_indices = list(range(batch_size))

    # 设置参数
    max_num_path = 4  # 最大路径数
    max_num_hop = 3  # 最大跳数
    k = 2  # 最小度数阈值
    token_dim = 32  # token维度
    inv_dim = 12  # 不变特征维度
    equ_dim = 12  # 等变特征维度
    cptoken_dim = 64  # cptoken维度
    node_dim = 128  # node_embedding维度

    # 处理每个分子样本
    all_inv_cptokens = []
    all_equ_cptokens = []
    all_num_valid_paths = []

    for idx in sample_indices:
        data = dataset[idx]

        # 检查节点特征
        if hasattr(data, 'x') and data.x is not None:
            node_features = data.x.to(device)
            feature_dim = data.x.size(1)
        else:
            # 如果没有节点特征，则使用one-hot编码的原子类型
            unique_z = torch.unique(data.z)
            feature_dim = len(unique_z)
            node_features = torch.zeros((data.z.size(0), feature_dim), device=device)
            for i, z in enumerate(data.z):
                node_features[i, (z == unique_z).nonzero(as_tuple=True)[0]] = 1.0

        # 将PyG数据转换为图字典格式
        graph = {
            'edge_index': data.edge_index.to(device),
            'edge_attr': data.edge_attr.to(device) if hasattr(data,
                                                              'edge_attr') and data.edge_attr is not None else None,
            'pos': {i: pos.cpu().numpy() for i, pos in enumerate(data.pos)},
            'attr': {i: feat.to(device) for i, feat in enumerate(node_features)}
        }

        # 第一步：使用PathFinder为所有节点生成centerpath
        all_centerpaths = {}
        for node_idx in range(len(data.z)):
            result = FindCenterPath_Node(max_num_path, max_num_hop, node_idx, graph, k)
            if result['centerpathlist']:  # 只保存有centerpath的节点
                all_centerpaths[node_idx] = [(node_idx, result['centerpathlist'])]

        # 第二步：初始化TokenEncoder并对每个centerpath进行编码
        token_encoder = TokenEncoder(feature_dim, token_dim, inv_dim, equ_dim).to(device)

        # 存储每个节点的所有centerpath的编码结果
        node_tokens = {}

        for node_idx, paths in all_centerpaths.items():
            node_inv_tokens = []
            node_equ_tokens = []

            for path_data in paths:
                node_id, centerpaths = path_data
                for centerpath in centerpaths:
                    # 构造正确的格式传递给CenterPathEncoder_one
                    centerpath_data = (node_id, centerpath)
                    try:
                        inv_tokens, equ_tokens = CenterPathEncoder_one(
                            graph, centerpath_data, max_num_hop, token_dim, token_encoder, inv_dim, equ_dim
                        )
                        node_inv_tokens.append(inv_tokens)
                        node_equ_tokens.append(equ_tokens)
                    except Exception as e:
                        print(f"编码节点 {node_idx} 的centerpath时出错: {str(e)}")
                        continue

            if node_inv_tokens and node_equ_tokens:  # 只保存成功编码的节点
                node_tokens[node_idx] = {
                    'inv_tokens': node_inv_tokens,
                    'equ_tokens': node_equ_tokens
                }

        # 第三步：初始化TokenLevelAttn并处理编码后的tokens
        token_level_attn = TokenLevelAttn(
            token_dim=token_dim,
            cptoken_dim=cptoken_dim,
            num_heads=4,
            num_layers=3,
            dropout=0.1
        ).to(device)

        # 存储每个节点的cptoken
        node_cptokens = {}

        for node_idx, tokens_dict in node_tokens.items():
            inv_tokens_list = tokens_dict['inv_tokens']
            equ_tokens_list = tokens_dict['equ_tokens']

            # 处理不变性tokens
            inv_cptokens = []
            for inv_tokens in inv_tokens_list:
                # 将tokens列表转换为批量格式
                tokens_batch = torch.stack([token for token in inv_tokens]).unsqueeze(0)  # [1, num_tokens, token_dim]
                try:
                    # 使用TokenLevelAttn处理
                    cptoken = token_level_attn(tokens_batch)
                    inv_cptokens.append(cptoken)
                except Exception as e:
                    print(f"处理节点 {node_idx} 的不变性tokens时出错: {str(e)}")
                    continue

            # 处理等变性tokens
            equ_cptokens = []
            for equ_tokens in equ_tokens_list:
                # 将tokens列表转换为批量格式
                tokens_batch = torch.stack([token for token in equ_tokens]).unsqueeze(0)  # [1, num_tokens, token_dim]
                try:
                    # 使用TokenLevelAttn处理
                    cptoken = token_level_attn(tokens_batch)
                    equ_cptokens.append(cptoken)
                except Exception as e:
                    print(f"处理节点 {node_idx} 的等变性tokens时出错: {str(e)}")
                    continue

            if inv_cptokens and equ_cptokens:  # 只保存成功处理的节点
                node_cptokens[node_idx] = {
                    'inv_cptokens': torch.cat(inv_cptokens, dim=0),  # [num_paths, cptoken_dim]
                    'equ_cptokens': torch.cat(equ_cptokens, dim=0)  # [num_paths, cptoken_dim]
                }

        # 选择第一个节点的cptokens进行后续处理
        if node_cptokens:
            first_node = list(node_cptokens.keys())[0]
            inv_cptokens = node_cptokens[first_node]['inv_cptokens']
            equ_cptokens = node_cptokens[first_node]['equ_cptokens']

            # 填充到max_num_path
            num_valid = inv_cptokens.shape[0]
            if num_valid < max_num_path:
                pad_size = max_num_path - num_valid
                inv_pad = torch.zeros((pad_size, cptoken_dim), device=device)
                equ_pad = torch.zeros((pad_size, cptoken_dim), device=device)
                inv_cptokens = torch.cat([inv_cptokens, inv_pad], dim=0)
                equ_cptokens = torch.cat([equ_cptokens, equ_pad], dim=0)
            elif num_valid > max_num_path:
                inv_cptokens = inv_cptokens[:max_num_path]
                equ_cptokens = equ_cptokens[:max_num_path]
                num_valid = max_num_path

            all_inv_cptokens.append(inv_cptokens.unsqueeze(0))  # [1, max_num_path, cptoken_dim]
            all_equ_cptokens.append(equ_cptokens.unsqueeze(0))  # [1, max_num_path, cptoken_dim]
            all_num_valid_paths.append(num_valid)

    # 合并所有样本的cptokens
    if all_inv_cptokens and all_equ_cptokens:
        batch_inv_cptokens = torch.cat(all_inv_cptokens, dim=0)  # [batch_size, max_num_path, cptoken_dim]
        batch_equ_cptokens = torch.cat(all_equ_cptokens, dim=0)  # [batch_size, max_num_path, cptoken_dim]
        batch_num_valid_paths = torch.tensor(all_num_valid_paths, device=device)  # [batch_size]

        # 第四步：初始化CenterPathLevelAttn并处理cptoken
        cp_level_attn = CenterPathLevelAttn(
            cptoken_dim=cptoken_dim,
            node_dim=node_dim,
            num_heads=4,
            num_layers=2,
            max_num_path=max_num_path,
            dropout=0.1
        ).to(device)

        # 处理不变性和等变性cptoken
        inv_node_embeddings = cp_level_attn(batch_inv_cptokens, batch_num_valid_paths)
        equ_node_embeddings = cp_level_attn(batch_equ_cptokens, batch_num_valid_paths)

        # 打印结果
        print("\nCenterPathLevelAttn处理结果:")
        print(f"  不变性node_embedding形状: {inv_node_embeddings.shape}")
        print(f"  等变性node_embedding形状: {equ_node_embeddings.shape}")
        print(f"  第一个不变性node_embedding前5个值: {inv_node_embeddings[0, :5]}")
        print(f"  第一个等变性node_embedding前5个值: {equ_node_embeddings[0, :5]}")

        # 验证结果
        assert inv_node_embeddings.shape == (len(all_num_valid_paths), node_dim), "不变性node_embedding维度错误"
        assert equ_node_embeddings.shape == (len(all_num_valid_paths), node_dim), "等变性node_embedding维度错误"

        print("CenterPathLevelAttn测试通过!")

        return inv_node_embeddings, equ_node_embeddings
    else:
        print("没有足够的样本进行CenterPathLevelAttn测试")
        return None, None


if __name__ == "__main__":
    test_centerpath_level_attn_with_qm9()