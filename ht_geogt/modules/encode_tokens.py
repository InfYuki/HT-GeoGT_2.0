import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sympy.physics.wigner import clebsch_gordan


# 预计算并缓存Clebsch-Gordan系数
def precompute_cg_coeffs(max_degree=2):
    cg_coeffs = {}
    for l1 in range(max_degree + 1):
        for l2 in range(max_degree + 1):
            for l in range(abs(l1 - l2), min(l1 + l2 + 1, max_degree + 1)):
                cg_coeffs[(l1, l2, l, 0, 0, 0)] = float(clebsch_gordan(l1, l2, l, 0, 0, 0))
    return cg_coeffs


# 全局缓存的CG系数
CG_COEFFS = precompute_cg_coeffs(max_degree=2)


def calculate_Wcenters_torch(point1, point2, point3):
    """计算三个点的重心 - PyTorch版本"""
    return (point1 + point2 + point3) / 3


def euclidean_distance_torch(point1, point2):
    """计算两点间的欧氏距离 - PyTorch版本"""
    return torch.norm(point1 - point2)


def calculate_spherical_harmonics_torch(vec, max_degree=2):
    """计算球谐函数 - PyTorch版本"""
    device = vec.device
    r = torch.norm(vec)
    if r < 1e-10:
        return torch.zeros((max_degree + 1) ** 2, device=device)

    # 计算球坐标
    theta = torch.acos(vec[2] / r)
    phi = torch.atan2(vec[1], vec[0])

    # 初始化结果
    Y = []

    # 计算球谐函数
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            # 计算关联Legendre多项式
            if m == 0:
                # P_l^0(x) = Legendre多项式
                x = torch.cos(theta)
                if l == 0:
                    P = torch.ones_like(x)
                elif l == 1:
                    P = x
                elif l == 2:
                    P = 0.5 * (3 * x * x - 1)
                else:
                    # 对于更高阶，可以使用递归关系
                    P = ((2 * l - 1) * x * calculate_legendre_torch(l - 1, x) -
                         (l - 1) * calculate_legendre_torch(l - 2, x)) / l
            else:
                # 对于m!=0的情况，使用近似计算
                # 这里简化处理，实际应用中可能需要更精确的计算
                P = torch.sin(theta) ** abs(m)

            # 计算球谐函数
            if m >= 0:
                Y_lm = torch.cos(m * phi) * P
            else:
                Y_lm = torch.sin(abs(m) * phi) * P

            Y.append(Y_lm)

    return torch.stack(Y)


def calculate_legendre_torch(n, x):
    """计算Legendre多项式 - PyTorch版本"""
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * calculate_legendre_torch(n - 1, x) -
                (n - 1) * calculate_legendre_torch(n - 2, x)) / n


def calculate_equivariant_features_steerable_torch(point1, point2, point3, max_degree=2):
    """计算高阶可转向特征 - PyTorch版本"""
    device = point1.device
    v1 = point2 - point1
    v2 = point3 - point1

    # 使用预计算的特征
    # 这里简化处理，实际应用可能需要更精确的计算
    feat_dim = (max_degree + 1) ** 4  # 简化的特征维度
    features = torch.zeros(feat_dim, device=device)

    # 计算一些基本特征
    dot_product = torch.dot(v1, v2)
    cross_product = torch.cross(v1, v2)
    v1_norm = torch.norm(v1)
    v2_norm = torch.norm(v2)

    # 填充特征向量
    features[0] = dot_product / (v1_norm * v2_norm + 1e-8)  # 夹角余弦
    features[1:4] = cross_product / (v1_norm * v2_norm + 1e-8)  # 归一化叉积
    features[4] = v1_norm
    features[5] = v2_norm

    return features


class GaussianBasisKernel(nn.Module):
    def __init__(self, K=8, gamma_min=0.005, gamma_max=2.0):
        super().__init__()
        self.gamma = nn.Parameter(
            torch.logspace(
                start=math.log10(gamma_min),
                end=math.log10(gamma_max),
                steps=K
            ),
            requires_grad=False
        )

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * d.pow(2).unsqueeze(-1))


class TokenEncoder(nn.Module):
    def __init__(self, feature_dim, token_dim, inv_dim, equ_dim):
        super(TokenEncoder, self).__init__()
        self.linear_layer_inv = nn.Linear(feature_dim, token_dim - inv_dim)
        self.linear_layer_equ = nn.Linear(feature_dim, token_dim - equ_dim)

        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.linear_layer_inv.weight)
        nn.init.xavier_uniform_(self.linear_layer_equ.weight)
        nn.init.constant_(self.linear_layer_inv.bias, 0.0)
        nn.init.constant_(self.linear_layer_equ.bias, 0.0)

        # 添加高斯核
        self.gaussian_kernel = GaussianBasisKernel(K=inv_dim // 3, gamma_min=0.01, gamma_max=2.0)


def CenterPathEncoder_one(graph, centerpath_one, max_num_hop, token_dim, token_encoder, inv_dim=12, equ_dim=12):
    """把一条centerpath编码成一组token - GPU优化版本"""
    # 获取设备
    device = next(token_encoder.parameters()).device

    # 使用token_encoder中的高斯核
    gaussian_kernel = token_encoder.gaussian_kernel

    # 从centerpath_one中提取节点ID和路径
    node_id, path = centerpath_one

    # 过滤掉路径中的-1填充
    path = [p for p in path if p != -1]

    # 获取中心节点
    center_idx = len(path) // 2
    startnode = node_id

    # 检查中心节点是否在图中
    if startnode not in graph['pos'] or startnode not in graph['attr']:
        raise ValueError(f"中心节点 {startnode} 不在图中")

    # 计算最大有效跳数
    k = len(path) // 2  # 不包括中心节点自身

    # 初始化tokens列表，大小为max_num_hop+1
    inv_tokens = []
    equ_tokens = []

    # 获取节点位置和属性
    startnode_pos = graph['pos'][startnode]
    startnode_attr = graph['attr'][startnode]

    # 使用传入的token_encoder
    linear_layer_inv = token_encoder.linear_layer_inv
    linear_layer_equ = token_encoder.linear_layer_equ

    for i in range(max_num_hop + 1):
        if i == 0:
            # 处理中心节点
            node_feature = startnode_attr
            feat_token_inv = linear_layer_inv(node_feature)
            feat_token_equ = linear_layer_equ(node_feature)

            # i==0时,距离为0,高斯核值为1
            dist_token = torch.ones(inv_dim, device=device)
            equ_token = torch.zeros(equ_dim, device=device)

            # 拼接特征
            inv_token = torch.cat([feat_token_inv, dist_token])
            equ_token = torch.cat([feat_token_equ, equ_token])

            inv_tokens.append(inv_token)
            equ_tokens.append(equ_token)

        elif 0 < i <= k:
            # 获取i跳距离的两个邻居节点
            left_idx = center_idx - i
            right_idx = center_idx + i

            if 0 <= left_idx < len(path) and 0 <= right_idx < len(path):
                left_neighbor = path[left_idx]
                right_neighbor = path[right_idx]

                # 验证节点在图中存在
                if (left_neighbor not in graph['pos'] or right_neighbor not in graph['pos'] or
                        left_neighbor not in graph['attr'] or right_neighbor not in graph['attr']):
                    continue

                # 获取节点特征
                leftnode_pos = graph['pos'][left_neighbor]
                left_feat = graph['attr'][left_neighbor]

                rightnode_pos = graph['pos'][right_neighbor]
                right_feat = graph['attr'][right_neighbor]

                center_feat_inv = inv_tokens[0][:token_dim - inv_dim]
                center_feat_equ = equ_tokens[0][:token_dim - equ_dim]

                # 计算三点重心 - 使用PyTorch版本
                Wcenter = calculate_Wcenters_torch(startnode_pos, leftnode_pos, rightnode_pos)

                # 计算距离 - 使用PyTorch版本
                dist_start = euclidean_distance_torch(startnode_pos, Wcenter)
                dist_left = euclidean_distance_torch(leftnode_pos, Wcenter)
                dist_right = euclidean_distance_torch(rightnode_pos, Wcenter)

                # 高斯核
                dist_tensor = torch.stack([dist_start, dist_left, dist_right])
                dist_features = gaussian_kernel(dist_tensor).flatten()

                # 调整维度
                if len(dist_features) >= inv_dim:
                    dist_token = dist_features[:inv_dim]
                else:
                    repeats = inv_dim // len(dist_features) + 1
                    dist_token = dist_features.repeat(repeats)[:inv_dim]

                # 计算等变特征 - 使用PyTorch版本
                try:
                    equ_features = calculate_equivariant_features_steerable_torch(
                        startnode_pos, leftnode_pos, rightnode_pos, max_degree=2
                    )

                    if len(equ_features) >= equ_dim:
                        equ_token = equ_features[:equ_dim]
                    else:
                        pad_size = equ_dim - len(equ_features)
                        equ_token = torch.cat([equ_features, torch.zeros(pad_size, device=device)])

                except Exception as e:
                    equ_token = torch.zeros(equ_dim, device=device)

                # 特征聚合
                left_feat_inv = linear_layer_inv(left_feat)
                right_feat_inv = linear_layer_inv(right_feat)
                left_feat_equ = linear_layer_equ(left_feat)
                right_feat_equ = linear_layer_equ(right_feat)

                feat_token_inv = torch.mean(torch.stack([center_feat_inv, left_feat_inv, right_feat_inv]), dim=0)
                feat_token_equ = torch.mean(torch.stack([center_feat_equ, left_feat_equ, right_feat_equ]), dim=0)

                # 拼接特征
                inv_token = torch.cat([feat_token_inv, dist_token])
                equ_token = torch.cat([feat_token_equ, equ_token])

                inv_tokens.append(inv_token)
                equ_tokens.append(equ_token)
            else:
                # 跳数超出路径范围
                continue
        else:
            # 超出有效路径跳数范围
            continue

    return inv_tokens, equ_tokens