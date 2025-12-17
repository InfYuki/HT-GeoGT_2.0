# ht_geogt/model/collating_geoformer.py 修改部分
from typing import Any, Dict, List

import torch


class GeoformerDataCollator:
    def __init__(self, max_nodes=None) -> None:
        self.max_nodes = max_nodes

    @staticmethod
    def _pad_attn_bias(attn_bias: torch.Tensor, max_node: int) -> torch.Tensor:
        N = attn_bias.shape[0]
        if N <= max_node:
            attn_bias_padded = torch.zeros(
                [max_node, max_node], dtype=torch.float
            ).fill_(float("-inf"))
            attn_bias_padded[:N, :N] = attn_bias
            attn_bias_padded[N:, :N] = 0
        else:
            print(
                f"Warning: max_node {max_node} is too small to hold all nodes {N} in a batch"
            )
            print("Play truncation...")

        return attn_bias_padded

    @staticmethod
    def _pad_feats(feats: torch.Tensor, max_node: int) -> torch.Tensor:
        N, *_ = feats.shape
        if N <= max_node:
            feats_padded = torch.zeros([max_node, *_], dtype=feats.dtype)
            feats_padded[:N] = feats
        else:
            print(
                f"Warning: max_node {max_node} is too small to hold all nodes {N} in a batch"
            )
            print("Play truncation...")

        return feats_padded

    def _check_attn_bias(self, feat: dict):
        num_node = len(feat["z"])
        if "attn_bias" not in feat:
            return torch.zeros([num_node, num_node]).float()
        else:
            return torch.tensor(feat["attn_bias"]).float()

    def _create_edge_matrix(self, edge_index, edge_attr, num_nodes, max_node):
        """从边索引和边特征创建边特征矩阵"""
        # 初始化边特征矩阵
        edge_dim = edge_attr.shape[-1] if edge_attr.dim() > 1 else 1
        if edge_dim == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        edge_matrix = torch.zeros(max_node, max_node, edge_dim, dtype=edge_attr.dtype)

        # 填充边特征
        row, col = edge_index
        for i in range(len(row)):
            if row[i] < max_node and col[i] < max_node:
                edge_matrix[row[i], col[i]] = edge_attr[i]

        return edge_matrix

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        batch = dict()

        max_node = (
            max(feat["z"].shape[0] for feat in features)
            if self.max_nodes is None
            else self.max_nodes
        )

        batch["z"] = torch.stack(
            [self._pad_feats(feat["z"], max_node) for feat in features]
        )
        batch["pos"] = torch.stack(
            [self._pad_feats(feat["pos"], max_node) for feat in features]
        )

        # 添加节点特征 x 的处理
        if hasattr(features[0], "x") and features[0].x is not None:
            batch["x"] = torch.stack(
                [self._pad_feats(feat["x"], max_node) for feat in features]
            )

        # 处理边信息
        edge_matrices = []
        for feat in features:
            num_nodes = feat["z"].shape[0]

            # 检查是否有边信息
            if hasattr(feat, "edge_index") and hasattr(feat, "edge_attr"):
                edge_matrix = self._create_edge_matrix(
                    feat.edge_index,
                    feat.edge_attr,
                    num_nodes,
                    max_node
                )
            else:
                # 如果没有边信息，创建一个空的边特征矩阵
                print('没有边信息\n')
                edge_matrix = torch.zeros(max_node, max_node, 4)  # 假设边特征维度为4

            edge_matrices.append(edge_matrix)

        batch["edge_attr"] = torch.stack(edge_matrices)
        batch["labels"] = torch.cat([feat["y"] for feat in features])

        return batch