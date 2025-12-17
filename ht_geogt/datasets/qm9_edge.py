import os
import torch
import numpy as np
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from torch_geometric.transforms import Compose


class QM9Edge(QM9_geometric):
    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            dataset_arg=None,
    ):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )

        self.label = dataset_arg
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM9Edge, self).__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def get(self, idx):
        data = super().get(idx)

        # 确保数据包含边信息
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            raise ValueError(f"数据点 {idx} 没有边信息")

        # 如果需要，可以添加边特征
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            # 创建简单的边特征（例如，基于原子间距离）
            row, col = data.edge_index
            pos = data.pos
            dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
            data.edge_attr = dist  # 简单地使用距离作为边特征

        return data