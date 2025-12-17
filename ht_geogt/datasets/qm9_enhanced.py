import os
import torch
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from torch_geometric.transforms import Compose


class QM9Enhanced(QM9_geometric):
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

        super(QM9_geometric, self).__init__(  # 跳过QM9_geometric的__init__
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        # 直接加载预处理的数据
        self.data, self.slices = torch.load(self.processed_paths[0])

        # 调试信息
        print(f"[DEBUG] QM9Enhanced 加载完成:")
        print(f"  数据文件: {self.processed_paths[0]}")
        print(f"  数据集大小: {len(self)}")
        if len(self) > 0:
            first_sample = self[0]
            print(f"  第一个样本x形状: {first_sample.x.shape}")

    @property
    def processed_file_names(self):
        # 返回您实际的文件名
        return ['data_v3.pt']

    def process(self):
        # 这个方法不会被调用，因为processed文件已存在
        # 但为了完整性，我们保留它
        print("注意: process()方法被调用，这通常不应该发生，因为processed文件应该已存在")
        pass

    def get_atomref(self, max_z=100):
        # 尝试使用原始QM9的atomref
        try:
            # 假设原始QM9数据在同级目录
            original_root = self.root.replace('_enhanced', '')
            if not os.path.exists(original_root):
                # 如果不存在，创建一个临时的原始QM9数据集来获取atomref
                import tempfile
                temp_dir = tempfile.mkdtemp()
                original_qm9 = QM9_geometric(root=temp_dir)
                atomref = original_qm9.atomref(self.label_idx)
            else:
                original_qm9 = QM9_geometric(root=original_root)
                atomref = original_qm9.atomref(self.label_idx)

            if atomref is None:
                return None
            if atomref.size(0) != max_z:
                tmp = torch.zeros(max_z).unsqueeze(1)
                idx = min(max_z, atomref.size(0))
                tmp[:idx] = atomref[:idx]
                return tmp
            return atomref
        except Exception as e:
            print(f"警告: 无法加载原始QM9的atomref: {e}")
            return None

    def _filter_label(self, data):
        if hasattr(data, 'y') and data.y is not None:
            if data.y.dim() > 1 and data.y.shape[1] > self.label_idx:
                data.y = data.y[:, self.label_idx: self.label_idx + 1]
            elif data.y.dim() == 1:
                data.y = data.y.unsqueeze(1)
        return data