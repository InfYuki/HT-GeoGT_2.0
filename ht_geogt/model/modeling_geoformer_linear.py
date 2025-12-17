from typing import Optional

from typing import Optional

import ase
import torch
from einops import rearrange, repeat
from torch import nn
from transformers import PreTrainedModel

from ht_geogt.datasets import QM9
from ht_geogt.datasets import QM9Enhanced
from ht_geogt.model import modeling_priors
from ht_geogt.model.configuration_geoformer import GeoformerConfig
from ht_geogt.model.modeling_geoformer_layers import (CosineCutoff,
                                                      ExpNormalSmearing,
                                                      VecLayerNorm,
                                                      act_class_mapping)

from typing import Optional

import ase
import torch
from einops import rearrange, repeat
from torch import nn
from transformers import PreTrainedModel

from ht_geogt.datasets import QM9
from ht_geogt.datasets import QM9Enhanced
from ht_geogt.model import modeling_priors
from ht_geogt.model.configuration_geoformer import GeoformerConfig
from ht_geogt.model.modeling_geoformer_layers import (CosineCutoff,
                                                      ExpNormalSmearing,
                                                      VecLayerNorm,
                                                      act_class_mapping)


class GeoformerMultiHeadAttention(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerMultiHeadAttention, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not (
                self.head_dim * config.num_attention_heads == self.embedding_dim
        ):
            raise AssertionError(
                "The embedding_dim must be divisible by num_heads."
            )

        self.act = act_class_mapping[config.activation_function]()
        self.cutoff = CosineCutoff(config.cutoff)

        self.dropout_module = nn.Dropout(
            p=config.attention_dropout, inplace=False
        )

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dk_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.du_update_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.du_norm = VecLayerNorm(
            self.embedding_dim, trainable=False, norm_type=config.norm_type
        )
        self.dihedral_proj = nn.Linear(
            self.embedding_dim, 2 * self.embedding_dim, bias=False
        )
        self.edge_attr_update = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.du_update_proj.weight)
        self.du_update_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dihedral_proj.weight)
        nn.init.xavier_uniform_(self.edge_attr_update.weight)
        self.edge_attr_update.bias.data.fill_(0.0)

    def forward(
            self,
            x: torch.Tensor,  # (B, N, F)
            vec: Optional[torch.Tensor],  # (B, N, N, 3)
            dist: Optional[torch.Tensor],  # (B, N, N)
            edge_attr: Optional[torch.Tensor],  # (B, N, N, F)
            key_padding_mask: Optional[torch.Tensor] = None,  # (B, N, N)
            center_nodes_mask: Optional[torch.Tensor] = None,  # (B, N) - 中心节点的掩码
            **kwargs,
    ):
        # 投影所有节点的特征
        k = rearrange(
            self.k_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        v = rearrange(
            self.v_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)

        # 如果没有提供中心节点掩码，默认使用全部节点
        if center_nodes_mask is None:
            center_nodes_mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)  # (B, N)

        # 只对中心节点计算查询向量
        # 首先创建一个掩码版本的x
        center_x = x.clone()
        center_x = center_x * center_nodes_mask.unsqueeze(-1)  # (B, N, F)

        # 只为中心节点计算查询向量
        q = rearrange(
            self.q_proj(center_x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)

        # 边缘特征投影
        dk = rearrange(
            self.act(self.dk_proj(edge_attr)),
            "b n m (h d) -> (b h) n m d",
            h=self.num_heads,
        )  # (BH, N, N, D)

        # 计算注意力权重，但只考虑中心节点的查询
        # 创建注意力掩码以仅考虑中心节点
        center_attn_mask = center_nodes_mask.unsqueeze(-1)  # (B, N, 1)

        # 计算注意力权重
        attn_weights = ((q.unsqueeze(-2) * k.unsqueeze(-3)) * dk).sum(
            dim=-1
        )  # (BH, N, N)

        # 应用中心节点掩码
        attn_weights = rearrange(
            attn_weights, "(b h) n m -> b h n m", h=self.num_heads
        )
        attn_weights = attn_weights * rearrange(center_nodes_mask, "b n -> b () n ()")  # 只保留中心节点的查询

        # 应用键填充掩码
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                rearrange(key_padding_mask, "b n m -> b () n m"),
                0.0,
            )

        attn_weights = rearrange(attn_weights, "b h n m -> (b h) n m")

        # 应用距离截断
        attn_scale = repeat(
            self.cutoff(dist), "b n m -> b h n m", h=self.num_heads
        )  # (B, H, N, N)
        attn_scale = rearrange(
            attn_scale, "b h n m -> (b h) n m", h=self.num_heads
        )  # (BH, N, N)

        # 计算最终的注意力概率
        attn_probs = self.act(attn_weights) * attn_scale  # (BH, N, N)

        # 计算每个节点的注意力加权值
        attn_per_nodes = attn_probs.unsqueeze(-1) * v.unsqueeze(
            -3
        )  # (BH, N, N, D)
        attn_per_nodes = rearrange(
            attn_per_nodes, "(b h) n m d -> b n m (h d)", h=self.num_heads
        )  # (B, N, N, F)

        # 对每个节点的注意力值进行汇总
        attn = attn_per_nodes.sum(dim=2)  # (B, N, F)

        # 计算方向更新
        du = (
                self.du_update_proj(attn_per_nodes)
                .masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
                .unsqueeze(-2)
                * vec.unsqueeze(-1)
        ).sum(
            dim=-3
        )  # (B, N, 3, F)
        du = self.du_norm(du)  # (B, N, 3, F)

        # 计算二面角投影
        ws, wt = torch.split(
            self.dihedral_proj(du), self.embedding_dim, dim=-1
        )  # (B, N, 3, F)

        # 仅为中心节点计算IPE
        center_ws = ws * center_nodes_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 3, F)
        ipe = (wt.unsqueeze(1) * center_ws.unsqueeze(2)).sum(dim=-2)  # (B, N, N, F)
        ipe = self.act(self.edge_attr_update(edge_attr)) * ipe  # (B, N, N, F)

        return attn, ipe


class GeoformerAttnBlock(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerAttnBlock, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.dropout_module = nn.Dropout(p=config.dropout, inplace=False)

        self.act = act_class_mapping[config.activation_function]()

        self.self_attn = GeoformerMultiHeadAttention(config)

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, config.ffn_embedding_dim),
            self.act,
            nn.Dropout(p=config.activation_dropout, inplace=False),
            nn.Linear(config.ffn_embedding_dim, self.embedding_dim),
        )

        self.attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        nn.init.xavier_uniform_(self.ffn[0].weight)
        self.ffn[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.ffn[3].weight)
        self.ffn[3].bias.data.fill_(0.0)
        self.attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()

    def forward(
            self,
            x: torch.Tensor,  # (B, N, F)
            vec: torch.Tensor,  # (B, N, N, 3)
            dist: torch.Tensor,  # (B, N, N)
            edge_attr: torch.Tensor,  # (B, N, N, ?)
            key_padding_mask: Optional[torch.Tensor],  # (B, N, N)
            center_nodes_mask: Optional[torch.Tensor] = None,  # (B, N) - 中心节点的掩码
            **kwargs,
    ):
        # attention
        dx, dedge_attr = x, edge_attr
        x, edge_attr = self.self_attn(
            x=x,
            vec=vec,
            dist=dist,
            edge_attr=edge_attr,
            key_padding_mask=key_padding_mask,
            center_nodes_mask=center_nodes_mask,
        )

        x = self.dropout_module(x)
        x = x + dx
        x = self.attn_layer_norm(x)

        # ipe update
        edge_attr = edge_attr + dedge_attr

        # ffn
        dx = x
        x = self.ffn(x)
        x = self.dropout_module(x)
        x = x + dx

        x = self.final_layer_norm(x)

        return x, edge_attr


class GeoformerEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerEncoder, self).__init__(*args, **kwargs)

        self.pad_token_id = config.pad_token_id
        self.embedding_dim = config.embedding_dim
        self.cutoff = config.cutoff

        # 添加中心节点比例参数，默认为0.2（20%的节点作为中心节点）
        self.center_node_ratio = getattr(config, 'center_node_ratio', 0.2)

        # 添加一个投影层将输入特征映射到embedding_dim
        self.feature_proj = nn.Linear(11, self.embedding_dim)  # 假设输入特征是11维

        self.distance_expansion = ExpNormalSmearing(
            cutoff=config.cutoff,
            num_rbf=config.num_rbf,
            trainable=config.rbf_trainable,
        )
        self.dist_proj = nn.Linear(config.num_rbf, self.embedding_dim)
        self.act = act_class_mapping[config.activation_function]()

        self.layers = nn.ModuleList(
            [GeoformerAttnBlock(config) for _ in range(config.num_layers)]
        )

        self.x_in_layernorm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.feature_proj.weight)
        self.feature_proj.bias.data.fill_(0.0)
        self.distance_expansion.reset_parameters()
        nn.init.xavier_uniform_(self.dist_proj.weight)
        self.dist_proj.bias.data.fill_(0.0)
        for layer in self.layers:
            layer.reset_parameters()
        self.x_in_layernorm.reset_parameters()

    def _select_center_nodes(self, z, padding_mask):
        """选择中心节点"""
        B, N = z.shape
        # 创建一个随机掩码，排除padding节点
        valid_nodes = ~padding_mask  # (B, N)

        # 计算每个batch中有效节点的数量
        valid_counts = valid_nodes.sum(dim=1)  # (B,)

        # 初始化中心节点掩码
        center_nodes_mask = torch.zeros_like(padding_mask, dtype=torch.bool)

        for b in range(B):
            valid_count = valid_counts[b].item()
            if valid_count > 0:
                # 计算中心节点数量，至少为1
                num_centers = max(1, int(valid_count * self.center_node_ratio))

                # 获取有效节点的索引
                valid_indices = torch.nonzero(valid_nodes[b]).squeeze(-1)

                # 随机选择中心节点
                if len(valid_indices) > 0:  # 确保有有效节点
                    perm = torch.randperm(len(valid_indices), device=z.device)
                    selected_indices = valid_indices[perm[:num_centers]]
                    center_nodes_mask[b, selected_indices] = True

        return center_nodes_mask

    def forward(
            self,
            x: torch.Tensor,  # (B, N, F)
            z: torch.Tensor,  # (B, N)
            pos: torch.Tensor,  # (B, N, 3)
            **kwargs,
    ):
        B, N, *_ = z.shape
        # generate mask
        padding_mask = z == self.pad_token_id  # (B, N)
        pos_mask = ~(
                padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
        )  # (B, N, N)
        dist = torch.norm(
            pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1
        )  # (B, N, N)
        loop_mask = torch.eye(N, dtype=torch.bool, device=dist.device)
        loop_mask = repeat(loop_mask, "n m -> b n m", b=B)  # (B, N, N)
        dist = dist.masked_fill(loop_mask, 0.0)  # (B, N, N)
        adj_mask = (dist < self.cutoff) & pos_mask  # (B, N, N)
        loop_adj_mask = ~loop_mask & adj_mask  # (B, N, N)

        vec = (pos.unsqueeze(1) - pos.unsqueeze(2)) / (
                dist.unsqueeze(-1) + 1e-8
        )  # (B, N, N, 3)
        vec = vec.masked_fill(
            ~loop_adj_mask.unsqueeze(-1), 0.0
        )  # (B, N, N, 3)

        key_padding_mask = (
            (~adj_mask)
            .masked_fill(padding_mask.unsqueeze(-1), False)
            .masked_fill(padding_mask.unsqueeze(-2), True)
        )

        # 选择中心节点
        center_nodes_mask = self._select_center_nodes(z, padding_mask)

        # 将11维特征投影到embedding_dim维
        x = self.feature_proj(x)  # (B, N, embedding_dim)

        x = self.x_in_layernorm(x)
        edge_attr = self.distance_expansion(dist)  # (B, N, N, num_rbf)
        edge_attr = self.act(self.dist_proj(edge_attr))  # (B, N, N, F)
        edge_attr = edge_attr.masked_fill(
            ~adj_mask.unsqueeze(-1), 0.0
        )  # (B, N, N, F)

        for layer in self.layers:
            x, edge_attr = layer(
                x=x,
                vec=vec,
                dist=dist,
                edge_attr=edge_attr,
                key_padding_mask=key_padding_mask,
                center_nodes_mask=center_nodes_mask,
            )

        return x, edge_attr


class GeoformerScalarDecoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerScalarDecoder, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.act = act_class_mapping[config.activation_function]()
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            self.act,
            nn.Linear(self.embedding_dim // 2, self.num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.classifier[2].weight)
        self.classifier[2].bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        return self.classifier(x) + edge_attr.sum() * 0


class GeoformerDipoleMomentDecoder(GeoformerScalarDecoder):
    def __init__(self, config, *args, **kwargs):
        super(GeoformerDipoleMomentDecoder, self).__init__(
            config, *args, **kwargs
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        x = self.classifier(x) + edge_attr.sum() * 0  # (B, N, 1)

        # Get center of mass.
        z = kwargs["z"]  # (B, N)
        pos = kwargs["pos"]  # (B, N, 3)
        padding_mask = kwargs["padding_mask"]  # (B, N)
        mass = (
            self.atomic_mass[z].masked_fill(padding_mask, 0.0).unsqueeze(-1)
        )  # (B, N, 1)
        c = torch.sum(mass * pos, dim=-2) / torch.sum(mass, dim=-2)
        x = x * (pos - c.unsqueeze(-2))
        return x  # (B, N, 3)


class GeoformerElectronicSpatialExtentDecoder(GeoformerScalarDecoder):
    def __init__(self, config, *args, **kwargs):
        super(GeoformerElectronicSpatialExtentDecoder, self).__init__(
            config, *args, **kwargs
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        x = self.classifier(x) + edge_attr.sum() * 0  # (B, N, 1)

        # Get center of mass.
        z = kwargs["z"]  # (B, N)
        pos = kwargs["pos"]  # (B, N, 3)
        padding_mask = kwargs["padding_mask"]  # (B, N)
        mass = (
            self.atomic_mass[z].masked_fill(padding_mask, 0.0).unsqueeze(-1)
        )  # (B, N, 1)
        c = torch.sum(mass * pos, dim=-2) / torch.sum(mass, dim=-2)
        x = torch.norm(pos - c.unsqueeze(-2), dim=-1, keepdim=True) ** 2 * x
        return x  # (B, N, 1)


class GeoformerModel(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super(GeoformerModel, self).__init__(config, *inputs, **kwargs)

        #self.geo_encoder = GeoformerEncoder(config)
        # 创建两个独立的encoder流
        self.geo_encoder_stream1 = GeoformerEncoder(config)
        self.geo_encoder_stream2 = GeoformerEncoder(config)

        if config.decoder_type == "scalar":
            self.geo_decoder = GeoformerScalarDecoder(config)
        elif config.decoder_type == "dipole_moment":
            self.geo_decoder = GeoformerDipoleMomentDecoder(config)
        elif config.decoder_type == "electronic_spatial_extent":
            self.geo_decoder = GeoformerElectronicSpatialExtentDecoder(config)
        else:
            raise ValueError(f"Unknown decoder type: {config.decoder_type}")

        self.post_init()

    def init_weights(self):
        #self.geo_encoder.reset_parameters()
        self.geo_encoder_stream1.reset_parameters()
        self.geo_encoder_stream2.reset_parameters()

        self.geo_decoder.reset_parameters()


class GeoformerForEnergyRegression(GeoformerModel):
    def __init__(self, config, *inputs, **kwargs):
        super(GeoformerForEnergyRegression, self).__init__(
            config, *inputs, **kwargs
        )

        self.config = config
        self.aggr = config.aggr
        self.pad_token_id = config.pad_token_id
        self.prior_model = self._register_prior_model()
        mean = torch.scalar_tensor(0) if config.mean is None else config.mean
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if config.std is None else config.std
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std).float()
        self.register_buffer("std", std)

    def _register_prior_model(self):
        prior_model = None
        if self.config.prior_model is not None:
            assert hasattr(modeling_priors, self.config.prior_model), (
                f"Unknown prior model {self.config.prior_model}. "
                f"Available models are {', '.join(modeling_priors.__all__)}"
            )
            # initialize the prior model
            prior_model = getattr(modeling_priors, self.config.prior_model)(
                utils=QM9Enhanced(
                    root=self.config.dataset_root,
                    dataset_arg=self.config.dataset_arg,
                )
            )
        return prior_model

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F) - 现在直接接收节点特征  (B, N, 22) - 包含两个11维向量
        z: torch.Tensor,  # (B, N)
        pos: torch.Tensor,  # (B, N, 3)
        **kwargs,
    ):
        #print('000')
        # 添加调试信息
        #print(f"[DEBUG] GeoformerForEnergyRegression.forward:")
        #print(f"  x.shape: {x.shape}")
        #print(f"  z.shape: {z.shape}")
        #print(f"  pos.shape: {pos.shape}")
        #print(f"  x前几个值: {x[0, 0, :5] if x.dim() == 3 and x.shape[0] > 0 and x.shape[1] > 0 else 'N/A'}")

        x1 = x[:, :, :11]   # (B, N, 11) - 前11维给第一个流
        x2 = x[:, :, 11:]   # (B, N, 11) - 后11维给第二个流
        #x, edge_attr = self.geo_encoder(x=x, z=z, pos=pos)
        # 两个流都接收相同的输入
        x1, edge_attr1 = self.geo_encoder_stream1(x=x1, z=z, pos=pos)
        x2, edge_attr2 = self.geo_encoder_stream2(x=x2, z=z, pos=pos)

        # 简单的融合策略：对两个流的输出进行相加
        x_fused = x1 + x2
        edge_attr_fused = edge_attr1 + edge_attr2

        padding_mask = z == self.pad_token_id  # (B, N)

        # (B, N, 1) or (B, N, 3)
        x = self.geo_decoder(
            x=x_fused, edge_attr=edge_attr_fused, z=z, pos=pos, padding_mask=padding_mask
        )

        logits = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (B, N, 1)

        if self.std is not None:
            logits = logits * self.std

        logits = (
            self.prior_model(logits, z)
            if self.prior_model is not None
            else logits
        )

        if self.aggr == "sum":
            logits = logits.sum(dim=1)  # (B, 1)
        elif self.aggr == "mean":
            logits = logits.sum(dim=1) / (~padding_mask).sum(dim=-1).unsqueeze(
                -1
            )  # (B, 1)
        else:
            NotImplementedError(f"Unknown aggregation method: {self.aggr}")

        if self.config.decoder_type == "dipole_moment":
            logits = torch.norm(logits, dim=-1, keepdim=True)

        if self.mean is not None:
            logits = logits + self.mean

        return logits


def create_linear_model(config) -> GeoformerForEnergyRegression:
    model_config = GeoformerConfig(
        max_z=config.max_z,
        embedding_dim=config.embedding_dim,
        ffn_embedding_dim=config.ffn_embedding_dim,
        num_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        cutoff=config.cutoff,
        num_rbf=config.num_rbf,
        rbf_trainable=config.trainable_rbf,
        norm_type=config.norm_type,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        activation_dropout=config.activation_dropout,
        activation_function=config.activation_function,
        decoder_type=config.decoder_type,
        aggr=config.aggr,
        dataset_root=config.dataset_root,
        dataset_arg=config.dataset_arg,
        mean=config.mean,
        std=config.std,
        prior_model=config.prior_model,
        num_classes=config.num_classes,
        pad_token_id=config.pad_token_id,
    )

    return GeoformerForEnergyRegression(config=model_config)
