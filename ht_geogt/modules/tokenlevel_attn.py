import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TokenLevelAttn(nn.Module):
    """Token级的自注意力层

    功能：对一条centerpath上的tokens做多次自注意力机制，并聚合成一个cptoken

    参数：
    token_dim: int - 输入token的维度
    cptoken_dim: int - 输出cptoken的维度
    num_heads: int - 注意力头数
    num_layers: int - 自注意力层数
    dropout: float - dropout率
    """

    def __init__(self, token_dim, cptoken_dim, num_heads=4, num_layers=10, dropout=0.1):
        super(TokenLevelAttn, self).__init__()

        assert token_dim % num_heads == 0

        self.token_dim = token_dim
        self.cptoken_dim = cptoken_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = token_dim // num_heads

        # 多层自注意力
        self.layers = nn.ModuleList([
            TokenAttnLayer(token_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影层
        self.out_proj = nn.Linear(token_dim, cptoken_dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(cptoken_dim)

    def forward(self, tokens):
        """
        参数：
        tokens: tensor of shape (batch_size, num_tokens, token_dim)
            一条centerpath上的所有tokens

        返回：
        cptoken: tensor of shape (batch_size, cptoken_dim)
            聚合后的cptoken
        """
        batch_size, num_tokens, _ = tokens.size()

        # 创建mask：标记非0的token位置（有效token）
        mask = ~torch.all(tokens == 0, dim=-1)  # [batch_size, num_tokens]
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_tokens]

        # 多层自注意力处理
        x = tokens
        for layer in self.layers:
            x = layer(x, attn_mask)

        # 只对非填充token进行平均
        valid_tokens = mask.unsqueeze(-1)  # [batch_size, num_tokens, 1]
        x = x * valid_tokens
        output = x.sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1)

        # 通过输出投影层和层归一化
        cptoken = self.out_proj(output)
        cptoken = self.layer_norm(cptoken)

        return cptoken


class TokenAttnLayer(nn.Module):
    """单层自注意力模块"""

    def __init__(self, token_dim, num_heads, dropout=0.1):
        super(TokenAttnLayer, self).__init__()

        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads

        # Q,K,V的线性变换层
        self.q_linear = nn.Linear(token_dim, token_dim)
        self.k_linear = nn.Linear(token_dim, token_dim)
        self.v_linear = nn.Linear(token_dim, token_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.ReLU(),
            nn.Linear(token_dim * 4, token_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        参数：
        x: tensor of shape (batch_size, num_tokens, token_dim)
        mask: tensor of shape (batch_size, 1, 1, num_tokens)

        返回：
        tensor of shape (batch_size, num_tokens, token_dim)
        """
        # 多头自注意力
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # 分头
        batch_size, num_tokens, _ = x.size()
        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合
        attn_output = torch.matmul(attn_weights, v)

        # 重组多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.token_dim)

        # 第一个残差连接和层归一化
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络，残差连接和层归一化
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x

