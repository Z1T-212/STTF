import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SEModule(nn.Module):

    def __init__(self, channels, reduction,dropout):
        super(SEModule, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc4 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.to_out = nn.Sequential(
            nn.Linear(2*channels, channels),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        module_input = x
        # b,t,21,192
        # 全局池化
        # b,t,1,1
        x1 = self.avg(x)
        # x2 = x.max((2, 3), keepdim=True)
        x2 = self.max(x)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x2 = self.fc3(x2)
        x2 = self.relu2(x2)
        x2 = self.fc4(x2)
        # (b,1,1,2t)
        x_out = x1+x2
        # x_out = self.to_out(x_out)
        x_out = self.sigmoid(x_out)
        print(x_out.shape)
        # b,t,1,1
        # 将权重乘回去
        return module_input * x_out


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y):
        # y(bt,21,192) x(bt,1,192)
        h = self.heads
        k = self.to_k(x)
        # b,t,192
        v = self.to_v(x)
        # b,t*21,192
        q = self.to_q(y)
        # bt,21,192->bt 3 21 64
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        # bt,3,1,64
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        # b,3,21,1
        dots = einsum('b a h d, b a c d -> b a h c', q, k) * self.scale
        # dim为2
        attn = dots.softmax(dim=-1)
        # b,3,t*21,64
        out = einsum('b a h c, b a c d -> b a h d', attn, v)
        # bt,21,192
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Fusion_space(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.layers_ttoimg = Cross_Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.layers_imgtot = Cross_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.heads = heads
        self.lay = nn.LayerNorm(dim)
        self.senet = SEModule(channels=80, reduction=8, dropout=dropout)
    def forward(self, x, y):
        # y(bt,21,192) x(bt,1,192)
        # mask = self._get_key_padding_mask(80, v)
        x = self.lay(x)
        y = self.lay(y)
        x_img_text = self.layers_imgtot(y, x)
        x_img_text = x + x_img_text
        # x_text_img = self.layers_ttoimg(x_img_text, x)
        # x_text_img = x + x_text_img
        x1 = self.lay(x_img_text)
        # b,t,1,192
        x1 = rearrange(x1, '(b n) c d -> b n c d', n=80)
        x = rearrange(x, '(b n) c d -> b n c d', n=80)
        # 加入权重回去
        x_weight = self.senet(x1)
        x_total = x + x_weight
        x_out = x_total.squeeze(2)
        return self.lay(x_out)


class Fusion_tem(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.layers_ttoimg = Cross_Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.layers_imgtot = Cross_Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.lay = nn.LayerNorm(dim)
    def forward(self, x, y):
        # y(b,21,192) x(b,t,192)
        x = self.lay(x)
        y = self.lay(y)
        x1 = self.layers_imgtot(y, x)
        x1 = x + x1
        x1 = self.lay(x1)

        x2 = self.layers_ttoimg(x, y)
        x2 = y + x2
        x2 = self.lay(x2)
        x_pool = x1.mean(1, keepdim=True)
        y_copy = x2.mean(1, keepdim=True)
        # b,192,192
        fusion_x = torch.mul(x_pool, y_copy)
        fusion_y = torch.mul(y_copy, x_pool)
        fusion_newx = fusion_x.mean(1, keepdim=True)
        fusion_newy = fusion_y.mean(1, keepdim=True)
        fusion_all = fusion_newx+fusion_newy
        out = fusion_all.squeeze(1)
        return out