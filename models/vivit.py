import torch
from torch import nn
from einops import rearrange, repeat

from models.module import Fusion_space, Fusion_tem

from models.modules.TransformerEncoders import  TransformerEncoder_nopos


class ViViT(nn.Module):

    def __init__(self, num_frames, dim, depth, heads, visual_dim, pool, dim_head, dropout, activation, pos_flag):
        super().__init__()

        # 判别
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        # num_patch = (224/16)**2 = 14*14 = 196
        num_patches = (224 // 16) ** 2

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(visual_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        # 空间Transformer输入为（1，1，192）
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))

        self.space_fusion = Fusion_space(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)

        self.temporal_fusion = Fusion_tem(dim=dim, heads=heads, dim_head=dim_head,  dropout=dropout)

        self.space_transformer = TransformerEncoder_nopos(embed_dim=dim, num_heads=heads, attn_dropout=dropout, res_dropout=dropout, activ_dropout=dropout, activation=activation, num_layers=depth)
        # 时间Transformer输入为空间Transformer输出为（1，1，192）
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        # 输入时间Transformer
        self.temporal_transformer = TransformerEncoder_nopos(embed_dim=dim, num_heads=heads, attn_dropout=dropout, res_dropout=dropout, activ_dropout=dropout, activation=activation, num_layers=depth)

        self.dropout = nn.Dropout(dropout)
        self.pool = pool

    def forward(self, visual_m, visual_l, text):
        # x=（b，10, t 2048） y=(b,21 192)
        y = text
        x = self.to_patch_embedding(visual_m)
        # （b，t，196，192）
        b, t, n, _ = x.shape
        # 从（1，1，192）复制为（b,t,1,192）相当于复制了space_token的n和d维度（1，192）
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        # 按照维度2拼接 即n x变成（b,t,197,192）
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)
        # (b,10,t,2048)->(b10,t,2048)
        x = rearrange(x, 'b t n d -> (b t) n d')
        # 输入space
        x = rearrange(x, 'b t n -> t b n')
        x = self.space_transformer(x,None,None)
        # x[:, 0]所有行第一个数，
        # b80,1,2048
        x = rearrange(x, 'b t n -> t b n')
        x = x[:, 0]
        x = x.unsqueeze(1)
        # 空间融合部分
        # (b,1,21,192)
        y_space = y.unsqueeze(1)
        # （b,80,21,2048）
        y_space = y_space.repeat(1,t,1,1)
        # (b80,21,2048)
        y_space = repeat(y_space, 'b t n d -> (b t) n d')
        # 16张图片与16句相同的话融合，输出保持维度不变（b*t,192）输出为（b*t,192）输出为（b,t,192）b,80,2048
        fusion1 = self.space_fusion(x,y_space)
        # b,81,2048
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        # 连接为（b,17,192）
        fusion1 = torch.cat((cls_temporal_tokens, fusion1), dim=1)
        # temporal
        fusion1 = rearrange(fusion1, 'b t n -> t b n')
        x_t = self.temporal_transformer(fusion1, None, visual_l)
        x_t = rearrange(x_t, 'b t n -> t b n')
        x_t = x_t[:, 0]
        # (b,1,192) (b,1,192)
        x_t = x_t.unsqueeze(1)
        # (b,80,192) (b,21,192)
        fusion2 = self.temporal_fusion(x_t, y)
        return fusion2

