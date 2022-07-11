
import torch
import torch.nn as nn
from .modules.TransformerEncoders import *


class textembedding(nn.Module):
    def __init__(self, dim, embed_dim, activation, vocab_size, wordvec_dim, proj_l_drop, pos_flag ,pos_dropout,num_heads,attn_dropout,res_dropout,activ_dropout,num_layers):
        super(textembedding, self).__init__()
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()

        self.embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, embed_dim),
                        self.activ,
                        nn.Dropout(p=proj_l_drop),
                        )
        # self.Transformer= Transformer(dim=512, depth=6, heads=3, dim_head=64, mlp_dim=768, dropout=0.)
        self.t_l = nn.Linear(embed_dim, dim)
        self.sentence_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.TransformerEncoder_text = TransformerEncoder(embed_dim, pos_flag, pos_dropout, num_heads, attn_dropout,
                                                          res_dropout, activ_dropout, activation, num_layers)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)        

    def forward(self, text, text_len):
        """
        Args:
            text: [Tensor] (batch_size, max_text_length)
            text_len: [Tensor] (batch_size,)
        return:
            text_embedding: [Tensor] (batch_size, image_num, embed_dim)(CLS代表整个句子的融合特征)
        """
        # (2, 21, 300)
        text_embedding = self.embed(text)
        # (21 2 512)
        text_embedding = self.proj_l(text_embedding).permute(1, 0, 2)
        text_embedding = self.TransformerEncoder_text(text_embedding, None, text_len)
        # (2 21 512)
        text_embedding = text_embedding.permute(1, 0, 2)
        # (2 21 192)
        text_embedding = self.t_l(text_embedding)

        return text_embedding