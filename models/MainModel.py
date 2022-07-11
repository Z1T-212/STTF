import numpy as np
import math
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from models.Embedding import textembedding
from models.OutLayers import OutOpenEnded, OutMultiChoices, OutCount
from models.vivit import ViViT


class mainmodel(nn.Module):
    def __init__(self, args):
        super(mainmodel, self).__init__()
        self.question_type = args.question_type
        self.vivit = ViViT(num_frames=args.num_clips, dim=args.dim, depth=args.depth,
                      heads=args.heads, visual_dim=args.visual_dim, pool='cls', dim_head=args.dim_head,
                      dropout=args.dropout, activation=args.activation, pos_flag=args.pos_flag)
        self.textembedding = textembedding(dim=args.dim, embed_dim=args.word_embed_dim, activation=args.activation, vocab_size=args.vocab_size, wordvec_dim=args.wordvec_dim, proj_l_drop=args.proj_l_drop,
                      pos_flag=args.pos_flag, pos_dropout=args.pos_dropout, num_heads=args.num_heads, attn_dropout=args.attn_dropout, res_dropout=args.res_dropout,
                      activ_dropout=args.activ_dropout, num_layers=args.num_layers)
        if self.question_type in ['none', 'frameqa']:
            self.outlayer = OutOpenEnded(args.dim, args.num_classes, args.dropout, args.activation)
        elif self.question_type in ['count']:
            self.outlayer = OutCount(args.dim, args.drorate, args.activation)
        else:
            self.outlayer = OutMultiChoices(args.dim, args.drorate, args.activation)

    def forward(self, visual_m, visual_s, question, question_len, answers, answers_len):
        """
        Args:
            visual_m: [Tensor] (batch_size, levels, 2048)
            visual_m: [Tensor] (batch_size, levels, 16, 2048)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [None or Tensor], if a tensor shape is (batch_size,)
            answers: [Tensor] (batch_size, 5, max_answers_length)
            answers_len: [Tensor] (batch_size, 5)
        return:
            question_embedding_v: [Tensor] (max_question_length, batch_size, embed_dim)
            visual_embedding_qu: [Tensor] (16, batch_size, embed_dim)
            question_embedding: [Tensor] (max_question_length, batch_size, embed_dim)
            question_len: [None or Tensor], if a tensor shape is (batch_size,)
        """
        if self.question_type in ['none', 'frameqa', 'count']:
            text = self.textembedding(question, question_len)
            vivit = self.vivit(visual_m, visual_s, text)
            out = self.outlayer(vivit)
        else:
            output_answer_list = []
            for i in range(5):
                answer_embedding = self.textembedding(answers[:, i, :], answers_len[:, i])
                text_embadding = self.textembedding(question, question_len)
                total_embadding = torch.cat((text_embadding, answer_embedding), dim=1)
                output = self.vivit(visual_m, visual_s, total_embadding)
                output_answer_list.append(output)
            # 铺平为一维
            out = torch.stack(output_answer_list, dim=2)
            out = rearrange(out, 'b t n -> (b n) t')
            out = self.outlayer(out)
        return out