import logging
import os
import numpy as np
import json
import pickle
import math
import h5py
import torch
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab_glove_matrix(vocab_path, glovept_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    with open(glovept_path, 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = torch.from_numpy(obj['glove']).type(torch.FloatTensor)
    return vocab, glove_matrix


def video_pad(sequence, obj_max_num, max_length):
    sequence = np.array(sequence)
    sequence_shape = np.array(sequence).shape
    current_length = sequence_shape[0]
    current_num = sequence_shape[1]
    pad = np.zeros((max_length, obj_max_num, sequence_shape[2]),dtype=np.float32)
    num_padding = max_length - current_length
    num_obj_padding = obj_max_num - current_num
    if num_padding <= 0:
        pad = sequence[:max_length]
    else:
        pad[:current_length, :current_num] = sequence
    return pad



class VideoQADataset(Dataset):
    def __init__(self, glovept_path, visual_m_path, question_type):
        self.glovept_path = glovept_path
        self.visual_m_path = visual_m_path
        self.question_type = question_type
        #load glovefile
        with open(glovept_path, 'rb') as f:
            obj = pickle.load(f)
            self.questions = torch.from_numpy(obj['questions']).type(torch.LongTensor)
            self.questions_len = torch.from_numpy(obj['questions_len']).type(torch.LongTensor)
            self.question_id = obj['question_id']
            self.video_ids = obj['video_ids']
            self.video_names = obj['video_names']
            if self.question_type in ['count']:
                self.answers = torch.from_numpy(np.array(obj['answers'])).type(torch.FloatTensor).unsqueeze(-1)
            else:
                self.answers = torch.from_numpy(np.array(obj['answers'])).type(torch.LongTensor)
            if self.question_type not in ['none', 'frameqa', 'count']:
                self.ans_candidates = torch.from_numpy(np.array(obj['ans_candidates'])).type(torch.LongTensor)
                self.ans_candidates_len = torch.from_numpy(np.array(obj['ans_candidates_len'])).type(torch.LongTensor)


    def __getitem__(self, idx):
        video_ids = self.video_ids[idx].item()
        # app_index =  self.app_feat_id_to_index[str(video_ids)]
        if self.question_type not in ['none']:
            ids = str(video_ids)
        else:
            ids = 'video' + str(video_ids)
        with h5py.File(self.visual_m_path, 'r') as f_app:
            appearance_feat = np.array(f_app['image_features'][ids])
        appearance_len = appearance_feat.shape[0]
        appearance_feat = video_pad(appearance_feat, 10, 80)
        appearance_feat = torch.from_numpy(appearance_feat)
        ans_candidates = torch.zeros(5).long()
        ans_candidates_len = torch.zeros(5).long()
        question = self.questions[idx]
        question_len = self.questions_len[idx]
        answer = self.answers[idx]
        if self.question_type not in ['none', 'frameqa', 'count']:
            ans_candidates = self.ans_candidates[idx]
            ans_candidates_len = self.ans_candidates_len[idx]
        return appearance_feat,appearance_len,question,question_len,ans_candidates, ans_candidates_len,answer

    def __len__(self):
        return self.questions.shape[0]

