import argparse
import numpy as np
import os

import msrvtt_qa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='msrvtt-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='MSRVTT_val_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='MSRVTT_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'msrvtt-qa':
        args.annotation_file = 'val_qa.json'.format(args.mode)
        # check if data folder exists
        msrvtt_qa.process_questions(args)