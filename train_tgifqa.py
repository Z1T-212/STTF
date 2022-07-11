import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn

###Data require
import sys

from models.MainModel import mainmodel

sys.path.append('../')
import argparse
from datasets.dataset_tgif import load_vocab_glove_matrix, VideoQADataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

###Model require

parser = argparse.ArgumentParser('net')
# ========================= Data Configs ==========================
parser.add_argument('--vocab_path', type=str, default='./datasets/TGIF/word/tgif-qa_action_vocab.json')
parser.add_argument('--glovept_path_train', type=str, default='./datasets/TGIF/word/tgif-qa_action_train_questions.pt')
parser.add_argument('--glovept_path_val', type=str, default='./datasets/TGIF/word/tgif-qa_action_val_questions.pt')
parser.add_argument('--glovept_path_test', type=str, default='./datasets/TGIF/word/tgif-qa_action_test_questions.pt')
parser.add_argument('--visual_m_path', type=str, default='./datasets/TGIF/word/video/tgif_btup_f_obj10.hdf5', help='')
parser.add_argument('--visual_a_path', type=str, default='./datasets/TGIF/word/video/res152_avgpool.hdf5', help='')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--question_type', type=str, default='action', help='frameqa | count | action | transition')

parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--visual_dim', type=int, default=2048)
parser.add_argument('--dim_head', type=int, default=32)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--scale_dim', type=int, default=4)
parser.add_argument('--word_embed_dim', type=int, default=512)
parser.add_argument('--activation', type=str, default='gelu', help='relu | prelu | elu | gelu')
parser.add_argument('--vocab_size', type=int)
parser.add_argument('--wordvec_dim', type=int, default=300)
parser.add_argument('--proj_l_drop', type=float, default=0.1)
parser.add_argument('--pos_flag', type=str, default='learned', help='learned | sincos')
parser.add_argument('--pos_dropout', type=float, default=0.0)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--attn_dropout', type=float, default=0.0)
parser.add_argument('--res_dropout', type=float, default=0.3)
parser.add_argument('--activ_dropout', type=float, default=0.1)
parser.add_argument('--num_layers', type=int, default=8)
# ========================= Model Configs ==========================
parser.add_argument('--drorate', type=float, default=0.1)
parser.add_argument('--device_ids', type=int, nargs='+', default=[0])
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--factor', type=float, default=0.5)
parser.add_argument('--maxepoch', type=int, default=40)
parser.add_argument('--num_clips', default=80, type=int)
parser.add_argument('--num_classes', type=int)
args = parser.parse_args()
# 加载问答以及字典
vocab, glove_matrix = load_vocab_glove_matrix(args.vocab_path, args.glovept_path_train)
if args.question_type in ['frameqa']:
    args.vocab_size = len(vocab['question_token_to_idx'])
    args.num_classes = len(vocab['answer_token_to_idx'])
elif args.question_type in ['count']:
    args.vocab_size = len(vocab['question_token_to_idx'])
else:
    args.vocab_size = len(vocab['question_answer_token_to_idx'])

# 保存路径
args.savepath = os.path.join('Result/tgifqa', args.vocab_path.split('/')[2]) + '/'
train_dataset = VideoQADataset(glovept_path=args.glovept_path_train,
                               visual_m_path=args.visual_m_path, question_type=args.question_type)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
val_dataset = VideoQADataset(glovept_path=args.glovept_path_val,
                             visual_m_path=args.visual_m_path, question_type=args.question_type)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
test_dataset = VideoQADataset(glovept_path=args.glovept_path_test,
                              visual_m_path=args.visual_m_path, question_type=args.question_type)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
print(len(train_dataset))

###Model
model = mainmodel(args).cuda(args.device_ids[0])
glove_matrix = glove_matrix.cuda(args.device_ids[0])
# 不需要参加反向传播求导
with torch.no_grad():
    model.textembedding.embed.weight.set_(glove_matrix)
torch.backends.cudnn.benchmark = True
# 学习率
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay,
                       amsgrad=False)
if args.question_type in ['frameqa']:
    criterion = nn.CrossEntropyLoss().cuda(args.device_ids[0])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=args.patience,factor=args.factor,verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
elif args.question_type in ['count']:
    criterion = nn.MSELoss().cuda(args.device_ids[0])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=args.patience,factor=args.factor,verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
else:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=args.patience,factor=args.factor,verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(args.device_ids[0])
    return torch.index_select(a, dim, order_index)

###train val test
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)
argsDict = args.__dict__
# 写入上面一堆参数
with open(args.savepath + 'params.txt', 'a') as f:
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
max_acc = 0.0
min_mae = float('Inf')
patience = 0
# pre = torch.load( args.savepath + 'epoch19.pt')
# model.load_state_dict(pre)
for epoch in range(args.maxepoch):
    model.train()
    total_loss = 0.0
    for i, batch_data in enumerate(train_loader):
        # 参数梯度设置0
        model.zero_grad()
        *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
        output = model(*batch_input)
        print(output.shape)
        print(i)
        if args.question_type in ['frameqa', 'count']:
            loss = criterion(output, batch_answer)
            loss.backward()
            total_loss += loss.data.cpu()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
            optimizer.step()
        else:
            expan_idx = np.concatenate(np.tile(np.arange(args.batch_size).reshape([args.batch_size, 1]), [1, 5])) * 5
            answers_agg = tile(batch_answer, 0, 5)
            loss = torch.max(torch.tensor(0.0).cuda(args.device_ids[0]),
                             1.0 + output - output[answers_agg + torch.from_numpy(expan_idx).cuda(args.device_ids[0])])
            loss = loss.sum()
            loss.backward()
            total_loss += loss.data.cpu()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
            optimizer.step()
    with open(args.savepath + 'log_1.txt', 'a') as out_file:
        out_file.write(
            "Epoch {} complete! Total Training loss: {}".format(epoch, total_loss / len(train_dataset)) + '\n')

    model.eval()
    correct = 0.0
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
            output = model(*batch_input)
            if args.question_type in ['frameqa']:
                _,pred = torch.max(output, 1)
                correct += torch.sum(pred == batch_answer)
            elif args.question_type in ['count']:
                output = (output + 0.5).long().clamp(min=1, max=10)
                correct += (output - batch_answer) ** 2
            else:
                preds = torch.argmax(output.view(batch_answer.shape[0], 5), dim=1)
                correct += torch.sum(preds == batch_answer)
        with open(args.savepath + 'log.txt', 'a') as out_file:
            out_file.write("Epoch {} complete!, val correct is: {}".format(epoch, float(correct)/len(val_dataset))+'\n')
    scheduler.step(correct)
    # scheduler.step()

    model.eval()
    correct = 0.0
    correct6 = 0.0
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
            output = model(*batch_input)
            if args.question_type in ['frameqa']:
                _, pred = torch.max(output, 1)
                correct += torch.sum(pred == batch_answer)
            elif args.question_type in ['count']:
                output6 = (output + 0.5).long().clamp(min=1, max=10)
                correct6 += (output6 - batch_answer) ** 2
            else:
                preds = torch.argmax(output.view(batch_answer.shape[0], 5), dim=1)
                print(preds.shape)
                print(batch_answer.shape)
                correct += torch.sum(preds == batch_answer)
            with open(args.savepath + 'log2.txt', 'a') as out_file:
                out_file.write(
                    "Epoch {} complete!, correct is {}, pre is: {}, test  is: {}".format(epoch, correct, preds ,batch_answer) + '\n')
        if args.question_type in ['count']:
            with open(args.savepath + 'log.txt', 'a') as out_file:
                out_file.write(
                    "Epoch {} complete!, test correct is: {}".format(epoch, float(correct6) / len(test_dataset)) + '\n')
        else:
            print(correct)
            print(len(test_loader))
            with open(args.savepath + 'log.txt', 'a') as out_file:
                out_file.write(
                    "Epoch {} complete!, test correct is: {}".format(epoch, float(correct) / len(test_dataset)) + '\n')
    ###savepath
    if args.question_type in ['frameqa']:
        if float(correct) / len(test_loader) > max_acc:
            max_acc = float(correct) / len(test_loader)
            torch.save(model.state_dict(), args.savepath + 'epoch' + str(epoch) + '.pt')
            patience = 0
        else:
            patience += 1
    elif args.question_type in ['count']:
        if float(correct6) / len(test_loader) < min_mae:
            min_mae = float(correct6) / len(test_loader)
            torch.save(model.state_dict(), args.savepath + 'epoch' + str(epoch) + '.pt')
            patience = 0
        else:
            patience += 1
    else:
        if float(correct) / len(test_loader) > max_acc:
            max_acc = float(correct) / len(test_loader)
            torch.save(model.state_dict(), args.savepath + 'epoch' + str(epoch) + '.pt')
            patience = 0
        else:
            patience += 1
    if patience >= 4 * args.patience:
        break