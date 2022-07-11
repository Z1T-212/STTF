import os

import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn

###Data require
import sys

from models.MainModel import mainmodel

sys.path.append('../')
import argparse
from datasets.dataset_msrvtt import load_vocab_glove_matrix, VideoQADataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

###Model require

parser = argparse.ArgumentParser('net')
# ========================= Data Configs ==========================
parser.add_argument('--question_type', type=str, default='none', help='none | frameqa | count | action | transition')
# parser.add_argument('--vocab_path', type=str, default='./datasets/MSVD/word/MSVD_vocab.json')
# parser.add_argument('--glovept_path_train', type=str, default='./datasets/MSVD/word/MSVD_train_questions.pt')
# parser.add_argument('--glovept_path_val', type=str, default='./datasets/MSVD/word/MSVD_val_questions.pt')
# parser.add_argument('--glovept_path_test', type=str, default='./datasets/MSVD/word/MSVD_test_questions.pt')
# parser.add_argument('--visual_m_path', type=str, default='./datasets/MSVD/word/video/msvd_btup_f_obj10.hdf5', help='')
# parser.add_argument('--visual_a_path', type=str, default='./datasets/MSVD/word/video/msvd_res152_avgpool.hdf5', help='')
# MSRVTT
parser.add_argument('--vocab_path', type=str, default='./datasets/MSRVTT/word/MSRVTT_vocab.json')
parser.add_argument('--glovept_path_train', type=str, default='./datasets/MSRVTT/word/MSRVTT_train_questions.pt')
parser.add_argument('--glovept_path_val', type=str, default='./datasets/MSRVTT/word/MSRVTT_val_questions.pt')
parser.add_argument('--glovept_path_test', type=str, default='./datasets/MSRVTT/word/MSRVTT_test_questions.pt')
parser.add_argument('--visual_m_path', type=str, default='./datasets/MSRVTT/word/video/msrvtt_btup_f_obj10.hdf5', help='')
parser.add_argument('--visual_a_path', type=str, default='./datasets/MSRVTT/word/video/msrvtt_res152_avgpool.hdf5', help='')

parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--visual_dim', type=int, default=2048)
parser.add_argument('--dim_head', type=int, default=32)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--scale_dim', type=int, default=4)
parser.add_argument('--word_embed_dim', type=int, default=512)
parser.add_argument('--activation', type=str, default='gelu', help='relu | prelu | elu | gelu')
parser.add_argument('--vocab_size', type=int, default=4809)
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
args.vocab_size = len(vocab['question_token_to_idx'])
args.num_classes = len(vocab['answer_token_to_idx'])
# print(args.vocab_size)
# print(args.num_classes)

# 保存路径
args.savepath = os.path.join('Result/msrvtt', args.vocab_path.split('/')[2]) + '/'
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
# 交叉熵损失
criterion = nn.CrossEntropyLoss()
# 学习率衰减
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args.patience, factor=args.factor,
#                                                  verbose=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
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
# pre = torch.load( args.savepath + 'epoch.pt')
# model.load_state_dict(pre)
def batch_accuracy(output, batch_answer):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = output.detach().argmax(1)
    agreeing = (predicted == batch_answer)
    return agreeing

for epoch in range(args.maxepoch):
    model.train()
    total_acc, count = 0, 0
    total_loss, avg_loss = 0.0, 0.0
    for i, batch_data in enumerate(train_loader):
        # 参数梯度设置0
        model.zero_grad()
        *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
        output = model(*batch_input)
        print(output.shape)
        print(i)
        loss = criterion(output, batch_answer)
        loss.backward()
        total_loss += loss.detach()
        avg_loss = total_loss / (i + 1)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
        optimizer.step()
        aggreeings = batch_accuracy(output, batch_answer)

        total_acc += aggreeings.sum().item()  # 正确的个数
        count += batch_answer.size(0)  # 答案
        train_accuracy = total_acc / count
    with open(args.savepath + 'log_1119.txt', 'a') as out_file:
        out_file.write(
            "Epoch {} complete! Total Training loss: {}".format(epoch, total_loss / len(train_dataset)) + '\n')

    model.eval()
    eval_loss, total_acc, count = 0.0, 0.0, 0
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
            output = model(*batch_input)
            loss = criterion(output, batch_answer)
            eval_loss += loss.data.cpu()
            preds = output.detach().argmax(1)
            agreeings = (preds == batch_answer)
            preds = output.argmax(1)
            answer_vocab = vocab['answer_idx_to_token']
            total_acc += agreeings.float().sum().item()
            count += batch_answer.size(0)
            acc = total_acc / count
        with open(args.savepath + 'log_1119.txt', 'a') as out_file:
            out_file.write(
                "Epoch {} complete! Total val loss: {}".format(epoch, eval_loss / len(val_dataset)) + '\n'+
                '~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % acc + '\n'
            )
        with open(args.savepath + 'log_1119.txt', 'a') as out_file:
            out_file.write(
                "Epoch {} complete!, val correct is: {}".format(epoch, float(count) / len(val_dataset)) + '\n')
    scheduler.step(total_acc)

    model.eval()
    total_acc, count = 0.0, 0
    what_acc, who_acc, how_acc, when_acc, where_acc = 0., 0., 0., 0., 0.
    what_count, who_count, how_count, when_count, where_count = 0, 0, 0, 0, 0
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            *batch_input, batch_answer = [Variable(x.cuda(args.device_ids[0])) for x in batch_data]
            output = model(*batch_input)
            preds = output.detach().argmax(1)
            agreeings = (preds == batch_answer)
            what_idx = []
            who_idx = []
            how_idx = []
            when_idx = []
            where_idx = []
            key_word = batch_input[-4][:, 0].to('cpu')  # batch-based questions word
            for i, word in enumerate(key_word):
                word = int(word)
                if vocab['question_idx_to_token'][word] == 'what':
                    what_idx.append(i)
                elif vocab['question_idx_to_token'][word] == 'who':
                    who_idx.append(i)
                elif vocab['question_idx_to_token'][word] == 'how':
                    how_idx.append(i)
                elif vocab['question_idx_to_token'][word] == 'when':
                    when_idx.append(i)
                elif vocab['question_idx_to_token'][word] == 'where':
                    where_idx.append(i)
            preds = output.argmax(1)
            answer_vocab = vocab['answer_idx_to_token']
            total_acc += agreeings.float().sum().item()
            count += batch_answer.size(0)
            what_acc += agreeings.float()[what_idx].sum().item() if what_idx != [] else 0
            who_acc += agreeings.float()[who_idx].sum().item() if who_idx != [] else 0
            how_acc += agreeings.float()[how_idx].sum().item() if how_idx != [] else 0
            when_acc += agreeings.float()[when_idx].sum().item() if when_idx != [] else 0
            where_acc += agreeings.float()[where_idx].sum().item() if where_idx != [] else 0
            what_count += len(what_idx)
            who_count += len(who_idx)
            how_count += len(how_idx)
            when_count += len(when_idx)
            where_count += len(where_idx)
        acc = total_acc / count
        what_acc = what_acc / what_count
        who_acc = who_acc / who_count
        how_acc = how_acc / how_count
        when_acc = when_acc / when_count
        where_acc = where_acc / where_count

        with open(args.savepath + 'log_1119.txt', 'a') as out_file:
            out_file.write(
                "--- Epoch {} complete! test: " + '\n' +
                '~~~~~~ Test Accuracy: %.4f ~~~~~~~' % acc + '\n' +
                '~~~~~~ Test Who Accuracy: %.4f ~~~~~~' % who_acc + '\n' +
                '~~~~~~ Test what Accuracy: %.4f ~~~~~~' % what_acc + '\n' +
                '~~~~~~ Test how Accuracy: %.4f ~~~~~~' % how_acc + '\n' +
                '~~~~~~ Test when Accuracy: %.4f ~~~~~~' % when_acc + '\n' +
                '~~~~~~ Test where Accuracy: %.4f ~~~~~~' % where_acc + '\n'
            )

    ###savepath
    if acc > max_acc:
        max_acc = acc
        torch.save(model.state_dict(), args.savepath + 'epoch1119' + str(epoch) + '.pt')
        patience=0
    else:
        patience+=1
    if patience>=4*args.patience:
        break


