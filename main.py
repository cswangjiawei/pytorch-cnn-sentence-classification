import argparse
from util import load_datasets
from model import SentClassif
import time
import torch.nn as nn
from torch.optim import Adadelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train import train_model, evaluate
import torch
import numpy as np
import random
from tensorboardX import SummaryWriter
import os


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification')
    parser.add_argument('--word_dim', default=300)
    parser.add_argument('--out_dim', default=100, help='The out channel of cnn')
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--savedir', default='data/model/')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--early_stop', default=10)
    parser.add_argument('--least_epoch', default=15)
    parser.add_argument('--optimizer', choices=['Adadelta', 'Adam', 'Sgd'], default='Adadelta')
    parser.add_argument('--lr', default=1.)
    parser.add_argument('--filters', default=[3, 4, 5])
    parser.add_argument('--pretrain', action='store_true')

    args = parser.parse_args()

    data_iters, text_vocab, label_vocab = load_datasets(args.batch_size, args.pretrain)
    label_vocab_size = len(label_vocab) - 2
    model = SentClassif(args.word_dim, args.out_dim, label_vocab_size, text_vocab, args.dropout, args.filters)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adadelta(model.parameters(), lr=args.lr)

    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.size())
    #     print('*'*100)

    best_acc = -1
    patience_count = 0
    model_name = args.savedir + 'best.pt'
    writer = SummaryWriter('log')

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    train_begin = time.time()

    print('train begin with use pretrain wordvectors :', args.pretrain, time.asctime(time.localtime(time.time())))
    print('*'*100)
    print()

    for epoch in range(args.epochs):
        epoch_begin = time.time()

        print('train {}/{} epoch starting:'.format(epoch+1, args.epochs))
        loss = train_model(data_iters['train_iter'], model, criterion, optimizer)
        acc = evaluate(data_iters['dev_iter'], model)
        print('acc:', acc)
        writer.add_scalar('dev_acc', acc, epoch)
        if acc > best_acc:
            patience_count = 0
            best_acc = acc
            print('new best_acc:', best_acc)
            torch.save(model.state_dict(), model_name)
        else:
            patience_count += 1

        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        print('train {}th cost {}s'.format(epoch+1, cost_time))
        print('-'*100)
        print()
        writer.add_scalar('train_loss', loss, epoch)
        if patience_count > args.early_stop and epoch + 1 > args.least_epoch:
            break

    train_end = time.time()

    train_cost = train_end - train_begin
    hour = int(train_cost / 3600)
    min = int((train_cost % 3600)/60)
    second = train_cost % 3600 % 60
    print('train total cost {}h {}m {}s'.format(hour, min, second))
    model.load_state_dict(torch.load(model_name))
    test_acc = evaluate(data_iters['test_iter'], model)
    print('The test accuracy:', test_acc)
