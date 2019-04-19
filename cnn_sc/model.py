import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class SentClassif(nn.Module):
    def __init__(self, word_dim, out_dim,  label_size, vocab, dropout, filters):
        super(SentClassif, self).__init__()

        self.embedding = nn.Embedding(len(vocab), word_dim)
        self.cnn_list = nn.ModuleList()
        if vocab.vectors is not None:
            self.embedding.weight.data.copy_(vocab.vectors)
        else:
            self.embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(len(vocab), word_dim)))
        for kernel_size in filters:
            self.cnn_list.append(nn.Conv1d(word_dim, out_dim, kernel_size, padding=int((kernel_size-1)/2)))

        self.out2tag = nn.Linear(out_dim*3, label_size)

        self.drop = nn.Dropout(dropout)

    def random_embedding(self, vocab_size, word_dim):
        pretrain_emb = np.empty([vocab_size, word_dim])
        scale = np.sqrt(3.0 / word_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, word_dim])
        return pretrain_emb

    def forward(self, word_input):
        batch_size = word_input.size(0)
        word_represent = self.embedding(word_input)
        word_represent = word_represent.transpose(1, 2)
        out = []
        for cnn in self.cnn_list:
            cnn_out = cnn(word_represent)
            cnn_out = F.relu(cnn_out)
            cnn_out = F.max_pool1d(cnn_out, cnn_out.size(2)).view(batch_size, -1)
            out.append(cnn_out)

        cat_out = torch.cat(out, 1)

        out = self.drop(cat_out)
        tag_socre = self.out2tag(out)

        return tag_socre
