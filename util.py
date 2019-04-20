from torchtext import data
from torchtext.datasets.trec import TREC
from torchtext.data import BucketIterator
from torchtext.vocab import Vectors
import re
import os

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def load_datasets(batch_size, pretrain):
    text = data.Field(tokenize=tokenize_line, lower=True, batch_first=True)
    label = data.Field(tokenize=tokenize_line, batch_first=True)
    train_dev_data, test_data = TREC.splits(text_field=text, label_field=label, root='data')
    train_data, dev_data = train_dev_data.split(split_ratio=0.9)
    if pretrain:
        print('use pretrain word vectors')
        cache = '.vector_cache'
        if not os.path.exists('.vector_cache'):
            os.mkdir('.vector_cache')
        vectors = Vectors(name='data/glove/glove.6B.300d.txt', cache=cache)
        text.build_vocab(train_data, dev_data, test_data, vectors=vectors)
    else:
        text.build_vocab(train_data, dev_data, test_data)
    label.build_vocab(train_data)

    train_iter, dev_iter, test_iter = BucketIterator.splits((train_data, dev_data, test_data), batch_sizes=(batch_size, batch_size, batch_size),
                                                            sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)
    data_iters = {'train_iter': train_iter, 'dev_iter': dev_iter, 'test_iter': test_iter}
    print('vocabulary size:', len(text.vocab))

    return data_iters, text.vocab, label.vocab


if __name__ == '__main__':
    load_datasets(50, False)
