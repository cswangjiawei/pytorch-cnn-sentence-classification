def make_tsv():
    with open('data/rt-polaritydata/rt-polarity.tsv', 'w', encoding='utf-8') as f:
        lines = open('data/rt-polaritydata/rt-polarity.pos', 'r', encoding='utf-8').readlines()
        for line in lines:
            f.write(line.strip())
            f.write('\t')
            f.write('pos'+'\n')

        lines = open('data/rt-polaritydata/rt-polarity.neg', 'r', encoding='utf-8').readlines()
        for line in lines:
            f.write(line.strip())
            f.write('\t')
            f.write('neg'+'\n')


if __name__ == '__main__':
    make_tsv()
