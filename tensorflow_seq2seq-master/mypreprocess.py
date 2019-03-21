import nltk
import itertools

FILEPATH_N = './data/prenormal.txt'

FILEPATH_S = './data/presimple.txt'

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '

CH_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

MAX_LENGTH = 30
# 最常出现的一万项


most_vocab_size = 10000


class mypreprocess:
    def __init__(self):
        self.data_source=self.read_data(FILEPATH_N)
        self.data_target = self.read_data(FILEPATH_S)


    def read_data(self,filepath):
        data = open(filepath).readlines()
        data = [line[:-1] for line in data]
        data_lines=[line.lower() for line in data]
        lines = [self.filter_line(nline, EN_WHITELIST, en_ch=True) for nline in data_lines]
        data_lines_list = [line.split(" ") for line in lines]
        return data_lines_list

    def process_all_data(self,data_source, data_target, en_ch=True):



        data_lines_list = data_source + data_target
        # print(data_lines_list[0])

        freq_dist = nltk.FreqDist(itertools.chain(*data_lines_list))

        # print(freq_dist["country"])

        VOCAB = freq_dist.most_common(most_vocab_size)
        # print(VOCAB[0])

        # build most frequence word dictionary
        int2word = ['<PAD>'] + ['<UNK>'] + ['<GO>'] + ['<EOS>'] + [x[0] for x in VOCAB]
        word2int = dict([(w, i) for i, w in enumerate(int2word)])

        for line in data_lines_list:
            for i in range(len(line)):
                line[i] = word2int.get(line[i], '<UNK>')

        for line in data_lines_list:
            if len(line) < MAX_LENGTH:
                for _ in range(MAX_LENGTH - len(line)):
                    line.append(word2int.get('<PAD>'))

        return data_lines_list, int2word,word2int

## 正文开始
    def getDic(self,data_source, data_target, en_ch=True):
        data = data_source + data_target

        if en_ch:
            data_lines = [line.lower() for line in data]
            lines = [self.filter_line(nline, EN_WHITELIST, en_ch=True) for nline in data_lines]
        else:
            lines = [self.filter_line(nline, CH_BLACKLIST, en_ch=False) for nline in data]

        data_lines_list = [line.split(" ") for line in lines]

        freq_dist = nltk.FreqDist(itertools.chain(*data_lines_list))

        # print(freq_dist["country"])

        VOCAB = freq_dist.most_common(most_vocab_size)
        # print(VOCAB[0])

        # build most frequence word dictionary
        int2word = ['<PAD>'] + ['<UNK>'] + ['<GO>'] + ['<EOS>'] + [x[0] for x in VOCAB]
        word2int = dict([(w, i) for i, w in enumerate(int2word)])
        return word2int

    def process_data(self):

        data_lines_list,int2word,word2int = self.process_all_data(self.data_source, self.data_target, en_ch=True)

        input_source_int = data_lines_list[:len(self.data_source)]
        output_target_int = data_lines_list[len(self.data_source):]
        # output_target_int= [['GO']+[letter for letter in line]+['<EOS>'] for line in output_target_int]
        output_target_int = [[letter for letter in line] + ['<EOS>'] for line in output_target_int]

        return input_source_int, output_target_int, int2word, word2int

    def filter_line(self,line, charlist, en_ch=True):
        if en_ch:
            return "".join([ch for ch in line if ch in charlist])
        else:
            return "".join([ch for ch in line if ch not in charlist])


# source_int,target_int ,int2word, word2int= mypreprocess().process_data()
# print((source_int[:10]))
# print((target_int[:10]))
# print(len(int2word))