import csv
import numpy as np

words = []
pos_tags = []
sentences = []
pos_temp = []

with open('penntreebank.conllx', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) != 0:
            words.append(row[1])
            # pos_tags.append(row[3])
        else:
            sentences.append(words)
            # pos_temp.append(pos_tags)
            words = []
            # pos_tags = []


word_max_size = 15
_PAD = 0
_GO = 1
_EOW = 2
_UNK = 3

word_voc = []

ss = sentences[:251]

for s in ss:
    # print(s)
    for j in range(len(s)):
        if s[j] not in word_voc:
            word_voc.append(s[j])

x = len(word_voc)
y = word_voc
print(x)
# print(word_voc)
# for i in word_voc:
#     print(i)

char_words = np.ndarray(shape=[x, word_max_size], dtype=np.int32)
print(np.shape(char_words))
for i in range(len(y)):
    if y[i]=="<blank>":
        char_words[i][:] = _PAD
        continue
    char_words[i][0]=_GO
    for j in range(1,word_max_size):
        if j < len(y[i])+1:
            char_words[i][j] = ord(y[i][j-1])
        elif j == len(y[i])+1:
            char_words[i][j] = _EOW
        else:
            char_words[i][j] = _PAD
    if char_words[i][word_max_size-1] != _PAD:
        char_words[i][word_max_size-1] =_EOW

for i in range(x):
    print(word_voc[i])
    print(char_words[i])

# print(char_words)