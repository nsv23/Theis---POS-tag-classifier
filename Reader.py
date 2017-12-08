import csv
from itertools import islice
import numpy as np

words = []
pos_tags = []
sentences = []
test_sentences = []
pos_temp = []
test_pos_temp = []

char_codes = {
            0 : 0,
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
            "h": 8,
            "i": 9,
            "j": 10,
            "k": 11,
            "l": 12,
            "m": 13,
            "n": 14,
            "o": 15,
            "p": 16,
            "q": 17,
            "r": 18,
            "s": 19,
            "t": 20,
            "u": 21,
            "v": 22,
            "w": 23,
            "x": 24,
            "y": 25,
            "z": 26,
            "A": 27,
            "B": 28,
            "C": 29,
            "D": 30,
            "E": 31,
            "F": 32,
            "G": 33,
            "H": 34,
            "I": 35,
            "J": 36,
            "K": 37,
            "L": 38,
            "M": 39,
            "N": 40,
            "O": 41,
            "P": 42,
            "Q": 43,
            "R": 44,
            "S": 45,
            "T": 46,
            "U": 47,
            "V": 48,
            "W": 49,
            "X": 50,
            "Y": 51,
            "Z": 52,
            "0": 53,
            "1": 54,
            "2": 55,
            "3": 56,
            "4": 57,
            "5": 58,
            "6": 59,
            "7": 60,
            "8": 61,
            "9": 62,
            ".": 63,
            ",": 64,
            "'": 65,
            "``": 66,
            "''": 67,
            "-": 68,
            "--": 69,
            ":": 70,
            ";": 71,
            "(": 72,
            ")": 73,
            "{": 74,
            "}": 75,
            "[": 76,
            "]": 77,
            "$": 78,
            "#": 79,
            "`": 80,
            "&": 81,
            "%": 82,
            "\\": 83,
            "\"": 84,
            "/": 85,
            "?": 86,
            "@": 87,
            "!": 88,
            "*": 89,
            "=": 90
        }

pos_ind_values = {
            "CC": 1,
            "CD": 2,
            "DT": 3,
            "EX": 4,
            "FW": 5,
            "IN": 6,
            "JJ": 7,
            "JJR": 8,
            "JJS": 9,
            "LS": 10,
            "MD": 11,
            "NN": 12,
            "NNS": 13,
            "NNP": 14,
            "NNPS": 15,
            "PDT": 16,
            "POS": 17,
            "PRP": 18,
            "PRP$": 19,
            "RB": 20,
            "RBR": 21,
            "RBS": 22,
            "RP": 23,
            "SYM": 24,
            "TO": 25,
            "UH": 26,
            "VB": 27,
            "VBD": 28,
            "VBG": 29,
            "VBN": 30,
            "VBP": 31,
            "VBZ": 32,
            "WDT": 33,
            "WP": 34,
            "WP$": 35,
            "WRB": 36,
            ".": 37,
            ",": 38,
            "'": 39,
            "``": 40,
            "''": 40,
            "-": 41,
            "--": 41,
            ":": 41,
            ";": 41,
            "(": 42,
            ")": 43,
            "{": 42,
            "}": 43,
            "[": 42,
            "]": 43,
            "$": 44,
            "#": 45
        }

with open('penntreebank.conllx', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) != 0:
            words.append(row[1])
            pos_tags.append(row[3])
        else:
            if len(sentences) < 39208:
                sentences.append(words)
                pos_temp.append(pos_tags)
                words = []
                pos_tags = []
            else:
                test_sentences.append(words)
                test_pos_temp.append(pos_tags)
                words = []
                pos_tags = []
# print(pos_temp)



# dict = {}
# dict["<pad>"] = len(dict)  # 0 padding
# words = ["<pad>"]  # keeps order of insertion in dict
#
# res_sentences = []
# for sentence in sentences:
#     res_sentence = []
#     for k in range(20):
#         if k < len(sentence):
#             word = sentence[k]
#             if word not in dict:
#                 dict[word] = len(dict)
#                 words.append(word)
#             res_sentence.append(dict[word])
#         else:
#             res_sentence.append(dict["<pad>"])
#     res_sentences.append(res_sentence)
#
# print(res_sentences)


def retrieve_batch_sent(start, end, sent_max_len, word_max_len):
    word_voc = []
    word_voc.append(0)
    batch_sentences = []
    batch_pos = []
    temp_ind_sent = []
    word_id_mat = []
    word_voc_char = []
    char_id_mat = []
    pos_id_mat = []
    ind_pos_row = []

    for s in islice(sentences, start, end):
        # print(s)
        batch_sentences.append(s)
        for j in range(len(s)):
            if s[j] not in word_voc:
                word_voc.append(s[j])

    for p in islice(pos_temp, start, end):
        batch_pos.append(p)

    for s in batch_sentences:
        for w in s:
            if w in word_voc:
                index = word_voc.index(w)
                temp_ind_sent.append(index)
        temp_sent_pad = [0] * sent_max_len
        if len(temp_ind_sent) < sent_max_len:
            temp_sent_pad[:len(temp_ind_sent)] = temp_ind_sent
        elif len(temp_ind_sent) >= sent_max_len:
            temp_sent_pad[:sent_max_len] = temp_ind_sent[:sent_max_len]

        # temp_sent_pad[:sent_max_len] = temp_ind_sent
        word_id_mat.append(temp_sent_pad)
        temp_ind_sent = []

    for r in batch_pos:
        for w in r:
            ind_pos_row.append(pos_ind_values[w])
        padded_pos = [0] * sent_max_len
        if len(ind_pos_row) < sent_max_len:
            padded_pos[:len(ind_pos_row)] = ind_pos_row
        elif len(ind_pos_row) >= sent_max_len:
            padded_pos[:sent_max_len] = ind_pos_row[:sent_max_len]
        # padded_pos[:sent_max_len] = ind_pos_row
        pos_id_mat.append(padded_pos)
        ind_pos_row = []

    for w in word_voc:
        if w != 0:
            for c in w:
                word_voc_char.append(char_codes[c])
            word_voc_char_pad = [0] * word_max_len
            if len(word_voc_char) < word_max_len:
                word_voc_char_pad[:len(word_voc_char)] = word_voc_char
            elif len(word_voc_char) >= word_max_len:
                word_voc_char_pad[:word_max_len] = word_voc_char[:word_max_len]
            # word_voc_char_pad[:word_max_len] = word_voc_char
            char_id_mat.append(word_voc_char_pad)
            word_voc_char = []
            # print(char_id_mat)
        else:
            word_voc_char.append(char_codes[w])
            # word_voc_char_pad = [0] * len(max(word_voc[1:], key=len))
            word_voc_char_pad = word_voc_char * word_max_len
            char_id_mat.append(word_voc_char_pad)
            word_voc_char = []
            # print(char_id_mat)
        # temp_sent_pad = [0] * len(max(batch_sentences, key=len))

    # print('\n')
    # print(batch_sentences)
    # print(len(word_voc))
    # print(len(max(batch_sentences, key=len)))
    # print(word_id_mat)
    # print(char_id_mat)
    # print(len(word_id_mat))
    # print(len(char_id_mat))
    # print(len(pos_id_mat))
    # print(pos_id_mat)
    # print(len(max(word_voc[1:], key=len)))

    return char_id_mat, word_id_mat, pos_id_mat # word_voc, len(max(word_voc[1:], key=len)), len(max(batch_sentences, key=len))

# print(len(char_codes))

# word_max_size = 15
# _PAD = 0
# _GO = 1
# _EOW = 2
# _UNK = 3
# batch_sentences = []
# word_voc = []
#
# for s in sentences:
#     # print(s)
#     for j in range(len(s)):
#         if s[j] not in word_voc:
#             word_voc.append(s[j])
#
#
# char_words = np.ndarray(shape=[len(word_voc), word_max_size], dtype=np.int32)
# print("hi")
# for i in range(len(word_voc)):
#     if word_voc[i]=="<blank>":
#         char_words[i][:] = _PAD
#         continue
#     char_words[i][0]=_GO
#     for j in range(1,word_max_size):
#         if j < len(word_voc[i])+1:
#             char_words[i][j] = ord(word_voc[i][j-1])
#         elif j == len(word_voc[i])+1:
#             char_words[i][j] = _EOW
#         else:
#             char_words[i][j] = _PAD
#     if char_words[i][word_max_size-1] != _PAD:
#         char_words[i][word_max_size-1] =_EOW
#
# print(char_words)