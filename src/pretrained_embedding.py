import os
import torch
import argparse
import data_utils
import numpy as np
import pickle as pkl

from sys import path
from torch import nn
from tqdm import tqdm

def generate_embedding_from_glove(args):
    data_dir = args.data_dir
    min_freq = 2
    max_vocab_size = 15000
    pretrained_embedding_dir = args.pretrained_embedding

    word_manager = data_utils.SymbolsManager(True)
    word_manager.init_from_file("{}/vocab.q.txt".format(data_dir), min_freq, max_vocab_size)

    glove2vec = {}
    words_arr = []
    cnt_find = 0
    with open(pretrained_embedding_dir, "r") as f:
        for l in tqdm(f):
            line = l.split()
            word = line[0]
            words_arr.append(word)
            vect = np.array(line[1:]).astype(np.float)
            glove2vec[word] = vect

    word2vec = {}
    word_arr = word_manager.symbol2idx.keys()
    for w in tqdm(word_arr):
        if w in glove2vec.keys():
            word2vec[w] = glove2vec[w]

    print len(word2vec)
    out_file = "{}/pretrain.pkl".format(data_dir)
    with open(out_file, "wb") as out_data:
        pkl.dump(word2vec, out_data)

def make_pretrained_embedding(embedding_size, opt):
    # use glove pretrained embedding and vocabulary to generate a embedding matrix
    data_dir = opt.data_dir
    min_freq = 2
    max_vocab_size = 15000
    torch.manual_seed(opt.seed)

    word2vec = pkl.load( open("{}/pretrain.pkl".format("../data/TextData"), "rb" ) )
    managers = pkl.load( open("{}/map.pkl".format(data_dir), "rb" ) )
    word_manager, form_manager = managers
    
    num_embeddings, embedding_dim = embedding_size
    weight_matrix = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float)
    cnt_change = 0
    for i in range(num_embeddings):
        word = word_manager.idx2symbol[i]
        if word in word2vec:
            weight_matrix[i] = torch.from_numpy(word2vec[word])
            cnt_change += 1
        else:
            weight_matrix[i] = torch.randn((embedding_dim, ))
    print cnt_change
    return weight_matrix

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-data_dir', type=str, default='../data/TextData', help='data path')
    main_arg_parser.add_argument('-pretrained_embedding', type=str, default="/home/lishucheng/projects/Tools-and-Resources/glove/glove.6B.300d.txt")

    args = main_arg_parser.parse_args()
    if os.path.exists("{}/pretrain.pkl".format(args.data_dir)):
        print "word embedding has been generated !"
    else:
        generate_embedding_from_glove(args)