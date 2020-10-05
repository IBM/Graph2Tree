import math
import tree
import torch
import sympy
import random
import graph_utils

import numpy as np
import pickle as pkl

from random import randint
from operator import itemgetter
from sympy.parsing.sympy_parser import parse_expr


class SymbolsManager():
    def __init__(self, whether_add_special_tags):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0
        self.whether_add_special_tags = whether_add_special_tags
        if whether_add_special_tags:
            # PAD: padding token = 0
            self.add_symbol('<P>')
            # GO: start token = 1
            self.add_symbol('<S>')
            # EOS: end token = 2
            self.add_symbol('<E>')
            # UNK: unknown token = 3
            self.add_symbol('<U>')
            # NON: non-terminal token = 4
            self.add_symbol('<N>')

    def add_symbol(self,s):
        if s not in self.symbol2idx:
            self.symbol2idx[s] = self.vocab_size
            self.idx2symbol[self.vocab_size] = s
            self.vocab_size += 1
        return self.symbol2idx[s]
    
    def get_symbol_idx(self,s):
        if s not in self.symbol2idx:
            if self.whether_add_special_tags:
                return self.symbol2idx['<U>']
            else:
                print("not reached!")
                return 0
        return self.symbol2idx[s]

    def get_idx_symbol(self, idx):
        if idx not in self.idx2symbol:
            return '<U>'
        return self.idx2symbol[idx]
    
    def init_from_file(self, fn, min_freq, max_vocab_size):
        # the vocab file is sorted by word_freq
        print "loading vocabulary file: {}".format(fn)
        with open(fn,"r") as f:
            for line in f:
                l_list = line.strip().split('\t')
                c = int(l_list[1])
                if c >= min_freq:
                    self.add_symbol(l_list[0])
                if self.vocab_size > max_vocab_size:
                    break

    def get_symbol_idx_for_list(self,l):
        r = []
        for i in range(len(l)):
            r.append(self.get_symbol_idx(l[i]))
        return r
    
class MinibatchLoader():
    def __init__(self, opt, mode, using_gpu, word_manager):
        data = pkl.load(open("{}/{}.pkl".format(opt.data_dir, mode), "rb" ))
        graph_data = graph_utils.read_graph_data("{}/graph.{}".format(opt.data_dir, mode))

        self.enc_batch_list = []
        self.enc_len_batch_list = []
        self.dec_batch_list = []

        p = 0
        while p + opt.batch_size <= len(data):

            batch_graph = [graph_data[p + idx] for idx in range(opt.batch_size)]
            combine_batch_graph = graph_utils.cons_batch_graph(batch_graph)
            vector_batch_graph = graph_utils.vectorize_batch_graph(combine_batch_graph,word_manager)

            self.enc_batch_list.append(vector_batch_graph)
            self.enc_len_batch_list.append(len(batch_graph[0]['g_ids']))

            tree_batch = []
            for i in range(opt.batch_size):
                tree_batch.append(data[p+i][2])
            self.dec_batch_list.append(tree_batch)
            p += opt.batch_size
        self.num_batch = len(self.enc_batch_list)
        assert(len(self.enc_batch_list) == len(self.dec_batch_list))

    def random_batch(self):
        p = randint(0,self.num_batch-1)
        return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]

    def all_batch(self):
        r = []
        for p in range(self.num_batch):
            r.append([self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]])
        return r

def convert_to_tree(r_list, i_left, i_right, form_manager):
    t = tree.Tree()
    level = 0
    left = -1
    for i in range(i_left, i_right):
        if r_list[i] == form_manager.get_symbol_idx('('):
            if level == 0:
                left = i
            level = level + 1
        elif r_list[i] == form_manager.get_symbol_idx(')'):
            level = level -1
            if level == 0:
                if i == left+1:
                    c = r_list[i]
                else:
                    c = convert_to_tree(r_list, left + 1, i, form_manager)
                t.add_child(c)
        elif level == 0:
            t.add_child(r_list[i])
    return t

def is_all_same(c1, c2, form_manager):
    all_same = False
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
    if all_same == False:
        if is_solution_same(c1, c2, form_manager):
            return True
        return False
    else:
        return True

def is_solution_same(i1, i2, form_manager):
    c1 = " ".join([form_manager.get_idx_symbol(x) for x in i1])
    c2 = " ".join([form_manager.get_idx_symbol(x) for x in i2])
    if ('=' not in c1) or ('=' not in c2):
        return False
    elif ('<U>' in c1) or ('<U>' in c2):
        return False
    else:
        try:
            s1 = c1.split('=')
            s2 = c2.split('=')
            eq1 = []
            eq2 = []
            x = sympy.Symbol('x')
            eq1.append(parse_expr(s1[0]))
            eq1.append(parse_expr(s1[1]))
            eq2.append(parse_expr(s2[0]))
            eq2.append(parse_expr(s2[1]))
            res1 = sympy.solve(sympy.Eq(eq1[0], eq1[1]), x)
            res2 = sympy.solve(sympy.Eq(eq2[0], eq2[1]), x)
            
            if not res1 or not res2:
                return False

            return res1[0] == res2[0]

        except BaseException:
            print c1
            print c2
            return False


def compute_accuracy(candidate_list, reference_list, form_manager):
    if len(candidate_list) != len(reference_list):
        print("candidate list has length {}, reference list has length {}\n".format(len(candidate_list), len(reference_list)))
    
    len_min = min(len(candidate_list), len(reference_list))
    c = 0
    for i in range(len_min):
        # print "length:", len_min

        if is_all_same(candidate_list[i], reference_list[i], form_manager):
            # print "{}->True".format(i)
            c = c+1
        else:
            # print "{}->False".format(i)
            pass
    return c/float(len_min)


def compute_tree_accuracy(candidate_list_, reference_list_, form_manager):
    candidate_list = []

    for i in range(len(candidate_list_)):
        candidate_list.append(candidate_list_[i])
    reference_list = []
    for i in range(len(reference_list_)):
        reference_list.append(reference_list_[i])
    return compute_accuracy(candidate_list, reference_list, form_manager)
