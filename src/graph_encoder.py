import os
import time
import random
import data_utils
import nn_modules
import numpy as np
import pickle as pkl
from tree import Tree
from pretrained_embedding import make_pretrained_embedding

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim

class GraphEncoder(nn.Module):
    def __init__(self, opt, input_size):
        super(GraphEncoder, self).__init__()
        self.opt = opt

        if opt.dropoutagg > 0:
            self.dropout = nn.Dropout(opt.dropoutagg)

        self.graph_encode_direction = opt.graph_encode_direction
        self.sample_size_per_layer = opt.sample_size_per_layer
        self.sample_layer_size = opt.sample_layer_size
        self.hidden_layer_dim = opt.rnn_size

        if self.opt.dropout_en_in > 0:
            self.input_dropout = nn.Dropout(opt.dropout_en_in)
        self.word_embedding_size = 300
        self.embedding = nn.Embedding(
            input_size, self.word_embedding_size, padding_idx=0)
        self.embedding.weight.data = make_pretrained_embedding(self.embedding.weight.size(), opt)

        self.fw_aggregators = []
        self.bw_aggregators = []

        self.fw_aggregator_0 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_1 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_2 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_3 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_4 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_5 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregator_6 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)

        self.bw_aggregator_0 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_1 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_2 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_3 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_4 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_5 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.bw_aggregator_6 = nn_modules.MeanAggregator(
            2*self.hidden_layer_dim, self.hidden_layer_dim, concat=True)
        self.fw_aggregators = [self.fw_aggregator_0, self.fw_aggregator_1, self.fw_aggregator_2,
                               self.fw_aggregator_3, self.fw_aggregator_4, self.fw_aggregator_5, self.fw_aggregator_6]
        self.bw_aggregators = [self.bw_aggregator_0, self.bw_aggregator_1, self.bw_aggregator_2,
                               self.bw_aggregator_3, self.bw_aggregator_4, self.bw_aggregator_5, self.bw_aggregator_6]

        self.Linear_hidden = nn.Linear(
            2 * self.hidden_layer_dim, self.hidden_layer_dim)

        self.concat = opt.concat

        self.using_gpu = False
        if self.opt.gpuid > -1:
            self.using_gpu = True

        self.embedding_bilstm = nn.LSTM(input_size=self.word_embedding_size, hidden_size=self.hidden_layer_dim/2, bidirectional=True, bias = True, batch_first = True, dropout= self.opt.dropout_en_out, num_layers=1)
        self.padding_vector = torch.randn(1,self.hidden_layer_dim, dtype = torch.float, requires_grad=True)

    def forward(self, graph_batch):
        fw_adj_info, bw_adj_info, feature_info, batch_nodes = graph_batch

        # print self.hidden_layer_dim

        if self.using_gpu:
            fw_adj_info = fw_adj_info.cuda()
            bw_adj_info = bw_adj_info.cuda()
            feature_info = feature_info.cuda()
            batch_nodes = batch_nodes.cuda()

        feature_by_sentence = feature_info[:-1,:].view(batch_nodes.size()[0], -1)
        feature_sentence_vector = self.embedding(feature_by_sentence)
        if self.opt.dropout_en_in > 0:
            feature_sentence_vector = self.input_dropout(feature_sentence_vector)
        output_vector, (ht,_) = self.embedding_bilstm(feature_sentence_vector)

        feature_vector = output_vector.contiguous().view(-1, self.hidden_layer_dim)
        if self.using_gpu:
            feature_embedded = torch.cat([feature_vector, self.padding_vector.cuda()], 0)
        else:
            feature_embedded = torch.cat([feature_vector, self.padding_vector], 0)

        batch_size = feature_embedded.size()[0]
        node_repres = feature_embedded.view(batch_size, -1)

        fw_sampler = nn_modules.UniformNeighborSampler(fw_adj_info)
        bw_sampler = nn_modules.UniformNeighborSampler(bw_adj_info)
        nodes = batch_nodes.view(-1, )

        fw_hidden = F.embedding(nodes, node_repres)
        bw_hidden = F.embedding(nodes, node_repres)

        fw_sampled_neighbors = fw_sampler((nodes, self.sample_size_per_layer))
        bw_sampled_neighbors = bw_sampler((nodes, self.sample_size_per_layer))

        fw_sampled_neighbors_len = torch.tensor(0)
        bw_sampled_neighbors_len = torch.tensor(0)

        # begin sampling
        for layer in range(self.sample_layer_size):
            if layer == 0:
                dim_mul = 1
            else:
                dim_mul = 1
            if self.using_gpu and layer <= 6:
                self.fw_aggregators[layer] = self.fw_aggregators[layer].cuda()
            if layer == 0:
                neigh_vec_hidden = F.embedding(
                    fw_sampled_neighbors, node_repres)
                tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                tmp_mask = torch.sign(tmp_sum)
                fw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
            else:
                if self.using_gpu:
                    neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros(
                        [1, dim_mul * self.hidden_layer_dim]).cuda()], 0))
                else:
                    neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros(
                        [1, dim_mul * self.hidden_layer_dim])], 0))

            if layer > 6:
                    fw_hidden = self.fw_aggregators[6](
                        (fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            else:
                    fw_hidden = self.fw_aggregators[layer](
                        (fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))

            if self.graph_encode_direction == "bi":
                if self.using_gpu and layer <= 6:
                    self.bw_aggregators[layer] = self.bw_aggregators[layer].cuda(
                    )

                if layer == 0:
                    neigh_vec_hidden = F.embedding(
                        bw_sampled_neighbors, node_repres)
                    tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                    tmp_mask = torch.sign(tmp_sum)
                    bw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
                else:
                    if self.using_gpu:
                        neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros(
                            [1, dim_mul * self.hidden_layer_dim]).cuda()], 0))
                    else:
                        neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros(
                            [1, dim_mul * self.hidden_layer_dim])], 0))
                if self.opt.dropoutagg > 0:
                    bw_hidden = self.dropout(bw_hidden)
                    neigh_vec_hidden = self.dropout(neigh_vec_hidden)

                if layer > 6:
                    bw_hidden = self.bw_aggregators[6](
                        (bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
                else:
                    bw_hidden = self.bw_aggregators[layer](
                        (bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
        fw_hidden = fw_hidden.view(-1, batch_nodes.size()
                                   [1], self.hidden_layer_dim)

        if self.graph_encode_direction == "bi":
            bw_hidden = bw_hidden.view(-1, batch_nodes.size()
                                       [1], self.hidden_layer_dim)
            hidden = torch.cat([fw_hidden, bw_hidden], 2)
        else:
            hidden = fw_hidden

        pooled = torch.max(hidden, 1)[0]
        graph_embedding = pooled.view(-1, self.hidden_layer_dim)

        return hidden, graph_embedding, output_vector
 