import os
import re
import time
import random
import warnings
import argparse
import data_utils
import graph_utils

import numpy as np
import pickle as pkl
from tree import Tree

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim

from graph_encoder import GraphEncoder

warnings.filterwarnings('ignore')

class Dec_LSTM(nn.Module):
    def __init__(self, opt):
        super(Dec_LSTM, self).__init__()
        self.opt = opt
        self.word_embedding_size = 300
        self.i2h = nn.Linear(self.word_embedding_size+2*opt.rnn_size, 4*opt.rnn_size)
        self.h2h = nn.Linear(opt.rnn_size, 4*opt.rnn_size)

        if opt.dropout_de_out > 0:
            self.dropout = nn.Dropout(opt.dropout_de_out)

    def forward(self, x, prev_c, prev_h, parent_h, sibling_state):
        input_cat = torch.cat((x, parent_h, sibling_state),1)
        gates = self.i2h(input_cat) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4,1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        if self.opt.dropout_de_out > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return cy, hy

class DecoderRNN(nn.Module):
    def __init__(self, opt, input_size):
        super(DecoderRNN, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
        self.word_embedding_size = 300
        self.embedding = nn.Embedding(input_size, self.word_embedding_size, padding_idx=0)

        self.lstm = Dec_LSTM(self.opt)
        if opt.dropout_de_in > 0:
            self.dropout = nn.Dropout(opt.dropout_de_in)

    def forward(self, input_src, prev_c, prev_h, parent_h, sibling_state):

        src_emb = self.embedding(input_src)
        if self.opt.dropout_de_in > 0:
            src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h, parent_h, sibling_state)
        return prev_cy, prev_hy

class AttnUnit(nn.Module):
    def __init__(self, opt, output_size):
        super(AttnUnit, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
        self.separate_attention = True
        if self.separate_attention:
            self.linear_att = nn.Linear(3*self.hidden_size, self.hidden_size)
        else:
            self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.linear_out = nn.Linear(self.hidden_size, output_size)
        if opt.dropout_for_predict > 0:
            self.dropout = nn.Dropout(opt.dropout_for_predict)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top, enc_2):
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0,2,1), attention)

        if self.separate_attention:
            dot_2 = torch.bmm(enc_2, dec_s_top.unsqueeze(2))
            attention_2 = self.softmax(dot_2.squeeze(2)).unsqueeze(2)
            enc_attention_2 = torch.bmm(enc_2.permute(0,2,1), attention_2)

        if self.separate_attention:
            hid = F.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2), enc_attention_2.squeeze(2),dec_s_top), 1)))
        else:
            hid = F.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2),dec_s_top), 1)))
        h2y_in = hid
        if self.opt.dropout_for_predict > 0:
            h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)

        return pred

def get_dec_batch(dec_tree_batch, opt, using_gpu, form_manager):
    queue_tree = {}
    for i in range(1, opt.batch_size+1):
        queue_tree[i] = []
        queue_tree[i].append({"tree" : dec_tree_batch[i-1], "parent": 0, "child_index": 1})

    cur_index, max_index = 1,1
    dec_batch = {}
    # max_index: the max number of sequence decoder in one batch
    while (cur_index <= max_index):
        max_w_len = -1
        batch_w_list = []
        for i in range(1, opt.batch_size+1):
            w_list = []
            if (cur_index <= len(queue_tree[i])):
                t = queue_tree[i][cur_index - 1]["tree"]

                for ic in range (t.num_children):
                    if isinstance(t.children[ic], Tree):
                        w_list.append(4)
                        queue_tree[i].append({"tree" : t.children[ic], "parent" : cur_index, "child_index": ic + 1})
                    else:
                        w_list.append(t.children[ic])
                if len(queue_tree[i]) > max_index:
                    max_index = len(queue_tree[i])
            if len(w_list) > max_w_len:
                max_w_len = len(w_list)
            batch_w_list.append(w_list)
        dec_batch[cur_index] = torch.zeros((opt.batch_size, max_w_len + 2), dtype=torch.long)
        for i in range(opt.batch_size):
            w_list = batch_w_list[i]
            if len(w_list) > 0:
                for j in range(len(w_list)):
                    dec_batch[cur_index][i][j+1] = w_list[j]
                # add <S>, <E>
                if cur_index == 1:
                    dec_batch[cur_index][i][0] = 1
                else:
                    dec_batch[cur_index][i][0] = form_manager.get_symbol_idx('(')
                dec_batch[cur_index][i][len(w_list) + 1] = 2

        if using_gpu:
            dec_batch[cur_index] = dec_batch[cur_index].cuda()
        cur_index += 1

    return dec_batch, queue_tree, max_index


def eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, word_manager, form_manager):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    attention_decoder_optimizer.zero_grad()
    enc_batch, enc_len_batch, dec_tree_batch = train_loader.random_batch()

    enc_max_len = enc_len_batch

    enc_outputs = torch.zeros((opt.batch_size, enc_max_len, encoder.hidden_layer_dim), requires_grad=True)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()

    fw_adj_info = torch.tensor(enc_batch['g_fw_adj'])
    bw_adj_info = torch.tensor(enc_batch['g_bw_adj'])
    feature_info = torch.tensor(enc_batch['g_ids_features'])
    batch_nodes = torch.tensor(enc_batch['g_nodes'])

    node_embedding, graph_embedding, structural_info = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes))

    enc_outputs = node_embedding

    graph_cell_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
    graph_hidden_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
    if using_gpu:
        graph_cell_state = graph_cell_state.cuda()
        graph_hidden_state = graph_hidden_state.cuda()
    
    graph_cell_state = graph_embedding
    graph_hidden_state = graph_embedding

    dec_s = {}
    for i in range(opt.dec_seq_length + 1):
        dec_s[i] = {}
        for j in range(opt.dec_seq_length + 1):
            dec_s[i][j] = {}

    loss = 0
    cur_index = 1

    dec_batch, queue_tree, max_index = get_dec_batch(dec_tree_batch, opt, using_gpu, form_manager)

    while (cur_index <= max_index):
        for j in range(1, 3):
            dec_s[cur_index][0][j] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
            if using_gpu:
                dec_s[cur_index][0][j] = dec_s[cur_index][0][j].cuda()

        sibling_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
        if using_gpu:
                sibling_state = sibling_state.cuda()

        if cur_index == 1:
            for i in range(opt.batch_size):
                dec_s[1][0][1][i, :] = graph_cell_state[i]
                dec_s[1][0][2][i, :] = graph_hidden_state[i]

        else:
            for i in range(1, opt.batch_size+1):
                if (cur_index <= len(queue_tree[i])):
                    par_index = queue_tree[i][cur_index - 1]["parent"]
                    child_index = queue_tree[i][cur_index - 1]["child_index"]
                    
                    dec_s[cur_index][0][1][i-1,:] = \
                        dec_s[par_index][child_index][1][i-1,:]
                    dec_s[cur_index][0][2][i-1,:] = dec_s[par_index][child_index][2][i-1,:]

                flag_sibling = False
                for q_index in range(len(queue_tree[i])):
                    if (cur_index <= len(queue_tree[i])) and (q_index < cur_index - 1) and (queue_tree[i][q_index]["parent"] == queue_tree[i][cur_index - 1]["parent"]) and (queue_tree[i][q_index]["child_index"] < queue_tree[i][cur_index - 1]["child_index"]):
                        flag_sibling = True
                        sibling_index = q_index
                if flag_sibling:
                    sibling_state[i - 1, :] = dec_s[sibling_index][dec_batch[sibling_index].size(1) - 1][2][i - 1,:]
                
        parent_h = dec_s[cur_index][0][2]
        for i in range(dec_batch[cur_index].size(1) - 1):
            teacher_force = random.random() < opt.teacher_force_ratio
            if teacher_force != True and i > 0:
                input_word = pred.argmax(1)
            else:
                input_word = dec_batch[cur_index][:, i]

            dec_s[cur_index][i+1][1], dec_s[cur_index][i+1][2] = decoder(input_word, dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h, sibling_state)
            pred = attention_decoder(enc_outputs, dec_s[cur_index][i+1][2], structural_info)
            loss += criterion(pred, dec_batch[cur_index][:,i+1])
        cur_index = cur_index + 1

    loss = loss / opt.batch_size
    loss.backward()
    torch.nn.utils.clip_grad_value_(encoder.parameters(),opt.grad_clip)
    torch.nn.utils.clip_grad_value_(decoder.parameters(),opt.grad_clip)
    torch.nn.utils.clip_grad_value_(attention_decoder.parameters(),opt.grad_clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    attention_decoder_optimizer.step()
    return loss

def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)

def do_generate(encoder, decoder, attention_decoder, graph_input, word_manager, form_manager, opt, using_gpu, checkpoint):    
    prev_c = torch.zeros((1, encoder.hidden_layer_dim), requires_grad=False)
    prev_h = torch.zeros((1, encoder.hidden_layer_dim), requires_grad=False)
    if using_gpu:
        prev_c = prev_c.cuda()
        prev_h = prev_h.cuda()

    graph_size = len(graph_input['g_nodes'][0])
    enc_outputs = torch.zeros((1, graph_size, encoder.hidden_layer_dim), requires_grad=False)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()

    if graph_input['g_fw_adj'] == []:
        return "None"
    fw_adj_info = torch.tensor(graph_input['g_fw_adj'])
    bw_adj_info = torch.tensor(graph_input['g_bw_adj'])
    feature_info = torch.tensor(graph_input['g_ids_features'])
    batch_nodes = torch.tensor(graph_input['g_nodes'])

    node_embedding, graph_embedding, structural_info = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes))
    enc_outputs = node_embedding
    prev_c = graph_embedding
    prev_h = graph_embedding

    queue_decode = []
    queue_decode.append({"s": (prev_c, prev_h), "parent":0, "child_index":1, "t": Tree()})
    head = 1
    while head <= len(queue_decode) and head <=100:
        s = queue_decode[head-1]["s"]
        parent_h = s[1]
        t = queue_decode[head-1]["t"]

        sibling_state = torch.zeros((1, encoder.hidden_layer_dim), dtype=torch.float, requires_grad=False)

        if using_gpu:
            sibling_state = sibling_state.cuda()
        flag_sibling = False
        for q_index in range(len(queue_decode)):
            if (head <= len(queue_decode)) and (q_index < head - 1) and (queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) and (queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
                flag_sibling = True
                sibling_index = q_index
        if flag_sibling:
            sibling_state = queue_decode[sibling_index]["s"][1]

        if head == 1:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long)
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx('(')], dtype=torch.long)
        if using_gpu:
            prev_word = prev_word.cuda()
        i_child = 1
        while True:
            curr_c, curr_h = decoder(prev_word, s[0], s[1], parent_h, sibling_state)
            prediction = attention_decoder(enc_outputs, curr_h, structural_info)
 
            s = (curr_c, curr_h)
            _, _prev_word = prediction.max(1)
            prev_word = _prev_word

            if int(prev_word[0]) == form_manager.get_symbol_idx('<E>') or t.num_children >= checkpoint["opt"].dec_seq_length:
                break
            elif int(prev_word[0]) == form_manager.get_symbol_idx('<N>'):
                queue_decode.append({"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index":i_child, "t": Tree()})
                t.add_child(int(prev_word[0]))
            else:
                t.add_child(int(prev_word[0]))
            i_child = i_child + 1
        head = head + 1
    for i in range(len(queue_decode)-1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"]-1]["t"].children[cur["child_index"]-1] = cur["t"]
    return queue_decode[0]["t"].to_list(form_manager)

def main(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    managers = pkl.load( open("{}/map.pkl".format(opt.data_dir), "rb" ) )
    word_manager, form_manager = managers
    using_gpu = False
    if opt.gpuid > -1:
        using_gpu = True

    encoder = GraphEncoder(opt, word_manager.vocab_size)
    decoder = DecoderRNN(opt, form_manager.vocab_size)
    attention_decoder = AttnUnit(opt, form_manager.vocab_size)
    
    if using_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attention_decoder = attention_decoder.cuda()

    # print(encoder)
    # print(decoder)
    # print(attention_decoder)

    for name, param in encoder.named_parameters():
        if param.requires_grad:
            if ("embedding.weight" in name):
                    # print("Do not initialize pretrained embedding parameters")
                pass
            else:
                init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in attention_decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)

    train_loader = data_utils.MinibatchLoader(opt, 'train', using_gpu, word_manager)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    step = 0
    epoch = 0
    optim_state = {"learningRate" : opt.learning_rate}

    print("using adam")
    encoder_optimizer = optim.Adam(encoder.parameters(),  lr=optim_state["learningRate"], weight_decay=1e-5)
    decoder_optimizer = optim.Adam(decoder.parameters(),  lr=optim_state["learningRate"])
    attention_decoder_optimizer = optim.Adam(attention_decoder.parameters(),  lr=optim_state["learningRate"])

    criterion = nn.NLLLoss(size_average=False, ignore_index=0)

    print("Starting training.")
    encoder.train()
    decoder.train()
    attention_decoder.train()
    iterations = opt.max_epochs * train_loader.num_batch
    start_time = time.time()
    restarted = False

    best_val_acc = 0
    loss_to_print = 0
    for i in range(iterations):

        epoch = i // train_loader.num_batch
        train_loss = eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, word_manager, form_manager)

        loss_to_print += train_loss

        if i == iterations - 1 or i % opt.print_every == 0:
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = decoder
            checkpoint["attention_decoder"] = attention_decoder
            checkpoint["opt"] = opt
            checkpoint["i"] = i
            checkpoint["epoch"] = epoch
            torch.save(checkpoint, "{}/valid/model_g2t".format(opt.checkpoint_dir) + str(i))

        if i % opt.print_every == 0:
            end_time = time.time()
            print("{}/{}, train_loss = {}, time since last print = {}".format( i, iterations, loss_to_print/opt.print_every, (end_time - start_time)/60))
            loss_to_print = 0
            start_time = time.time()

    print "best_acc: ",best_val_acc


if __name__ == "__main__":
    start = time.time()
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-data_dir', type=str, default='../data/GraphConstruction', help='data path')
    main_arg_parser.add_argument('-seed',type=int,default=400,help='torch manual random number generator seed')
    main_arg_parser.add_argument('-checkpoint_dir',type=str, default= 'checkpoint_dir', help='output directory where checkpoints get written')
    main_arg_parser.add_argument('-print_every',type=int, default=100,help='how many steps/minibatches between printing out the loss')
    main_arg_parser.add_argument('-rnn_size', type=int,default=300, help='size of LSTM internal state')
    main_arg_parser.add_argument('-num_layers', type=int, default=1, help='number of layers in the LSTM')

    main_arg_parser.add_argument('-teacher_force_ratio',type=float, default=1.0)

    main_arg_parser.add_argument('-dropout_en_in',type=float, default=0.1,help='dropout for encoder, input')
    main_arg_parser.add_argument('-dropout_en_out',type=float, default=0.3,help='dropout for encoder, output')
    main_arg_parser.add_argument('-dropout_de_in',type=float, default=0.1,help='dropout for decoder, input')
    main_arg_parser.add_argument('-dropout_de_out',type=float, default=0.3,help='dropout for decoder, output')
    main_arg_parser.add_argument('-dropout_for_predict',type=float, default=0.1,help='dropout used in attention decoder, in prediction')

    main_arg_parser.add_argument('-dropoutagg',type=float,default=0,help='dropout for regularization, used after each aggregator. 0 = no dropout')
    main_arg_parser.add_argument('-dec_seq_length',type=int, default=35,help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-batch_size',type=int, default=30,help='number of sequences to train on in parallel')

    main_arg_parser.add_argument('-max_epochs',type=int, default=800,help='number of full passes through the training data')
    main_arg_parser.add_argument('-learning_rate',type=float, default=1e-3,help='learning rate')
    main_arg_parser.add_argument('-init_weight',type=float, default=0.08,help='initailization weight')
    main_arg_parser.add_argument('-grad_clip',type=int, default=5,help='clip gradients at this value')

    # some arguments of graph encoder
    main_arg_parser.add_argument('-graph_encode_direction',type=str, default='uni',help='graph encode direction: bi or uni')
    main_arg_parser.add_argument('-sample_size_per_layer',type=int, default=10,help='sample_size_per_layer')
    main_arg_parser.add_argument('-sample_layer_size',type=int, default=3,help='sample_layer_size')
    main_arg_parser.add_argument('-concat',type=bool, default=True,help='concat in aggregators settings')

    args = main_arg_parser.parse_args()
    print(args)
    main(args)
    end = time.time()
    print("total time: {} minutes\n".format((end - start)/60))
