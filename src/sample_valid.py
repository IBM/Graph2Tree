import copy
import torch
import random
import warnings
import argparse
import data_utils
import graph_utils
import numpy as np
import pickle as pkl

from tree import Tree
from graph2tree import *

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-data_dir', type=str, default='../data/GraphConstruction', help='data path')
    main_arg_parser.add_argument('-model_dir', type=str, default='checkpoint_dir/valid/', help='model checkpoint to use for sampling')
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')
    main_arg_parser.add_argument('-output_model', type=str, default='checkpoint_dir/output_model', help='best model output')

    args = main_arg_parser.parse_args()

    managers = pkl.load( open("{}/map.pkl".format(args.data_dir), "rb" ) )
    word_manager, form_manager = managers

    data = pkl.load(open("{}/valid.pkl".format(args.data_dir), "rb"))
    graph_test_list = graph_utils.read_graph_data("{}/graph.valid".format(args.data_dir))
    
    max_acc = 0
    max_index = 0

    model_num = 0
    while(1):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print model_num
        try:
            checkpoint = torch.load(args.model_dir + "model_g2t" + str(model_num))
        except BaseException:
            break

        encoder = checkpoint["encoder"]
        decoder = checkpoint["decoder"]
        attention_decoder = checkpoint["attention_decoder"]

        encoder.eval()
        decoder.eval()
        attention_decoder.eval()

        reference_list = []
        candidate_list = []
        add_acc = 0.0
            
        for i in range(len(data)):
            x = data[i]
            reference = x[1]
            graph_batch = graph_utils.cons_batch_graph([graph_test_list[i]])
            graph_input = graph_utils.vectorize_batch_graph(graph_batch, word_manager)
            candidate = do_generate(encoder, decoder, attention_decoder, graph_input, word_manager, form_manager, args, True, checkpoint)
            candidate = [int(c) for c in candidate]
            num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== "(")
            num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== ")")
            diff = num_left_paren - num_right_paren
            if diff > 0:
                for i in range(diff):
                    candidate.append(form_manager.symbol2idx[")"])
            elif diff < 0:
                candidate = candidate[:diff]
            ref_str = convert_to_string(reference, form_manager)
            cand_str = convert_to_string(candidate, form_manager)
            reference_list.append(reference)
            candidate_list.append(candidate)
        val_acc = data_utils.compute_tree_accuracy(candidate_list, reference_list, form_manager)
        print("ACCURACY = {}\n".format(val_acc))
        if val_acc >= max_acc:
            max_acc = val_acc
            max_index = model_num
        
        model_num += 100
    print "max accuracy:", max_acc
    best_valid_model = torch.load(args.model_dir + "model_g2t" + str(max_index))
    torch.save(best_valid_model, args.output_model)


