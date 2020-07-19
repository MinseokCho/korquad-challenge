from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm, trange

from models.modeling_bert import QuestionAnswering, Config
from utils.optimization import AdamW, WarmupLinearSchedule
from utils.tokenization import BertTokenizer
from utils.korquad_utils import read_squad_examples, convert_examples_to_features
from utils.korquad_dataset import KorquadDataset

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint", default='pretrain_ckpt/bert_small_ckpt.bin',
                        type=str,
                        help="checkpoint")
    parser.add_argument("--model_config", default='data/bert_small.json',
                        type=str)
    parser.add_argument("--vocab", type=str)
    # Other parameters
    parser.add_argument("--train_file", default='data/KorQuAD_v1.0_train.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=96, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--grad_noise', default='false', type=str)
    parser.add_argument('--gs_noise', default='false', type=str)
    parser.add_argument('--max_num_example', default=80000, type=int)

    args = parser.parse_args()
    args.grad_noise = args.grad_noise.lower() == 'true'
    args.gs_noise = args.gs_noise.lower() == 'true'

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    ckpt_list_fn = os.path.join(args.output_dir, "ckpt_list.txt")
    if os.path.isfile(ckpt_list_fn):
        os.remove(ckpt_list_fn)

    tokenizer = BertTokenizer(args.vocab,
                              max_len=args.max_seq_length,
                              do_basic_tokenize=True)
    # Prepare model
    config = Config.from_json_file(args.model_config)
    model = QuestionAnswering(config)
    
    if 'bert' in args.checkpoint:
        model.bert.load_state_dict(torch.load(args.checkpoint))
        logger.info("bert parameters are loaded from %s"%args.checkpoint)
    else:
        model.load_state_dict(torch.load(args.checkpoint))
        logger.info("all parameter is loaded from %s"%args.checkpoint)
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    model.to(device)

    train_examples = read_squad_examples(input_file=args.train_file, is_training=True, version_2_with_negative=False)
    num_train_examples = len(train_examples)

    cached_train_features_file = args.train_file + '_{0}_{1}_{2}'.format(str(args.max_seq_length), str(args.doc_stride),
                                                                            str(args.max_query_length))
    is_dynamic = False
    if num_train_examples <= args.max_num_example:
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                is_dynamic=False)
            logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
        del train_examples
        num_train_features = len(train_features)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    else:
        # For being able to get data dynamically (for large datasets)
        is_dynamic = True

        pkl_list = []; num_train_features = 0
        if not os.path.isfile(cached_train_features_file + "_0"):
            batch = 10000
            unique_id = 1000000000
            for start, end in zip(range(0, len(train_examples), batch), range(batch, len(train_examples)+batch, batch)):
                file_path = cached_train_features_file + "_%d"%start
                unique_id = convert_examples_to_features(
                                examples=train_examples[start:end],
                                tokenizer=tokenizer,
                                max_seq_length=args.max_seq_length,
                                doc_stride=args.doc_stride,
                                max_query_length=args.max_query_length,
                                is_training=True,
                                is_dynamic=True,
                                file_path=file_path,
                                init_unique_id=unique_id)
                pkl_list.append(file_path)
            num_train_features = unique_id - 1000000000

            with open(cached_train_features_file + "_info", 'w') as fw:
                fw.write("%d\n"%num_train_features)
                for pkl in pkl_list:
                    fw.write(pkl + '\n')
        else:
            with open(cached_train_features_file + "_info", 'r') as f:
                line_cnt = 0
                for line in f:
                    if line_cnt == 0:
                        num_train_features = int(line.strip())
                    else:
                        pkl_list.append(line.strip())
                    line_cnt += 1
        del train_examples

        train_data = KorquadDataset(pkl_list, args.train_batch_size, tokenizer)




    num_train_optimization_steps = int(num_train_features / args.train_batch_size) * args.num_train_epochs

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=num_train_optimization_steps*0.1,
                                     t_total=num_train_optimization_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", num_train_examples)
    logger.info("  Num split examples = %d", num_train_features)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    num_train_step = num_train_optimization_steps

    model.train()
    global_step = 0
    epoch = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        if is_dynamic:
            num_step = int(num_train_step / args.num_train_epochs)
            if num_step * args.train_batch_size != num_train_features:
                num_step += 1
            iter_bar = tqdm(list(range(num_step)), desc="Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")
        else:
            iter_bar = tqdm(train_dataloader, desc="Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")
        tr_step, total_loss, mean_loss = 0, 0., 0.
        for step, batch in enumerate(iter_bar):
            if is_dynamic:
                input_ids, input_mask, segment_ids, start_positions, end_positions = train_data.next_batch()
                if n_gpu == 1:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    start_positions = start_positions.to(device)
                    end_positions = end_positions.to(device)
            else:
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch

            if args.grad_noise:
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, cal_grad=True)
            else:
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if args.grad_noise or args.gs_noise:
                if args.grad_noise:
                    adv_grad = model.bert.embedding_output.grad
                    adv_loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, adv_grad, cal_grad=False, is_gs_noise=False)
                else:
                    adv_loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, adv_grad=None, cal_grad=False, is_gs_noise=True)
                if n_gpu > 1:
                    adv_loss = adv_loss.mean()  # mean() to average on multi-gpu.
                if args.fp16:
                    with amp.scale_loss(adv_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    adv_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            tr_step += 1
            total_loss += loss.item()
            mean_loss = total_loss / tr_step
            iter_bar.set_description("Train Step(%d / %d) (Mean loss=%5.5f) (loss=%5.5f)" %
                                     (global_step, num_train_step, mean_loss, loss.item()))

        logger.info("** ** * Saving file * ** **")
        model_checkpoint = "korquad_%d.bin" % (epoch)
        logger.info(model_checkpoint)
        output_model_file = os.path.join(args.output_dir,model_checkpoint)
        if n_gpu > 1:
            torch.save(model.module.state_dict(), output_model_file)
        else:
            torch.save(model.state_dict(), output_model_file)
        with open(os.path.join(args.output_dir, "ckpt_list.txt"), 'a') as fw:
            fw.write(model_checkpoint + '\n')
        epoch += 1


if __name__ == "__main__":
    main()
