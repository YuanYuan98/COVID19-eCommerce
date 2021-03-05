import os
import setproctitle
import argparse
from Trainer import *
import torch
import random
import numpy as np
from config import Config
from absl import logging
from absl import app as app_google
import pandas as pd
from prediction_base import *


def parse_args():
    parser = argparse.ArgumentParser(description="covid")
    parser.add_argument('--gpu', type=str, default='3',
                        help='GPU.')
    parser.add_argument('--hidden_size', type=int, default=2,
                        help='hidden size')
    parser.add_argument('--folder_file', type=str, default='',
                        help='')
    parser.add_argument('--file', type=str, default='data_origin',
                        help='')
    parser.add_argument('--seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--epoch_num', type=int, default=1000,
                        help='epoch num')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop')
    parser.add_argument('--model', type=int, default=1,
                        help='model selection')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0,
                        help='teacher_forcing_ratio')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience to reduce lr')
    parser.add_argument('--cases', type=str, default='cases',
                        help='cases')
    parser.add_argument('--src', type=int, default=10,
                        help='days_before for prediction')
    parser.add_argument('--trg', type=int, default=7,
                        help='number of days to be predicted')
    parser.add_argument('--b_type', type=int, default=2,
                        help='behavior type')
    parser.add_argument('--p_type', type=int, default=1,
                        help='product type')
    parser.add_argument('--province', type=str, default='Anhui',
                        help='province')
    parser.add_argument('--city', type=int, default=0,
                        help='whether train model for each provvince')
    parser.add_argument('--flag', type=int, default=0,
                        help='negative or positive')
    return parser.parse_args()

def seed_set():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

args = parse_args()

def main(argv):

    
    setproctitle.setproctitle("covid-19_app")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    result = []
    args.city=0
    args.dropout=0.3
    args.src=14
    args.trg = 7

    # negative product
    args.flag=1

    args.seed=100
    seed_set()
    app = Trainer(args,logging)
    mae,info = app.train_step()
    

if __name__ == '__main__':
    app_google.run(main)

