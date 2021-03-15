from argparse import ArgumentParser
import sys

def get_default_args():
    parser = ArgumentParser(description='setting for RotRqCnn', usage='%(prog)s [-h] [--option VALUE]')
    parser.add_argument('-d', '--dataset', type=str, default='mnist')
    parser.add_argument('--ensemble_num', type=int, default=1, help='Number of ensemble member')
    parser.add_argument('--train_epoch', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--get_data_only', type=bool, default=True, help='get the data or trainning')
    args = parser.parse_args()
    return args

