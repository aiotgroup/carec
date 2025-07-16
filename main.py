import json
import argparse
from trainer import train
import matplotlib.pyplot as plt
import numpy as np

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)       # Converting argparse Namespace to a dict.
    args.update(param)      # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='基于WiFi动作识别数据的类增量学习研究')
    parser.add_argument('--config', type=str, default='./exps/der.json',
                        help='Json file of settings.')
    parser.add_argument('--experiment_name',default='实验名称:CIL实验')
    
    return parser


if __name__ == '__main__':
    main()
