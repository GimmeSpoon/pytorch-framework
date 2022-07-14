import argparse
import json
import yaml
import sys
import os
from torch import float32

parser = argparse.ArgumentParser(prog="General Machine Learning Framework",
                                 description="General ML framework for various use. It provides CLI interface with argparser and basic training and inference actions.")

# Basic config
parser.add_argument('mode', type=str, choices=['dl', 'dt'], help='Configure ML types')
parser.add_argument('--input_path', type=argparse.FileType('r'), metavar='INPUT_PATH')
parser.add_argument('--output_path', type=str, metavar='OUTPUT_PATH')
parser.add_argument('--checkpoint_path', type=str, metavar='CHECKPOINT_PATH')

# Model structure


# Common Hyperparameters
parser.add_argument('-l', '--lr', type=float, default=0.001, metavar='LEARNING_RATE', help='initial learning rate', dest='learning_rate')

# You can also provide config values by file. If file path is passed, hyperparameters before will be ignored.
parser.add_argument('--config_path', type=str, default='', metavar='CONFIG_PATH')
parser.add_argument('--config_type', choices=['xml', 'json', 'yaml'], default='yaml', metavar='CONFIG_TYPE')

""" load config file into one object """
def load_config(path, type):
    config_file = open(path, 'rt')
    config = None
    if type == 'xml':
        parser.exit(message="XML is not supported yet.")
    elif type == 'json':
        config = json.load(config_file)
    elif type == 'yaml':
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
    return config

""" Assing properties form config file to args"""
def resolve_config(args, config):
    # Validate config properties here.
    #    
    config.update(vars(args))
    args = config
    return args

if __name__ == "__main__":
    args = parser.parse_args()
    # config file handle here
    if args.config_path:
        resolve_config(args=args, config=load_config(path=args.config_path, type=args.config_type))
        print(args)
    if args.mode == "dl":
        pass
    elif args.mode == "dt":
        pass