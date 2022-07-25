import argparse
import json
import yaml
import sys
import os

class Config(object):
    def __init__(self, dict:dict)->None:
        for k, v in dict.items():
            setattr(self, k, v)
    
    def update(self, args):
        for k, v in args.items():
            setattr(self, k, v)
        return self

parser = argparse.ArgumentParser(prog="General Machine Learning Framework",
                                 description="General ML framework for various use. It provides CLI interface with argparser and basic training and inference actions.")

# Basic config
parser.add_argument('mode', type=str, choices=['train', 'infer'], help='Configure ML types')
parser.add_argument('-s', '--silent', action='store_true', help='Skip logo if set')
parser.add_argument('--input_path', type=argparse.FileType('r'), metavar='INPUT_PATH')
parser.add_argument('--output_path', type=str, metavar='OUTPUT_PATH')
parser.add_argument('--checkpoint_path', type=str, metavar='CHECKPOINT_PATH')

# Common Hyperparameters
parser.add_argument('-l', '--lr', type=float, default=0.001, metavar='LEARNING_RATE', help='initial learning rate', dest='learning_rate')

# You can also provide config values by file. If file path is passed, hyperparameters before will be ignored.
parser.add_argument('--config_path', type=str, default='', metavar='CONFIG_PATH')
parser.add_argument('--config_type', choices=['xml', 'json', 'yaml'], default='yaml', metavar='CONFIG_TYPE')

def logo()->None:
    print('________                  _________                  ______  ')
    print('___  __ \___________________  ____/_________  __________  /_ ')
    print('__  / / /  _ \  _ \__  __ \  /    _  __ \  / / /  ___/_  __ \\')
    print('_  /_/ //  __/  __/_  /_/ / /___  / /_/ / /_/ // /__ _  / / /')
    print('/_____/ \___/\___/_  .___/\____/  \____/\__,_/ \___/ /_/ /_/')
    print('=============================================================')
    print('DeepCouch - Deep Learning Framework v0.0.1')
    print('Copyright 2022 Redcated')
    print('Licensed under the MIT License')
    print('=============================================================')

def load_config(path, type)->Config:
    """
    load config file into one object
    :param type:type(str)
    :return: Configuration object
    """
    config_file = open(path, 'rt')
    config = None
    if type == 'xml':
        parser.exit(message="XML is not supported yet.")
    elif type == 'json':
        config = json.load(config_file)
    elif type == 'yaml' or type == 'yml':
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
    else:
        print(f"Undefined extension : {type}")
    return Config(config)

def resolve_config(args, config:Config)->argparse.Namespace:
    """
    Assing properties form config file to args
    :param config: Configurations
    :return: args(Namespace)
    """
    # Validate config properties here.
    #    
    config.update(vars(args))
    return config

if __name__ == "__main__":
    args = parser.parse_args()
    # config file handle here
    if args.config_path:
        config = resolve_config(args=args, config=load_config(path=args.config_path, type=args.config_type))
    if not args.silent:
        logo()
    if args.mode == "train":
        pass
    elif args.mode == "infer":
        pass