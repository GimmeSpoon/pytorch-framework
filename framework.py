import argparse
import string
import pandas as pd
import numpy as np
import torch
from torch import nn

from torch import float32

parser = argparse.ArgumentParser(prog="General Machine Learning Framework",
                                 description="General ML framework for various use. It provides CLI interface with argparser and basic training and inference actions.")

""" Basic config """
parser.add_argument('mode')
parser.add_argument('--input_path', type=string, metavar='INPUT_PATH')
parser.add_argument('--output_path', type=string, metavar='OUTPUT_PATH')
parser.add_argument('--checkpoint_path', type=string, metavar='CHECKPOINT_PATH')

""" Hyperparameters """
parser.add_argument('-l', '--lr', type=float32, default=0.001, metavar='LEARNING_RATE', help='initial learning rate', dest='learning_rate')

""" You can also provide config values by file. If file path is passed, hyperparameters before will be ignored."""
parser.add_argument('--config_path', type=string, default='', metavar='CONFIG_PATH')
parser.add_argument('--config_type', choices=['xml', 'json', 'yaml'], default='yaml', metavar='CONFIG_TYPE')

""" config file handle here """

"""------------------------ """

args = parser.parse_args()

""" Data PreProcessing """

""" ------------------ """

"""              Train              """

""" ------------------------------- """

"""              Infer              """

""" ------------------------------- """