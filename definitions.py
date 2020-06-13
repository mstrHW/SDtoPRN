import os
import logging
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from typing import (
    Dict,
    List,
    Optional,
    Tuple
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# LOG_FILE = os.path.join(ROOT_DIR, 'log.log')
# logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)


def path_join(left: str, right: str) -> str:
    return os.path.join(left, right)


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# DATA_DIR = 'D:/Datasets/TestRW'
DATA_DIR = path_join(ROOT_DIR, 'data')
BASE_FILES = path_join(DATA_DIR, 'base_files')
CLEARED_DIR = path_join(DATA_DIR, 'cleared')
MERGED_DIR = path_join(DATA_DIR, 'merged')
FINAL_DIR = path_join(DATA_DIR, 'final')
DATASETS_DIR = path_join(DATA_DIR, 'datasets')
VENSIM_MODELS_DIR = path_join(ROOT_DIR, 'vensim_models')
TF_MODELS_DIR = path_join(ROOT_DIR, 'tf_models')
IMAGES_DIR = path_join(ROOT_DIR, 'images')
EXPERIMENTS_DIR = path_join(ROOT_DIR, 'experiments')
FILE_NAMES = ['labresult.txt', 'holesterin.csv', 'predst.txt', 'oks-events.txt', 'eho18-all-fixed.csv']

make_directory(BASE_FILES)
make_directory(CLEARED_DIR)
make_directory(MERGED_DIR)
make_directory(FINAL_DIR)
make_directory(DATASETS_DIR)
make_directory(VENSIM_MODELS_DIR)
make_directory(TF_MODELS_DIR)
make_directory(IMAGES_DIR)
make_directory(EXPERIMENTS_DIR)
