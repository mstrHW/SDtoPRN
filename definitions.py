import os

from typing import (
    Dict,
    List,
    Optional,
    Tuple
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def path_join(left: str, right: str) -> str:
    return os.path.join(left, right)


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


DATA_DIR = 'D:/Datasets/TestRW'
BASE_FILES = path_join(DATA_DIR, 'base_files')
CLEARED_DIR = path_join(DATA_DIR, 'cleared')
MERGED_DIR = path_join(DATA_DIR, 'merged')
FINAL_DIR = path_join(DATA_DIR, 'final')
DATASETS_DIR = path_join(DATA_DIR, 'datasets')
FILE_NAMES = ['labresult.txt', 'holesterin.csv', 'predst.txt', 'oks-events.txt', 'eho18-all-fixed.csv']

make_directory(BASE_FILES)
make_directory(CLEARED_DIR)
make_directory(MERGED_DIR)
make_directory(FINAL_DIR)
make_directory(DATASETS_DIR)
