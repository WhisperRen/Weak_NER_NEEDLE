import random
import json
import logging
from glob import glob

import pandas as pd
from tqdm import tqdm

from utils import setup_seed, IGNORE_CHARACTERS


ORIGIN_DATA_PATH = r'./origin/'
UNSUPERVISED_TRAIN_PATH = r'./unsupervised_train_sample.txt'
SUPERVISED_TRAIN_PATH = r'./supervised_train_sample.txt'
SUPERVISED_TEST_PATH = r'./supervised_test_sample.txt'
WEAK_SUPERVISED_TRAIN_PATH = r'./weak_supervised_train_sample.txt'
WEAK_SUPERVISED_VAL_PATH = r'./weak_supervised_val_sample.txt'

SUPERVISED_TRAIN_RATIO = 0.1
SUPERVISED_TEST_RATIO = 0.1
WEAK_SUPERVISED_TRAIN_RATIO = 0.7
WEAK_SUPERVISED_VAL_RATIO = 0.1

COVERAGE_RATE = 0.4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_weak_supervised_data():
    logger.info('-------------------Transfer supervised to weak-supervised data start-------------------')
    train_data = pd.read_csv(WEAK_SUPERVISED_TRAIN_PATH, names=['word', 'label'])
    val_data = pd.read_csv(WEAK_SUPERVISED_VAL_PATH, names=['word', 'label'])
    transfer_supervised_to_weak_supervised(train_data)
    transfer_supervised_to_weak_supervised(val_data)

    train_data.to_csv(WEAK_SUPERVISED_TRAIN_PATH, index=None, header=None)
    val_data.to_csv(WEAK_SUPERVISED_VAL_PATH, index=None, header=None)
    logger.info('------------------Transfer supervised to weak-supervised data done---------------------')


def transfer_supervised_to_weak_supervised(data):
    val_labels = data['label'].values
    entity_index = []
    i, j = 0, 0
    while i < len(val_labels) and j < len(val_labels):
        if val_labels[i] == 'O':
            i += 1
            continue
        cur_span = []
        if 'B-' in val_labels[i]:
            cur_span.append(i)
            j = i + 1
            while j < len(val_labels) and 'I-' in val_labels[j]:
                cur_span.append(j)
                j += 1
            entity_index.append(cur_span)
            i = j
    random.shuffle(entity_index)
    uncovered_index = entity_index[:int(len(entity_index) * (1 - COVERAGE_RATE))]
    for ind in uncovered_index:
        val_labels[ind] = 'O'
    data['label'] = val_labels


def split_sample(config):
    files = glob(ORIGIN_DATA_PATH + '*.txt')
    random.shuffle(files)
    file_num = len(files)
    n1 = int(file_num * SUPERVISED_TRAIN_RATIO)
    n2 = int(file_num * SUPERVISED_TEST_RATIO)
    n3 = int(file_num * WEAK_SUPERVISED_TRAIN_RATIO)
    supervised_train_files = files[:n1]
    supervised_test_files = files[n1: (n1 + n2)]
    weak_supervised_train_files = files[(n1 + n2): (n1 + n2 + n3)]
    weak_supervised_val_files = files[(n1 + n2 + n3):]

    # generate unsupervised corpus file
    logger.info('--------------------Generate unsupervised data file start----------------------')
    unsupervised_df = pd.DataFrame()
    for f in weak_supervised_train_files:
        temp = pd.read_csv(f, names=['word', 'label'])
        unsupervised_df = pd.concat([unsupervised_df, temp])
    unsupervised_df['word'] = unsupervised_df['word'].apply(lambda w: str(w))
    unsupervised_df = unsupervised_df[~unsupervised_df['word'].isin(IGNORE_CHARACTERS)].reset_index(drop=True)
    seg_points = _get_segment_points(config, unsupervised_df)
    with open(UNSUPERVISED_TRAIN_PATH, 'a', encoding='utf-8') as file:
        for i in tqdm(range(len(seg_points) - 1)):
            sentence = unsupervised_df[seg_points[i]: seg_points[i + 1]]['word'].values
            file.write(' '.join(sentence) + '\n')

    # merge files
    logger.info('--------------------Merge dataset files start----------------------')
    _merge_file(supervised_train_files, SUPERVISED_TRAIN_PATH)
    _merge_file(supervised_test_files, SUPERVISED_TEST_PATH)
    _merge_file(weak_supervised_train_files, WEAK_SUPERVISED_TRAIN_PATH)
    _merge_file(weak_supervised_val_files, WEAK_SUPERVISED_VAL_PATH)
    logger.info('--------------------Merge dataset files done----------------------')


def _get_segment_points(config, data_df):
    seg_points = [0]
    i = 0
    while True:
        if i + config.get('base_len') >= data_df.shape[0]:
            seg_points.append(data_df.shape[0])
            break
        if data_df.loc[i + config.get('base_len'), 'label'] == 'O':
            i += config.get('base_len')
            seg_points.append(i)
        else:
            i += 1
    return seg_points


def _merge_file(files, target_path):
    with open(target_path, 'a', encoding='utf-8') as file:
        for f in files:
            text = open(f, encoding='utf-8').read()
            file.write(text)


if __name__ == '__main__':
    setup_seed()
    with open(r'../config.json', 'r') as ff:
        config_ = json.load(ff).get('stage-1')

    # split dataset
    split_sample(config_)

    # transfer supervised to weak-supervised data according to coverage
    get_weak_supervised_data()
