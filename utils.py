import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence


SEED = 99

IGNORE_CHARACTERS = ('', ' ', '\n', '\r\n', '\t', '\u2003', '\u2002', '\u3000', '̂')


def setup_seed():
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # some cudnn methods can be random even after fixing the seed unless force it to be deterministic
    torch.backends.cudnn.deterministic = True


def collate_fn(batch, tokenizer, config):
    """
    :param batch: list of tokens and labels in batch
    :param tokenizer: tokenizer
    :param config: config
    :return: aligned_inputs, aligned_labels
    """
    inputs, labels = [], []
    for input_, label_ in batch:
        inputs.append(input_)
        labels.append(torch.from_numpy(np.array(label_)))
    # pad and truncate tokens
    aligned_inputs = tokenizer(inputs,
                               padding='longest',
                               max_length=config.get('max_len'),
                               truncation=True,
                               is_split_into_words=True,
                               return_tensors='pt')
    # pad labels
    aligned_labels = pad_sequence(labels, batch_first=True, padding_value=config.get('label_of_O'))

    # truncation ([CLS] tokens[:, :max_len-2] [SEP])
    if aligned_labels.shape[1] + 2 != aligned_inputs['input_ids'].shape[1]:
        aligned_labels = aligned_labels[:, :(config.get('max_len') - 2)]

    pad_tensor = torch.full((aligned_labels.shape[0], 1), config.get('label_of_O'))
    aligned_labels = torch.cat((aligned_labels, pad_tensor), dim=1)
    aligned_labels = torch.cat((pad_tensor, aligned_labels), dim=1)

    return aligned_inputs, aligned_labels.to(torch.long)


def collate_fn_weighted(batch, tokenizer, config):
    """
    :param batch: list of tokens and labels in batch
    :param tokenizer: tokenizer
    :param config: config
    :return: aligned_inputs, aligned_labels, weights
    """
    inputs, labels, weights = [], [], []
    for input_, label_, weight_ in batch:
        inputs.append(input_)
        labels.append(torch.from_numpy(np.array(label_)))
        weights.append(weight_)
    # pad and truncate tokens
    aligned_inputs = tokenizer(inputs,
                               padding='longest',
                               max_length=config.get('max_len'),
                               truncation=True,
                               is_split_into_words=True,
                               return_tensors='pt')
    # pad labels
    aligned_labels = pad_sequence(labels, batch_first=True, padding_value=config.get('label_of_O'))

    # truncation ([CLS] tokens[:, :max_len-2] [SEP])
    if aligned_labels.shape[1] + 2 != aligned_inputs['input_ids'].shape[1]:
        aligned_labels = aligned_labels[:, :(config.get('max_len') - 2)]

    pad_tensor = torch.full((aligned_labels.shape[0], 1), config.get('label_of_O'))
    aligned_labels = torch.cat((aligned_labels, pad_tensor), dim=1)
    aligned_labels = torch.cat((pad_tensor, aligned_labels), dim=1)

    return aligned_inputs, aligned_labels.to(torch.long), torch.from_numpy(np.array(weights))


def get_labels(config):
    label_path = config.get('label_path')
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f'label.txt does not exist in path {label_path}')
    labels = pd.read_csv(label_path, names=['label', 'id'])
    return dict(labels.values)


class DataSet(data.Dataset):

    def __init__(self, config, data_path_key):
        super(DataSet, self).__init__()
        self.base_len = config.get('base_len')
        data_path = config.get(data_path_key)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f'data file does not exist in {data_path}')
        self.data_df = pd.read_csv(data_path, names=['word', 'label'])
        self.data_df = self.data_df[~self.data_df['word'].isin(IGNORE_CHARACTERS)].dropna().reset_index(drop=True)
        self.data_df['word'] = self.data_df['word'].apply(lambda w: str(w))
        self.label2id = get_labels(config)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.seg_points = [0]
        # get segment points of sentences
        self._get_segment_points()

    def _get_segment_points(self):
        """
        get segment points of sentence, to avoid separating ont entity into two sentences
        :return:
        """
        i = 0
        while True:
            if i + self.base_len >= self.data_df.shape[0]:
                self.seg_points.append(self.data_df.shape[0])
                break
            if self.data_df.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                self.seg_points.append(i)
            else:
                i += 1

    def __len__(self):
        return len(self.seg_points) - 1

    def __getitem__(self, index):
        sentence_df = self.data_df[self.seg_points[index]: self.seg_points[index + 1]]
        sentence = sentence_df['word'].values.tolist()
        token_labels = [self.label2id.get(label, self.label2id['O']) for label in sentence_df['label'].values]
        return sentence, token_labels


class DataSetCompleted(data.Dataset):

    def __init__(self, config, data_df, seg_points):
        super(DataSetCompleted, self).__init__()
        self.base_len = config.get('base_len')
        self.data_df = data_df
        self.data_df = self.data_df[~self.data_df['word'].isin(IGNORE_CHARACTERS)].dropna().reset_index(drop=True)
        self.data_df['word'] = self.data_df['word'].apply(lambda w: str(w))
        self.label2id = get_labels(config)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.seg_points = seg_points

    def __len__(self):
        return len(self.seg_points) - 1

    def __getitem__(self, index):
        sentence_df = self.data_df[self.seg_points[index]: self.seg_points[index + 1]]
        sentence = sentence_df['word'].values.tolist()
        token_labels = [self.label2id.get(label, self.label2id['O']) for label in sentence_df['completed_label'].values]
        return sentence, token_labels


class DataSetWeighted(data.Dataset):

    def __init__(self, config, data_df, seg_points, weights):
        super(DataSetWeighted, self).__init__()
        self.base_len = config.get('base_len')
        self.data_df = data_df
        self.data_df = self.data_df[~self.data_df['word'].isin(IGNORE_CHARACTERS)].dropna().reset_index(drop=True)
        self.data_df['word'] = self.data_df['word'].apply(lambda w: str(w))
        self.label2id = get_labels(config)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.seg_points = seg_points
        self.weights = weights

    def __len__(self):
        return len(self.seg_points) - 1

    def __getitem__(self, index):
        sentence_df = self.data_df[self.seg_points[index]: self.seg_points[index + 1]]
        sentence = sentence_df['word'].values.tolist()
        token_labels = [self.label2id.get(label, self.label2id['O']) for label in sentence_df['completed_label'].values]
        return sentence, token_labels, self.weights[index]

