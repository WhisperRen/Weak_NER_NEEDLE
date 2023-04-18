import os
import json
import logging
import bisect

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.quantization import quantize_dynamic
from transformers import BertTokenizer
from scipy.stats import binned_statistic
from tqdm import tqdm

from models import AlbertForNER
from utils import setup_seed, DataSet, DataSetCompleted, collate_fn


COMPLETED_TRAIN_DATA_NAME = r'completed_train_sample.csv'
CONFIDENCE_ESTIMATION_NAME = r'train_confidence_estimation.json'
TRAIN_SEG_POINTS_NAME = r'train_seg_points.json'
COMPLETED_VAL_DATA_NAME = r'completed_val_sample.csv'
VAL_SEG_POINTS_NAME = r'val_seg_points.json'

CRF_SCORE_INF = 10000
MAX_CONFIDENCE = 0.95

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_and_complete_tag(ner_model, dataset, data_loader, device):
    ner_model.eval()
    ner_model = quantize_dynamic(ner_model, {nn.Linear}, dtype=torch.qint8)
    ner_model.to(device)
    completed_tags = []
    for inputs, labels in tqdm(data_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = ner_model(**inputs)
            logits = outputs[0]
            pred_tags, _ = ner_model.crf.decode(logits, inputs['attention_mask'].to(torch.bool))

        true_tags = labels.cpu().numpy().tolist()
        for s_index, tags in enumerate(pred_tags):
            # remove padding tags
            true_tags[s_index] = true_tags[s_index][:len(tags)]
            # id to labels, and remove [CLS] [SEP]
            pre_tags_decode = list(map(lambda idx: dataset.id2label[idx], tags))[1: -1]
            true_tags_decode = list(map(lambda idx: dataset.id2label[idx], true_tags[s_index]))[1: -1]
            completed_tags.extend([pre_tags_decode[i] if t == 'O' else t for i, t in enumerate(true_tags_decode)])
    return completed_tags


def label_completion(config):
    if not os.path.exists(config.get('completed_data_path')):
        os.makedirs(config.get('completed_data_path'))
    tokenizer = BertTokenizer.from_pretrained(config.get('supervised_checkpoint_save_path'),
                                              do_lower_case=config.get('do_lower_case'))
    model = AlbertForNER.from_pretrained(config.get('supervised_checkpoint_save_path'))

    train_dataset = DataSet(config, data_path_key='weak_supervised_train_data_path')
    val_dataset = DataSet(config, data_path_key='weak_supervised_val_data_path')
    config['label_of_O'] = train_dataset.label2id['O']
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=config.get('batch_size'),
                                   shuffle=False,
                                   collate_fn=lambda d: collate_fn(d, tokenizer, config))
    val_data_loader = DataLoader(dataset=val_dataset,
                                 batch_size=config.get('batch_size'),
                                 shuffle=False,
                                 collate_fn=lambda d: collate_fn(d, tokenizer, config))
    completed_train_tags_list = predict_and_complete_tag(model,
                                                         train_dataset,
                                                         train_data_loader,
                                                         config.get('device'))
    train_df = train_dataset.data_df
    train_df = train_df[: len(completed_train_tags_list)]
    train_df['completed_label'] = completed_train_tags_list

    train_df.to_csv(os.path.join(config.get('completed_data_path'), COMPLETED_TRAIN_DATA_NAME), index=None)
    with open(os.path.join(config.get('completed_data_path'), TRAIN_SEG_POINTS_NAME), 'w') as f:
        json.dump({'seg_points': train_dataset.seg_points}, f)
    logger.info(f'Save completed train dataset to {config.get("completed_data_path")}')

    completed_val_tags_list = predict_and_complete_tag(model,
                                                       val_dataset,
                                                       val_data_loader,
                                                       config.get('device'))
    val_df = val_dataset.data_df
    val_df['completed_label'] = completed_val_tags_list
    val_df.to_csv(os.path.join(config.get('completed_data_path'), COMPLETED_VAL_DATA_NAME), index=None)
    with open(os.path.join(config.get('completed_data_path'), VAL_SEG_POINTS_NAME), 'w') as f:
        json.dump({'seg_points': val_dataset.seg_points}, f)
    logger.info(f'Save completed val dataset to {config.get("completed_data_path")}')


def confidence_estimation(config):
    with open(os.path.join(config.get('completed_data_path'), VAL_SEG_POINTS_NAME), 'r') as s:
        val_seg_points = json.load(s).get('seg_points')
    val_data = pd.read_csv(os.path.join(config.get('completed_data_path'), COMPLETED_VAL_DATA_NAME))
    tokenizer = BertTokenizer.from_pretrained(config.get('supervised_checkpoint_save_path'),
                                              do_lower_case=config.get('do_lower_case'))
    model = AlbertForNER.from_pretrained(config.get('supervised_checkpoint_save_path'))

    val_dataset = DataSetCompleted(config, val_data, val_seg_points)
    config['label_of_O'] = val_dataset.label2id['O']
    val_data_loader = DataLoader(dataset=val_dataset,
                                 batch_size=config.get('batch_size'),
                                 shuffle=False,
                                 collate_fn=lambda d: collate_fn(d, tokenizer, config))
    device = config.get('device')
    model.eval()
    model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.to(device)
    profile_val_data = []
    for inputs, labels in tqdm(val_data_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(**inputs)
            logits = outputs[0]
            pred_tags, crf_scores = model.crf.decode(logits, inputs['attention_mask'].to(torch.bool))

        true_tags = labels.cpu().numpy().tolist()
        batch_profile_data = []
        for s_index, tags_raw in enumerate(pred_tags):
            # remove [CLS] [SEP]
            tags = tags_raw[1: -1]
            n_token = len(tags)
            # remove padding tags and [CLS] [SEP]
            true_tags[s_index] = true_tags[s_index][: (n_token + 2)][1: -1]
            # normalize crf score
            crf_scores[s_index] = crf_scores[s_index] / n_token

            batch_profile_data.append((crf_scores[s_index], int(tags == true_tags[s_index])))

        profile_val_data.extend(batch_profile_data)

    # calculate bins
    profile_val_data.sort(key=lambda p: p[0])
    scores = [p for p, _ in profile_val_data]
    query_acc = [a for _, a in profile_val_data]
    bins = scores[::len(scores) // config.get('bins_num')]
    bins[0] = -CRF_SCORE_INF
    bins[-1] = CRF_SCORE_INF
    bin_means, bin_edges, _ = binned_statistic(scores, query_acc, statistic='mean', bins=bins)
    # confidence estimation for train data
    train_data = pd.read_csv(os.path.join(config.get('completed_data_path'), COMPLETED_TRAIN_DATA_NAME))
    with open(os.path.join(config.get('completed_data_path'), TRAIN_SEG_POINTS_NAME), 'r') as s:
        train_seg_points = json.load(s).get('seg_points')
    mismatch_rates = []
    for i in range(len(train_seg_points) - 1):
        sentence = train_data[train_seg_points[i]: train_seg_points[i + 1]]
        total_tokens = len(sentence)
        mismatched_tokens = sentence['label'].value_counts()['O']
        mismatch_rates.append(mismatched_tokens / total_tokens)
    train_dataset = DataSetCompleted(config, train_data, train_seg_points)
    config['label_of_O'] = train_dataset.label2id['O']
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=config.get('batch_size'),
                                   shuffle=False,
                                   collate_fn=lambda d: collate_fn(d, tokenizer, config))
    train_data_confidences = []
    i = 0
    for inputs, labels in tqdm(train_data_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(**inputs)
            logits = outputs[0]
            pred_tags, crf_scores = model.crf.decode(logits, inputs['attention_mask'].to(torch.bool))

        for s_index, tags_raw in enumerate(pred_tags):
            # remove [CLS] [SEP]
            n_token = len(tags_raw) - 2
            # normalize crf score
            crf_s = crf_scores[s_index] / n_token
            confidence_index = bisect.bisect_right(bin_edges, crf_s) - 1
            confidence = 1 - mismatch_rates[i] + mismatch_rates[i] * bin_means[confidence_index]
            train_data_confidences.append(min(confidence, MAX_CONFIDENCE))
            i += 1
    with open(os.path.join(config.get('completed_data_path'), CONFIDENCE_ESTIMATION_NAME), 'w') as c:
        json.dump({'confidence': train_data_confidences}, c)
    logger.info(f'Save confidence estimation result to {config.get("completed_data_path")}')


if __name__ == '__main__':
    setup_seed()
    with open(r'../config.json', 'r') as f:
        config_ = json.load(f).get('stage-2')

    config_['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    label_completion(config_)

    confidence_estimation(config_)
