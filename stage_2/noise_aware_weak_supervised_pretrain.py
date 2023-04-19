import logging
import os
import json

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from models import AlbertForNER
from label_completion_and_confidence_estimation import (
    COMPLETED_TRAIN_DATA_NAME,
    CONFIDENCE_ESTIMATION_NAME,
    TRAIN_SEG_POINTS_NAME
)
from utils import setup_seed, DataSet, DataSetWeighted, collate_fn_weighted

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def noise_aware_model_train(config):
    if not os.path.exists(config.get('stage2_final_checkpoint_path')):
        os.makedirs(config.get('stage2_final_checkpoint_path'))

    # load supervised data
    supervised_dataset = DataSet(config, data_path_key='supervised_train_data_path')
    supervised_df = supervised_dataset.data_df
    supervised_df.rename(columns={'label': 'completed_label'}, inplace=True)
    supervised_seg_points = supervised_dataset.seg_points
    supervised_weights = [1.] * len(supervised_seg_points)

    # load completed weak-supervised data
    with open(os.path.join(config.get('completed_data_path'), CONFIDENCE_ESTIMATION_NAME), 'r') as f:
        completed_weights = json.load(f).get('confidence')
    with open(os.path.join(config.get('completed_data_path'), TRAIN_SEG_POINTS_NAME), 'r') as f:
        completed_seg_points = json.load(f).get('seg_points')
    completed_df = pd.read_csv(os.path.join(config.get('completed_data_path'), COMPLETED_TRAIN_DATA_NAME))
    completed_df.drop(columns='label', inplace=True)

    # concat supervised and weak-supervised data
    completed_df = pd.concat([completed_df, supervised_df])
    completed_seg_points.extend([s + completed_seg_points[-1] for s in supervised_seg_points][1:])
    completed_weights.extend(supervised_weights)
    dataset = DataSetWeighted(config=config,
                              data_df=completed_df,
                              seg_points=completed_seg_points,
                              weights=completed_weights)
    config['label_of_O'] = dataset.label2id['O']
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.get('batch_size'),
                             shuffle=True,
                             collate_fn=lambda d: collate_fn_weighted(d, tokenizer, config))
    tokenizer = BertTokenizer.from_pretrained(config.get('pretrained_weights_stage1'),
                                              do_lower_case=config.get('do_lower_case'))
    model = AlbertForNER.from_pretrained(config.get('pretrained_weights_stage1'),
                                         **{'num_labels': config.get('num_labels')})
    device = config.get('device')
    model.to(device)
    optimizer = optim.AdamW([{'params': model.albert.parameters(), 'lr': config.get('lr_pretrained')},
                             {'params': model.classifier.parameters(), 'lr': config.get('lr_crf')},
                             {'params': model.crf.parameters(), 'lr': config.get('lr_crf')}])
    model.train()
    for i in range(config.get('epoch_num_weak')):
        for j, (inputs_, labels_, weights_) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            weights_ = weights_.to(device)
            outputs = model(**inputs_, labels=labels_, crf_reduction='none')
            nll = outputs[0]
            null = -(1 - (-nll).exp()).log()
            if torch.isnan(null).any() or torch.isinf(null).any():
                nl = (1 - (-nll).exp())
                nl = nl + (nl < 1e-4).to(nl).detach() * (1e-4 - nl).detach()
                null = -nl.log()
            loss = (nll * weights_ + null * (1 - weights_)).sum()
            loss.backward()
            optimizer.step()
            if j % 100 == 0:
                logger.info(f'Epoch {i}, batch {j}, train loss is : {loss.item()}')
    model.eval()
    # save checkpoints
    tokenizer.save_vocabulary(config.get('stage2_final_checkpoint_path'))
    model.save_pretrained(config.get('stage2_final_checkpoint_path'))
    logger.info('Save model weights success')


if __name__ == '__main__':
    setup_seed()
    with open(r'../config.json', 'r') as f:
        config_ = json.load(f).get('stage-2')

    config_['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    noise_aware_model_train(config=config_)
