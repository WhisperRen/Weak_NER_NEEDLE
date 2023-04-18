import os
import json
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from models import AlbertForNER
from utils import setup_seed, DataSet, collate_fn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def supervised_ner_model_train(config):
    if not os.path.exists(config.get('supervised_checkpoint_save_path')):
        os.makedirs(config.get('supervised_checkpoint_save_path'))
    device = config.get('device')
    tokenizer = BertTokenizer.from_pretrained(config.get('pretrained_weights_stage1'),
                                              do_lower_case=config.get('do_lower_case'))
    dataset = DataSet(config, data_path_key='supervised_train_data_path')
    config['label_of_O'] = dataset.label2id['O']
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.get('batch_size'),
                             shuffle=True,
                             collate_fn=lambda d: collate_fn(d, tokenizer, config))
    model = AlbertForNER.from_pretrained(config.get('pretrained_weights_stage1'),
                                         **{'num_labels': config.get('num_labels')})
    model.to(device)
    optimizer = optim.AdamW([{'params': model.albert.parameters(), 'lr': config.get('lr_pretrained')},
                             {'params': model.classifier.parameters(), 'lr': config.get('lr_crf')},
                             {'params': model.crf.parameters(), 'lr': config.get('lr_crf')}])
    model.train()
    for i in range(config.get('epoch_num_supervised')):
        for j, (inputs_, labels_) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            outputs = model(**inputs_, labels=labels_)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            if j % 50 == 0:
                logger.info(f'Epoch {i}, batch {j}, train loss is: {loss.item()}')
    model.eval()
    # save checkpoint
    tokenizer.save_vocabulary(config.get('supervised_checkpoint_save_path'))
    model.save_pretrained(config.get('supervised_checkpoint_save_path'))
    logger.info('Save model weights success')


if __name__ == '__main__':
    setup_seed()
    with open(r'../config.json', 'r') as f:
        config_ = json.load(f).get('stage-2')

    config_['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    supervised_ner_model_train(config_)
