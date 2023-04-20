import os
import json
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score

from models import AlbertForNER
from utils import setup_seed, DataSet, collate_fn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def supervised_ner_model_train(config):
    if not os.path.exists(config.get('checkpoint_save_path')):
        os.makedirs(config.get('checkpoint_save_path'))
    device = config.get('device')
    tokenizer = BertTokenizer.from_pretrained(config.get('pretrained_weights_stage2'),
                                              do_lower_case=config.get('do_lower_case'))
    dataset = DataSet(config, data_path_key='supervised_train_data_path')
    config['label_of_O'] = dataset.label2id['O']
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.get('batch_size'),
                             shuffle=True,
                             collate_fn=lambda d: collate_fn(d, tokenizer, config))
    dataset_test = DataSet(config, data_path_key='supervised_test_data_path')
    data_loader_test = DataLoader(dataset=dataset_test,
                                  batch_size=config.get('batch_size'),
                                  shuffle=False,
                                  collate_fn=lambda d: collate_fn(d, tokenizer, config))

    model = AlbertForNER.from_pretrained(config.get('pretrained_weights_stage2'),
                                         **{'num_labels': config.get('num_labels')})
    model.to(device)
    optimizer = optim.AdamW([{'params': model.albert.parameters(), 'lr': config.get('lr_pretrained')},
                             {'params': model.classifier.parameters(), 'lr': config.get('lr_crf')},
                             {'params': model.crf.parameters(), 'lr': config.get('lr_crf')}])
    f1_scores = []
    for i in range(config.get('epoch_num')):
        model.train()
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
        # evaluate
        model.eval()
        pre_tags_decode, true_tags_decode = [], []
        for inputs_, labels_ in tqdm(data_loader_test):
            with torch.no_grad():
                inputs_ = inputs_.to(device)
                outputs = model(**inputs_)
                logits = outputs[0]
                pred_tags, _ = model.crf.decode(logits, inputs_['attention_mask'].to(torch.bool))

            true_tags = labels_.cpu().numpy().tolist()
            for s_index, tags in enumerate(pred_tags):
                # remove padding tags
                true_tags[s_index] = true_tags[s_index][:len(tags)]
                # id to labels, and remove [CLS] [SEP]
                pre_tags_decode.append(list(map(lambda idx: dataset_test.id2label[idx], tags))[1: -1])
                true_tags_decode.append(list(map(lambda idx: dataset_test.id2label[idx], true_tags[s_index]))[1: -1])
        logger.info(f'Epoch {i} validation report:')
        logger.info(classification_report(y_true=true_tags_decode, y_pred=pre_tags_decode))
        f1_scores.append(f1_score(y_true=true_tags_decode, y_pred=pre_tags_decode, average='macro'))

    # save checkpoints
    model.eval()
    tokenizer.save_vocabulary(config.get('checkpoint_save_path'))
    model.save_pretrained(config.get('checkpoint_save_path'))
    logger.info('Save model weights success')

    # save f1 scores in f1_scores.json
    with open(os.path.join(config.get('checkpoint_save_path'), 'f1_scores.json'), 'w') as f:
        json.dump({'f1_scores': f1_scores}, f)
    logger.info(f'Save f1_scores.json to {config.get("checkpoint_save_path")}')


if __name__ == '__main__':
    setup_seed()
    with open(r'../config.json', 'r') as f:
        config_ = json.load(f).get('stage-3')
    config_['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    supervised_ner_model_train(config_)
