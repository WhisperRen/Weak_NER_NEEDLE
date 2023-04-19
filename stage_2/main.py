import json

import torch

from stage_2.supervised_ner_model_train import supervised_ner_model_train
from stage_2.label_completion_and_confidence_estimation import label_completion, confidence_estimation
from stage_2.noise_aware_weak_supervised_pretrain import noise_aware_model_train
from utils import setup_seed


if __name__ == '__main__':
    setup_seed()
    with open(r'../config.json', 'r') as f:
        config = json.load(f).get('stage-2')
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # initialize ner model
    supervised_ner_model_train(config)

    # label completion and confidence estimation
    label_completion(config)
    confidence_estimation(config)

    # noise aware continual pretrain
    noise_aware_model_train(config)
