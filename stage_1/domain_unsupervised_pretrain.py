import json
import os

import torch
from transformers import (
    BertTokenizer,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    AlbertForMaskedLM,
    Trainer,
    TrainingArguments
)

from utils import setup_seed, SEED


if __name__ == '__main__':
    setup_seed()
    with open(r'../config.json', 'r') as f:
        config = json.load(f).get('stage-1')

    if not os.path.exists(config.get('checkpoint_save_path')):
        os.makedirs(config.get('checkpoint_save_path'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.get('pretrained_weights_path'),
                                              do_lower_case=config.get('do_lower_case'))
    dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                    file_path=config.get('unsupervised_train_data_path'),
                                    block_size=config.get('max_len'))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=config.get('mlm_probability'))
    mlm = AlbertForMaskedLM.from_pretrained(config.get('pretrained_weights_path')).to(device)

    training_args = TrainingArguments(
        output_dir=config.get('checkpoint_save_path'),
        overwrite_output_dir=True,
        learning_rate=config.get('lr'),
        num_train_epochs=config.get('epoch_num'),
        per_device_train_batch_size=config.get('batch_size'),
        save_strategy='epoch',
        seed=SEED
    )

    trainer = Trainer(
        model=mlm,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()

    tokenizer.save_vocabulary(r'../stage_2/results_from_stage1')
