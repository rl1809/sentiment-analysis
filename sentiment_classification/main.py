import os
import logging

import numpy as np

from config import load_config
from trainer import BertTrainer, RegularTrainer, TMixTrainer
from models.bert_model import ClassificationBert, MixText
from models.cnn_model import TextCNN
from models.rnn_model import SimpleLSTM
from utils import add_file_handler, load_csv_file
from dataset import RegularDataset

logger = logging.getLogger("train")

def main(args):

    model = MixText(num_classes=4, mix_option=True)
    trainer = TMixTrainer(config=vars(args), model=model)

    trainer.train()
    trainer.test()

if __name__ == '__main__':
    args = load_config()
    directory = os.path.dirname(args.save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file = os.path.join(args.save_path, 'training.log')
    add_file_handler(logger, log_file)
    main(args)
   

