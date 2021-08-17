import logging
from abc import abstractmethod

import torch
import torch.nn as nn
from tqdm import trange

from utils import log_line, log_dict, EarlyStopping

logger = logging.getLogger("train")


class Trainer:

    def __init__(self, model, config):

        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)

        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          self.learning_rate)

        self.save_path = config["save_path"]

        self.early_stopping = EarlyStopping(self.save_path)

        log_line(logger)
        logger.info("Parameters:")
        log_dict(logger, config)

    def train(self):
        try:
            for epoch in trange(self.epochs, desc='Epoch'):
                train_loss, val_loss, val_acc = self._train_epoch(epoch)

                log_line(logger)
                logger.info('EPOCH {} done : loss {}'.format(
                    epoch+1, train_loss))
                logger.info('DEV : loss {} - acc {}'.format(val_loss, val_acc))
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.stop:
                    break
        except KeyboardInterrupt:
            self.early_stopping.save_checkpoint()

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError


if __name__ == '__main__':
    trainer = Trainer()
