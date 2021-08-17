import os
import logging

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report


logger = logging.getLogger("train")


def log_line(logger):
    logger.info('-'*100)


def add_file_handler(logger, output_file):
    file_handler = logging.FileHandler(output_file, 'w',  encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)-15s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setStream(tqdm)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def log_dict(logger, dictionary):
    for key, value in dictionary.items():
        logger.info('- {}: {}'.format(key, value))


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, save_path, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.stop = False
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss, model):

        if not self.best_loss:
            self.best_loss = val_loss
            self.best_model = model
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_model = model
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            logger.info(
                f"""BAD EPOCHS: Early stopping counter
                {self.counter} of {self.patience}"""
                )
            if self.counter >= self.patience:
                self.stop = True
                self.save_checkpoint()

    def save_checkpoint(self):
        """Save best checkpoint"""
        directory = os.path.dirname(self.save_path)
        model_file = os.path.join(directory, 'best_model.pt')

        log_line(logger)
        logger.info('Exiting from training early.')
        logger.info("Saving model ...")
        torch.save(self.best_model.state_dict(), model_file)
        logger.info("Done")


def print_result(y_true, y_pred, label_names):
    logger.info("Results:")
    report = classification_report(y_true, y_pred, target_names=label_names)
    logger.info(f'\n{report}')
    log_line(logger)


def load_csv_file(filepath):
    dataframe = pd.read_csv(filepath)
    contents = dataframe['content'].values
    labels = dataframe['label'].values
    return contents, labels
