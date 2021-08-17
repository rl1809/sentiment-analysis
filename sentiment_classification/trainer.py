import os
import logging

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from base.base_trainer import Trainer
from dataset import get_dataloader, get_bert_dataloader
from utils import log_line, print_result, load_csv_file

logger = logging.getLogger("train")

DATA_FILE_PATH = '../data/preprocess/'


class BertTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(BertTrainer, self).__init__(*args, **kwargs)

        train_contents, train_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'train.csv'))
        val_contents, val_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'val.csv'))
        test_contents, test_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'test.csv'))

        self.train_loader = get_bert_dataloader(train_contents, train_labels)
        self.test_loader = get_bert_dataloader(test_contents, test_labels)
        self.val_loader = get_bert_dataloader(val_contents, val_labels)

        self.label_names = np.unique(train_labels)

        log_line(logger)
        logger.info("Data: {} train + {} dev + {} test samples".format(len(train_contents),
                                                                       len(val_contents),
                                                                       len(test_contents)))

    def _train_epoch(self, epoch):
        self.model.train()

        train_losses = []
        number_of_iters = len(self.train_loader)
        div = number_of_iters//10
        log_line(logger)
        for step, batch in enumerate(tqdm(self.train_loader, desc='iter')):

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            label = batch['label'].to(self.device)

            output = self.model(input_ids, attention_mask)

            loss = self.criterion(output, label).mean()
            self.optimizer.zero_grad()
            loss.backward()
            train_losses.append(loss.item())
            self.optimizer.step()
            if (step+1) % div == 0:
                logger.info('epoch {} - iter {}/{} - loss {}'.format(epoch+1,
                                                                     step,
                                                                     number_of_iters,
                                                                     loss.item()))
        train_loss = sum(train_losses)/len(train_losses)
        val_loss, val_acc = self._valid_epoch()

        return train_loss, val_loss, val_acc

    def _valid_epoch(self):
        self.model.eval()
        total = 0
        correct = 0
        val_losses = []
        tqdm.write("Start evaluate in validate data")
        with torch.no_grad():

            for step, batch in enumerate(tqdm(self.val_loader, desc='iter')):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)

                output = self.model(input_ids, attention_mask)

                _, predicted = torch.max(output.data, 1)
                total += label.shape[0]
                correct += (predicted == label).sum().item()

                loss = self.criterion(output, label).mean()
                val_losses.append(loss.item())
            val_acc = correct/total
            val_loss = sum(val_losses)/len(val_losses)
        return val_loss, val_acc

    def test(self):
        log_line(logger)
        logger.info('Testing using best model ...')
        model_path = os.path.join(self.save_path, 'best_model.pt')
        logger.info('Loading file {}'.format(model_path))
        model = self.model.load_state_dict(torch.load(model_path))
        text_contents = []
        predictions = []
        labels = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader, desc='iter')):
                text_content = batch['text_content']
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)

                output = model(input_ids, attention_mask)

                _, prediction = torch.max(output.data, 1)

                text_contents += text_content
                labels += label.tolist()
                predictions += prediction.tolist()
        data = zip(text_contents, labels, predictions)
        df = pd.DataFrame(data, columns=['Content', 'Label', 'Predict'])
        df.to_csv(os.path.join(self.save_path, 'test.csv'))

        print_result(labels, predictions, self.label_names)


class RegularTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(RegularTrainer, self).__init__(*args, **kwargs)

        train_contents, train_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'train.csv'))
        val_contents, val_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'val.csv'))
        test_contents, test_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'test.csv'))

        self.train_loader = get_dataloader(train_contents, train_labels)
        self.test_loader = get_dataloader(test_contents, test_labels)
        self.val_loader = get_dataloader(val_contents, val_labels)

        self.label_names = np.unique(train_labels)

        log_line(logger)
        logger.info("Data: {} train + {} dev + {} test samples".format(len(train_contents),
                                                                       len(val_contents),
                                                                       len(test_contents)))

    def _train_epoch(self, epoch):
        self.model.train()

        train_losses = []
        number_of_iters = len(self.train_loader)
        div = number_of_iters//10
        log_line(logger)
        for step, batch in enumerate(tqdm(self.train_loader, desc='iter')):

            content = batch['content'].to(self.device)
            label = batch['label'].to(self.device)

            output = self.model(content)

            loss = self.criterion(output, label).mean()
            self.optimizer.zero_grad()
            loss.backward()
            train_losses.append(loss.item())
            self.optimizer.step()
            if (step+1) % div == 0:
                logger.info('epoch {} - iter {}/{} - loss {}'.format(epoch+1,
                                                                     step,
                                                                     number_of_iters,
                                                                     loss.item()))
        train_loss = sum(train_losses)/len(train_losses)
        val_loss, val_acc = self._valid_epoch()

        return train_loss, val_loss, val_acc

    def _valid_epoch(self):
        self.model.eval()
        total = 0
        correct = 0
        val_losses = []
        tqdm.write("Start evaluate in validate data")
        with torch.no_grad():

            for step, batch in enumerate(tqdm(self.val_loader, desc='iter')):
                content = batch['content'].to(self.device)
                label = batch['label'].to(self.device)

                output = self.model(content)

                _, predicted = torch.max(output.data, 1)
                total += label.shape[0]
                correct += (predicted == label).sum().item()

                loss = self.criterion(output, label).mean()
                val_losses.append(loss.item())
            val_acc = correct/total
            val_loss = sum(val_losses)/len(val_losses)
        return val_loss, val_acc

    def test(self):
        log_line(logger)
        logger.info('Testing using best model ...')
        model_path = os.path.join(self.save_path, 'best_model.pt')
        logger.info('Loading file {}'.format(model_path))
        model = self.model.load_state_dict(torch.load(model_path))
        predictions = []
        labels = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader, desc='iter')):

                content = batch['content'].to(self.device)
                label = batch['label'].to(self.device)

                output = model(content)

                _, prediction = torch.max(output.data, 1)

                labels += label.tolist()
                predictions += prediction.tolist()

        print_result(labels, predictions, self.label_names)


def mixup_cross_entropy(criterion, pred, target, target2, lam):
    return lam*criterion(pred, target) + (1-lam)*criterion(pred, target2)


class TMixTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(TMixTrainer, self).__init__(*args, **kwargs)

        train_contents, train_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'train.csv'))
        val_contents, val_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'val.csv'))
        test_contents, test_labels = load_csv_file(
            os.path.join(DATA_FILE_PATH, 'test.csv'))

        self.train_loader = get_bert_dataloader(train_contents, train_labels)
        self.test_loader = get_bert_dataloader(test_contents, test_labels)
        self.val_loader = get_bert_dataloader(val_contents, val_labels)

        self.alpha = 0.5
        self.layer_list = [7, 9, 12]

        self.label_names = np.unique(train_labels)

        log_line(logger)
        logger.info("Data: {} train + {} dev + {} test samples".format(len(train_contents),
                                                                       len(val_contents),
                                                                       len(test_contents)))

    def _train_epoch(self, epoch):
        self.model.train()

        train_losses = []
        number_of_iters = len(self.train_loader)
        div = number_of_iters//10
        log_line(logger)
        for step, batch in enumerate(tqdm(self.train_loader, desc='iter')):
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            label = batch['label']
            indexes = torch.randperm(input_ids.size()[0])
            input_ids2, attention_mask2 = input_ids[indexes], attention_mask[indexes]
            label2 = label[indexes]
            input_ids, attention_mask = input_ids.to(
                self.device), attention_mask.to(self.device)
            input_ids2, attention_mask2 = input_ids2.to(
                self.device), attention_mask2.to(self.device)
            label, label2 = label.to(self.device), label2.to(self.device)

            lam = np.random.beta(self.alpha, self.alpha)
            mix_layer = np.random.choice(self.layer_list)
            mix_layer = mix_layer - 1
            output = self.model(input_ids, attention_mask, input_ids2,
                                attention_mask2, lam=lam, mix_layer=mix_layer)
            loss = mixup_cross_entropy(
                self.criterion, output, label, label2, lam)

            output = self.model(input_ids, attention_mask,
                                input_ids2, attention_mask2)

            loss = self.criterion(output, label).mean()
            self.optimizer.zero_grad()
            loss.backward()
            train_losses.append(loss.item())
            self.optimizer.step()
            if (step+1) % div == 0:
                logger.info('epoch {} - iter {}/{} - loss {}'.format(epoch+1,
                                                                     step,
                                                                     number_of_iters,
                                                                     loss.item()))
        train_loss = sum(train_losses)/len(train_losses)
        val_loss, val_acc = self._valid_epoch()

        return train_loss, val_loss, val_acc

    def _valid_epoch(self):
        self.model.eval()
        total = 0
        correct = 0
        val_losses = []
        tqdm.write("Start evaluate in validate data")
        with torch.no_grad():

            for step, batch in enumerate(tqdm(self.val_loader, desc='iter')):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)

                output = self.model(input_ids, attention_mask)

                _, predicted = torch.max(output.data, 1)
                total += label.shape[0]
                correct += (predicted == label).sum().item()

                loss = self.criterion(output, label).mean()
                val_losses.append(loss.item())
            val_acc = correct/total
            val_loss = sum(val_losses)/len(val_losses)
        return val_loss, val_acc

    def test(self):
        log_line(logger)
        logger.info('Testing using best model ...')
        model_path = os.path.join(self.save_path, 'best_model.pt')
        logger.info('Loading file {}'.format(model_path))
        model = self.model.load_state_dict(torch.load(model_path))
        text_contents = []
        predictions = []
        labels = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader, desc='iter')):
                text_content = batch['text_content']
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)

                output = model(input_ids, attention_mask)

                _, prediction = torch.max(output.data, 1)

                text_contents += text_content
                labels += label.tolist()
                predictions += prediction.tolist()
        data = zip(text_contents, labels, predictions)
        df = pd.DataFrame(data, columns=['Content', 'Label', 'Predict'])
        df.to_csv(os.path.join(self.save_path, 'test.csv'))

        print_result(labels, predictions, self.label_names)
