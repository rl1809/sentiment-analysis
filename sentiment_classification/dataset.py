"""Create dataset and dataloader from file"""
from torch.utils.data import DataLoader

from base.base_dataset import Dataset
from encode import BertTokenizer, TextTokenizer


class BertDataset(Dataset):
    """Create Torch tensor dataset"""

    def __init__(self, *args, **kwargs):
        super(BertDataset, self).__init__(*args, **kwargs)
        self.transform = BertTokenizer()
        self.transform.fit(*args)

    def __getitem__(self, index):
        raw_content, raw_label = self.contents[index], self.labels[index]
        content, label = self.transform(raw_content, raw_label)
        return {
            'text_content': raw_content,
            'input_ids': content['input_ids'].flatten(),
            'attention_mask': content['attention_mask'].flatten(),
            'label': label
        }


class RegularDataset(Dataset):
    def __init__(self, max_len, *args, **kwargs):
        super(RegularDataset, self).__init__(*args, **kwargs)
        self.transform = TextTokenizer(max_len)
        self.transform.fit(*args)
        self.word_index = self.transform.word_index

    def __getitem__(self, index):
        raw_content, raw_label = self.contents[index], self.labels[index]
        content, label = self.transform(raw_content, raw_label)
        return {
            'text_content': raw_content,
            'content': content,
            'label': label
        }


def get_dataloader(data, label, max_len=50, batch_size=16):
    """Create dataloader from dataset"""
    dataset = RegularDataset(max_len, data, label)

    return DataLoader(dataset, batch_size)


def get_bert_dataloader(data, label, batch_size=16):
    """Create dataloader from dataset"""
    dataset = BertDataset(data, label)

    return DataLoader(dataset, batch_size)
