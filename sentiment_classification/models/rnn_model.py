import logging

import torch
import torch.nn as nn

logger = logging.getLogger("train")


class SimpleLSTM(nn.Module):

    def __init__(self, embedding_weights, num_classes=2):
        super(SimpleLSTM, self).__init__()
        embed_dim = embedding_weights.shape[1]
        embedding_weights = torch.FloatTensor(embedding_weights)
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)

        self.lstm = nn.LSTM(
            embed_dim, 128, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(512, num_classes)

        logger.info('-'*100)
        logger.info("Model: {}".format(self))

    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.lstm(x)
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out
