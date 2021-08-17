import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("train")


class TextCNN(nn.Module):

    def __init__(self, embedding_weights, num_classes=2):
        super(TextCNN, self).__init__()
        filter_sizes = [2, 3, 4]
        num_filters = 32
        embed_size = embedding_weights.shape[1]
        embedding_weights = torch.FloatTensor(embedding_weights)
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        # self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(filter_sizes)*num_filters, num_classes)

        logger.info('-'*100)
        logger.info("Model: {}".format(self))

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class DPCNN(nn.Module):

    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.config = config
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(
            1, self.channel_size, (3, self.config.word_embedding_dimension), stride=1)
        self.conv3 = nn.Conv2d(
            self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*self.channel_size, 4)

    def forward(self, x):
        batch = x.shape[0]
        x = self.conv_region_embedding(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)
        x = x.view(batch, 2*self.channel_size)
        x = self.linear_out(x)
        return x

    def _block(self, x):
        x = self.padding_pool(x)
        px = self.pooling(x)

        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = x+px

        return x
