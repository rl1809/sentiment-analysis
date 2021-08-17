import logging

import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, \
    RobertaEmbeddings, \
    RobertaLayer, \
    RobertaPooler


logger = logging.getLogger("train")

PRETRAINED_PATH = 'vinai/phobert-base'


class ClassificationBert(nn.Module):

    def __init__(self, num_classes=2):
        super(ClassificationBert, self).__init__()
        self.bert = RobertaModel.from_pretrained(PRETRAINED_PATH)
        hidden_size = self.bert.config.hidden_size
        self.sequential = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )
        logger.info('-'*100)
        logger.info("Model: {}".format(self))

    def forward(self, input_ids, attention_mask):
        logits = self.bert(input_ids, attention_mask)[1]
        output = self.sequential(logits)
        return output


class RobertaModel4Mix(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaModel4Mix, self).__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder4Mix(config)
        self.pooler = RobertaPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask,
                input_ids2=None, attention_mask2=None,
                lam=None, mix_layer=1000, token_type_ids=None,
                position_ids=None, head_mask=None):

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)

            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2,
                                           lam, mix_layer, extended_attention_mask,
                                           extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class RobertaEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(RobertaEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([RobertaLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None,
                lam=None, mix_layer=1000, attention_mask=None,
                attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = lam * hidden_states + (1-lam)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = lam * hidden_states + (1-lam)*hidden_states2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class MixText(nn.Module):
    def __init__(self, num_classes=27, mix_option=False):
        super(MixText, self).__init__()

        if mix_option:
            self.bert = RobertaModel4Mix.from_pretrained('vinai/phobert-base')
        else:
            self.bert = RobertaModel.from_pretrained('vinai/phobert-base')
        hidden_size = self.bert.config.hidden_size
        self.linear = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(0.1),
                                    nn.Linear(hidden_size, num_classes))
        
        logger.info('-'*100)
        logger.info("Model: {}".format(self))

    def forward(self, input_ids, attention_mask, input_ids2=None,
                attention_mask2=None, lam=None, mix_layer=1000):

        if input_ids2 is not None:
            pooled_output = self.bert(
                input_ids, attention_mask, input_ids2, attention_mask2, lam, mix_layer)[1]

        else:

            pooled_output = self.bert(input_ids, attention_mask)[1]

        predict = self.linear(pooled_output)

        return predict