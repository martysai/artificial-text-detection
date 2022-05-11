from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5EncoderModel, T5PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_TO_HUB_NAME = {
    't5-base': 'sberbank-ai/ruT5-base',
    't5-large': 'sberbank-ai/ruT5-large',
}


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, last_hidden_state):
        pooled_output = self.dense(last_hidden_state)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MeanMaskedPooling(torch.nn.Module):
    def __init__(self):
        """
        An object to perform Mean Pooling that ignores PAD-token representations
        """
        super(MeanMaskedPooling, self).__init__()

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        lengths = pad_mask.sum(1).float()
        x = x.masked_fill((~pad_mask).unsqueeze(-1), 0.0)
        scaling = x.size(1) / lengths
        scaling = scaling.masked_fill(lengths == 0, 1.0).unsqueeze(-1)
        x = x.mean(1) * scaling
        return x


class T5ForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.pooler = Pooler(config)
        self.encoder = T5EncoderModel.from_pretrained(config._name_or_path)
        classifier_dropout = 0.2
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mean_pooling = MeanMaskedPooling()

        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False
        self.device_map = None


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mean_last_hidden_state = self.mean_pooling(outputs.last_hidden_state, attention_mask.bool())
        pooled_output = self.pooler(mean_last_hidden_state)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
