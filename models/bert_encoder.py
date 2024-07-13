import torch
import torch.nn as nn
from transformers import BertModel

class BertEncoder(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased'):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
