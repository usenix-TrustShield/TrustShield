from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn as nn
from transformers import AutoModel

def update_learning_rate_decay(optimizer_dict, new_lr):
    """
    Updates the learning rate for the edge clients accordingly.
    """
    for i in range(len(optimizer_dict)):
        optimizer_name = "optimizer" + str(i)
        optimizer_dict[optimizer_name].param_groups[0]["lr"] = new_lr
    return optimizer_dict

class BertModelForClassification(nn.Module):
    def __init__(self, n_classes) -> None:
        super(BertModelForClassification, self).__init__()
        self.initialize_bert()
        # Declare the required layers
        self.dropout = nn.Dropout(0.1)        
        self.relu    = nn.ReLU()
        self.fc1     = nn.Linear(768, 512)
        self.fc2     = nn.Linear(512, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def initialize_bert(self):
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

    def forward(self, sent_id, mask, attention=False):
        # pass the inputs to the underlying bert model  
        z = self.bert(sent_id, attention_mask=mask, return_dict=False, output_attentions=True)
        x = self.fc1(z[-2])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        if attention:
            return x, z[-1]
        return x
