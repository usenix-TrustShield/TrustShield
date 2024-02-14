from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch import nn
# from transformers import AutoModel
from transformers import AutoModel

def weights_init(model, torch_manual_seed=2304):
    torch.manual_seed(torch_manual_seed)
    torch.cuda.manual_seed_all(torch_manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def learning_rate_decay(optimizer_dict, decay_rate):
    for i in range(len(optimizer_dict)):
        optimizer_name = "optimizer" + str(i)
        old_lr = optimizer_dict[optimizer_name].param_groups[0]["lr"]
        optimizer_dict[optimizer_name].param_groups[0]["lr"] = old_lr * decay_rate
    return optimizer_dict

def update_learning_rate_decay(optimizer_dict, new_lr):
    for i in range(len(optimizer_dict)):
        optimizer_name = "optimizer" + str(i)
        optimizer_dict[optimizer_name].param_groups[0]["lr"] = new_lr
    return optimizer_dict

class Cifar10CNN(nn.Module):

    def __init__(self):
        super(Cifar10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)


        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)


        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)


        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        # self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)


        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = self.fc2(x)
        return x
    
# mnist classes and functions are the ones below

class Net2nn(nn.Module):
    def __init__(self):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Netcnn(nn.Module):
    def __init__(self):
        super(Netcnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

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
