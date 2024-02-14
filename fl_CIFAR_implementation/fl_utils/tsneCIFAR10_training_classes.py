import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
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
    
def getTheModel_CIFAR10():
    model = Cifar10CNN()
    return model

class CIFAR10Dataset(Dataset):
    def __init__(self,X,y,transform=None):
        self.data = X.numpy()
        self.label = y.numpy()
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):
        
        x = Image.fromarray(self.data[index])

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x,y)
    
    def pin_memory(self):
        
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()
        
        return self