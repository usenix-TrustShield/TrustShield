# Necessary packages
import torch
import time
import copy
import math
import os
import numpy as np
from tsneCIFAR10_training_classes import *
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as trfm
import argparse

def save_model(model,save_loc,name):
    isExist = os.path.exists(save_loc)
    if not isExist:
        os.makedirs(save_loc)
    torch.save(model.state_dict(), save_loc+"/"+name)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = 0
    overall_end = len(dataset)
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

def load_CIFAR10_dataset(save_dir,val_ratio):

    cifar_trainset = datasets.CIFAR10(root=save_dir,train=True,download=True)
    cifar_testset = datasets.CIFAR10(root=save_dir,train=False,download=True)

    X_train = torch.tensor(cifar_trainset.data)
    y_train = torch.tensor(cifar_trainset.targets,dtype=int)

    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_ratio,stratify=y_train,random_state=1)

    X_test = torch.tensor(cifar_testset.data)
    y_test = torch.tensor(cifar_testset.targets,dtype=int)

    return X_train, X_test, X_val, y_train, y_test, y_val

def create_CIFAR10_dataset(X,y):
    transform = trfm.Compose([
    trfm.ToTensor(),
    trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = CIFAR10Dataset(X,y,transform)
    return dataset

def create_CIFAR10_traindataset(X,y):
        transform = trfm.Compose([
        trfm.RandomCrop(32, padding=4),
        trfm.RandomHorizontalFlip(),
        trfm.ToTensor(),
        trfm.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset = CIFAR10Dataset(X,y,transform)
        return dataset

def train_CIFAR10_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=10000, wait_epoch=25):
    """
    A function to realize training stage of deep learning

    """
    history  = {'train_loss' : [], 'val_loss':[], 'train_accuracy':[], 'val_accuracy':[], 'num_epoch':0}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999999
    remaining_epoch = wait_epoch
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        if remaining_epoch==0:
            break
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]: # TODO prepare data loader accordingly
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = torch.squeeze(inputs, dim=1) # TODO check the size whether it is correct
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            history[phase+'_loss'].append(epoch_loss)
            history[phase+'_accuracy'].append(epoch_acc.item())
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    #best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    remaining_epoch = wait_epoch
                else:
                    remaining_epoch -=1
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    history['num_epoch']=epoch
    return model, history

def simulation_func(learning_rate, device, waiting_epoch, batch_size, model_name, history_name):
    print("Preprocessing...")
    # load the data
    dataset_loc = '/store/datasets/CIFAR10'
    X_train,X_test,X_val,y_train,y_test,y_val = load_CIFAR10_dataset(dataset_loc,0.1)

    val_dataset = create_CIFAR10_dataset(X_val,y_val)
    test_dataset = create_CIFAR10_dataset(X_test,y_test)
    training_dataset = create_CIFAR10_traindataset(X_train, y_train)
    # initialize the parameters
    image_datasets = {'train':training_dataset, 'val':val_dataset}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print("Training...")
    # training hyper-parameters
    model = getTheModel_CIFAR10()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    model_trained, training_history = train_CIFAR10_model(model=model,criterion=criterion, optimizer=optimizer, scheduler=lr_sch,dataloaders=dataloaders,dataset_sizes=dataset_sizes, device=device,wait_epoch=waiting_epoch) 
    save_model(model_trained,'./models', model_name+".pt")
    np.save(history_name+'.npy', training_history)

    print("Testing...")
    model_trained.to(device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    model_trained.eval()
    correct_pred = 0
    num_pred = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = torch.squeeze(inputs,dim=1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_trained(inputs)
            _, preds = torch.max(outputs, 1)
            correct_pred += torch.sum(labels == preds)
            num_pred += len(labels)
    accuracy = correct_pred/num_pred
    print(accuracy.item())
    print("Simulation is complete")

def testing_func(device, batch_size, model_name):
    print("Testing... Model")
    model_trained = getTheModel_CIFAR10()
    dataset_loc = '/store/datasets/CIFAR10'
    _,X_test,_,_,y_test,_ = load_CIFAR10_dataset(dataset_loc,0.1)
    test_dataset = create_CIFAR10_dataset(X_test,y_test)
    loc = "./models/"+model_name+".pt"
    model_trained.load_state_dict(torch.load(loc))
    model_trained.to(device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    model_trained.eval()
    correct_pred = 0
    num_pred = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = torch.squeeze(inputs,dim=1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_trained(inputs)
            _, preds = torch.max(outputs, 1)
            correct_pred += torch.sum(labels == preds)
            num_pred += len(labels)
    accuracy = correct_pred/num_pred
    print(accuracy.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="tsne")
    parser.add_argument("--device-no", type=int,default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wepoch", type=int, default=25)
    parser.add_argument("--bsize", type=int, default=1024)
    parser.add_argument("--trainphase", type=int, default=0)
    opt = parser.parse_args()
    device = torch.device("cuda:"+str(opt.device_no) if (torch.cuda.is_available()) else "cpu")
    
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(opt.device_no) if torch.cuda.is_available() else 'CPU'))

    print(opt)
    learning_rate = opt.lr
    waiting_epoch = opt.wepoch
    batch_size = opt.bsize
    model_name = "CIFAR10_models_"+opt.modelname
    history_name = "history_"+model_name
    isTraining = opt.trainphase
    if isTraining:
        simulation_func(learning_rate, device, waiting_epoch, batch_size, model_name, history_name)
    else:
        testing_func(device, batch_size, model_name)