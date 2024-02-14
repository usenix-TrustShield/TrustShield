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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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

def load_CIFAR10_dataset(save_dir,val_ratio):

    cifar_trainset = datasets.CIFAR10(root=save_dir,train=True,download=True)
    cifar_testset = datasets.CIFAR10(root=save_dir,train=False,download=True)

    X_train = torch.tensor(cifar_trainset.data)
    y_train = torch.tensor(cifar_trainset.targets,dtype=int)
    X_train = torch.from_numpy(np.arange(0,len(y_train),1)) # the line saving the indices


    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_ratio,stratify=y_train,random_state=1)

    X_test = torch.tensor(cifar_testset.data)
    y_test = torch.tensor(cifar_testset.targets,dtype=int)

    return X_train, X_test, X_val, y_train, y_test, y_val

def main_func(device, batch_size, model_name, is_train=0):
    # load the model
    print("Loading the Model")
    model_trained = getTheModel_CIFAR10()
    dataset_loc = '/store/datasets/CIFAR10'
    if is_train:
        X_test,_,_,y_test,_,_ = load_CIFAR10_dataset(dataset_loc,0.0002)
        
    else:
        _,X_test,_,_,y_test,_ = load_CIFAR10_dataset(dataset_loc,0.0002)
    print(np.shape(X_test))
    test_dataset = create_CIFAR10_dataset(X_test,y_test)
    # loc = "./models/"+model_name+".pt"
    # model_trained.load_state_dict(torch.load(loc))
    # # Remove the last 
    # model_trained.fc2 = Identity()
    # model = model_trained#torch.nn.Sequential(*list(model_trained.children())[:-1])
    # model.to(device)
    # print(model)
    # # Set the model in evaluation mode

    # # Example input data
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

    # model.eval()
    # # Forward pass to obtain features
    # correct_pred = 0
    # num_pred = 0
    # output_embeddings = np.zeros((y_test.shape[0],128))
    # with torch.no_grad():
    #     for i, (inputs, labels) in enumerate(test_loader):
    #         inputs = torch.squeeze(inputs,dim=1)
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         outputs = model(inputs)
    #         if (i+1)*batch_size>y_test.shape[0]:
    #             output_embeddings[(i)*batch_size:,:]=outputs.cpu().detach().numpy()
    #         else:
    #             output_embeddings[(i)*batch_size:(i+1)*batch_size,:]=outputs.cpu().detach().numpy()
    # #         _, preds = torch.max(outputs, 1)
    # #         correct_pred += torch.sum(labels == preds)
    # #         num_pred += len(labels)
    # # accuracy = correct_pred/num_pred
    # # print(accuracy.item())
    # print(np.shape(output_embeddings))

    # tsne = TSNE(n_components=2, random_state=42,n_iter=1000)
    # X_tsne = tsne.fit_transform(output_embeddings)
    # # Plot the t-SNE visualization
    # plt.figure(figsize=(10, 8))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, cmap='tab10')
    # plt.colorbar()
    # plt.title('t-SNE Visualization of CIFAR-10')
    # plt.show()
    # plt.savefig("trained_tsne_train.png")
    # np.save("tsneF_train.npy", X_tsne) 
    np.save("tsneF_train_y.npy", y_test)
    np.save("tsne_X_ind.npy", X_test) # the line saving the indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="tsne")
    parser.add_argument("--device-no", type=int,default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wepoch", type=int, default=25)
    parser.add_argument("--bsize", type=int, default=1024)
    opt = parser.parse_args()
    device = torch.device("cuda:"+str(opt.device_no) if (torch.cuda.is_available()) else "cpu")
    
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(opt.device_no) if torch.cuda.is_available() else 'CPU'))

    print(opt)
    learning_rate = opt.lr
    waiting_epoch = opt.wepoch
    batch_size = opt.bsize
    model_name = "CIFAR10_models_"+opt.modelname
    history_name = "history_"+model_name
    data_train = 1
    main_func(device, batch_size, model_name,data_train)