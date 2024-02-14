import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import models
from torchvision import transforms
from fl_utils import construct_models as cm
from statistics import NormalDist
from scipy import stats


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import models
from torchvision import transforms
from fl_utils import construct_models as cm
from statistics import NormalDist
from scipy import stats


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)

def train_with_clipping(model, train_loader, criterion, optimizer, device, clipping=True, clipping_threshold=10):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if clipping:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipping_threshold)
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def train_with_augmentation(model, train_loader, criterion, optimizer, device, clipping, clipping_threshold=10, use_augmentation=False, augment=None ):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if use_augmentation:
            data = augment(data)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if clipping:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipping_threshold)

        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def validation(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)


def get_model_names(model_dict):
    name_of_models = list(model_dict.keys())
    return name_of_models


def get_optimizer_names(optimizer_dict):
    name_of_optimizers = list(optimizer_dict.keys())
    return name_of_optimizers


def get_criterion_names(criterion_dict):
    name_of_criterions = list(criterion_dict.keys())
    return name_of_criterions


def get_x_train_sets_names(x_train_dict):
    name_of_x_train_sets = list(x_train_dict.keys())
    return name_of_x_train_sets


def get_y_train_sets_names(y_train_dict):
    name_of_y_train_sets = list(y_train_dict.keys())
    return name_of_y_train_sets


def get_x_valid_sets_names(x_valid_dict):
    name_of_x_valid_sets = list(x_valid_dict.keys())
    return name_of_x_valid_sets


def get_y_valid_sets_names(y_valid_dict):
    name_of_y_valid_sets = list(y_valid_dict.keys())
    return name_of_y_valid_sets


def get_x_test_sets_names(x_test_dict):
    name_of_x_test_sets = list(x_test_dict.keys())
    return name_of_x_test_sets


def get_y_test_sets_names(y_test_dict):
    name_of_y_test_sets = list(y_test_dict.keys())
    return name_of_y_test_sets


def create_model_optimizer_criterion_dict_for_cifar_cnn(number_of_samples, learning_rate, momentum, device, weight_decay=0):
    """
    A function to generate objects for each clients seperately

    number_of_samples   : the number of edge clients
    learning_rate       : the learning rate of the internal training
    momentum            : momentum of the internal learning
    device              : device of the edge client models to be assigned
    weight_decay        : decay factor of the internal weights
    ------
    model_dict          : the dictionary consisting of edge client models
    optimizer_dict      : the dictionary consisting of the optimizer of each edge models
    criterion_dict      : the loss function of the each edge client model
    """
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = cm.Cifar10CNN()

        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum,
                                         weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples):
    """
    A function to realize the model downlink of the edge clients from the cloud server
    main_model          : the model of the central server / cloud
    model_dict          : the model dictionary consisting of the edge users
    number_of_samples   : the number of edge users
    ------
    model_dict          : the dictionary consisting of the models updated with cloud's last sync
    """
    name_of_models = list(model_dict.keys())
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for i in range(number_of_samples):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(main_model_param_data_list)):
                sample_param_data_list[j].data = main_model_param_data_list[j].data.clone()
    return model_dict


def start_train_end_node_process_cifar(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict, batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, device,clipping=False, clipping_threshold =10):
    """
    A function to realize edge users' internal training mechanism.
    TODO can be parallelized to increase the speed of the simulations
    number_of_samples       : the number of edge devices
    x_train_dict            : the dictionary contains the data of each edge device for training
    y_train_dict            : the dictionary contains the labels of each edge device for training
    x_test_dict             : the dictionary contains the data of each edge device for testing
    y_test_dict             : the dictionary contains the labels of each edge device for testing
    batch_size              : the batch size of the internal training
    model_dict              : the models of each edge device
    criterion_dict          : the loss functions of each edge device
    optimizer_dict          : the learner of each edge device
    numEpoch                : the number of epoch each edge device performs training
    device                  : the device of the model will be assigned during training # TODO a parallelisation is possible , this should be a dictionary containing individual clients
    clipping                : the boolean for clipping
    clipping_threshold      : the threshold of the clipping
    ------
    """
    
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    transform_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4)])

    for i in range(number_of_samples): # TODO this is not in parallel, multiprocessing can be implemented here for speed purposes

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train_with_augmentation(model, train_dl, criterion, optimizer, device,
                                                                 clipping=clipping,
                                                                 clipping_threshold=clipping_threshold,
                                                                 use_augmentation=True, augment=transform_augment)

            test_loss, test_accuracy = validation(model, test_dl, criterion, device)


def get_averaged_weights_faster(model_dict, device):
    """
    A function to realize the FedAvg aggregation

    model_dict          : the dictionary storing the edge user models internally trained for the current round
    device              : the device name to conduct aggregation load and erase the models
    -------
    mean_weight_array   : the weight matrix formed by aggregation of the client uplink weights
    """
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    # named_parameters sends the layer name and the layer data as tuple 
    # parameters sends only sends the layer data without sending the layer name

    # initialize a weight_dict dictionary to store weights coming from the edge users
    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)              # get the shape of the weight matrix for each layer parameters
        w_shape.insert(0, len(model_dict))                  # add the number of edge users as first dimension to the weight dictionary
        weight_info = torch.zeros(w_shape, device=device)   # initialize the matrices with 0 initial value
        weight_dict.update({name: weight_info})             # append the dictionary layer by layer 

    weight_names_list = list(weight_dict.keys())
    # following lines takes client weights and writes all weights into single dictionary to take the mean to realize FedAvg
    with torch.no_grad():
        for i in range(len(model_dict)): # write all clients into one matrix
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        mean_weight_array = []
        for m in range(len(weight_names_list)): # take the mean of all weights
            mean_weight_array.append(torch.mean(weight_dict[weight_names_list[m]], 0))

    return mean_weight_array


def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, device):
    """
    A function to aggregate the weights coming from the edge users to the cloud
    main_model      : the central server model
    model_dict      : the dictionary containig the edge users' model
    device          : the device of which these models will be runned over
    -------
    main_model      : the central server model after the aggregation for the current communication round
    """
    mean_weight_array = get_averaged_weights_faster(model_dict, device) # aggregation function
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = mean_weight_array[j]
    return main_model


def unbias_noniid_edge_update(main_model, edge_model, device):
    #TODO comment the function
    mean_weight_array = get_averaged_weights_faster({'mainmodel':main_model, 'edgemodel':edge_model}, device) # aggregation function
    main_model_param_data_list = list(edge_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = mean_weight_array[j]
    return edge_model


def start_mining_validator_node_mnist(number_of_edge_clients, number_of_validators, x_val_dict, y_val_dict, model_dict, criterion_dict, device, batch_size, current_round,validation_threshold=0.2, initialization_rounds = 20, config="one_to_all", rgRatio=0.3):
    # TODO test this function
    name_of_x_val_sets = get_x_test_sets_names(x_val_dict)
    name_of_y_val_sets = get_y_test_sets_names(y_val_dict)

    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)

    mining_results_acc = []
    mining_results_loss = []
    miner_names = []

    if config=="one_to_all":

        for i in range(number_of_validators): # TODO this is not in parallel, multiprocessing can be implemented here for speed purposes
            val_ds = TensorDataset(x_val_dict[name_of_x_val_sets[i]], y_val_dict[name_of_y_val_sets[i]])
            val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

            mining__acc = []
            mining__loss = []
            
            for j in range(number_of_edge_clients):
                unverified_edge_model = model_dict[name_of_models[j]]
                unverified_edge_criterion = criterion_dict[name_of_criterions[j]]

                val_loss, val_accuracy = validation(unverified_edge_model, val_dl, unverified_edge_criterion, device)

                mining__acc.append(val_accuracy)
                mining__loss.append(val_loss)
            miner_names.append(name_of_y_val_sets[i])
            mining_results_acc.append(mining__acc)
            mining_results_loss.append(mining__loss)

    elif config=="one_to_one":
        
        one2one_validators = np.random.randint(0, number_of_validators, number_of_edge_clients) # generate a vector containing the validator ids for the size of edges
        mining__acc = []
        mining__loss = []
        for j in range(number_of_edge_clients):
            val_ds = TensorDataset(x_val_dict[name_of_x_val_sets[one2one_validators[j]]], y_val_dict[name_of_y_val_sets[one2one_validators[j]]])
            val_dl = DataLoader(val_ds, batch_size=batch_size * 2)
            unverified_edge_model = model_dict[name_of_models[j]]
            unverified_edge_criterion = criterion_dict[name_of_criterions[j]]
            val_loss, val_accuracy = validation(unverified_edge_model, val_dl, unverified_edge_criterion, device)

            mining__acc.append(val_accuracy)
            mining__loss.append(val_loss)
        mining_results_acc.append(mining__acc)
        mining_results_loss.append(mining__loss)
    
    elif config=="one_to_randgr":

        numberOfValidation = np.round(rgRatio*number_of_validators).astype(int)
        validators = np.transpose(np.array([np.random.choice(number_of_validators, size=numberOfValidation, replace=False) for i in range(number_of_edge_clients)]))
        
        for valround in range(numberOfValidation):
            mining__acc = []
            mining__loss = []
        
            for j in range(number_of_edge_clients):
                val_ds = TensorDataset(x_val_dict[name_of_x_val_sets[validators[valround, j]]], y_val_dict[name_of_y_val_sets[validators[valround, j]]])
                val_dl = DataLoader(val_ds, batch_size=batch_size * 2)
                unverified_edge_model = model_dict[name_of_models[j]]
                unverified_edge_criterion = criterion_dict[name_of_criterions[j]]
                val_loss, val_accuracy = validation(unverified_edge_model, val_dl, unverified_edge_criterion, device)
                mining__acc.append(val_accuracy)
                mining__loss.append(val_loss)
        
            mining_results_acc.append(mining__acc)
            mining_results_loss.append(mining__loss)

    else:
        print("ERROR: Configuration is not specified")

        
    if current_round>=initialization_rounds:
        validated_edges, blacklisted_edges = apply_validation_threshold_acc(mining_results_acc, validation_threshold, number_of_edge_clients, name_of_models)        
    else:
        validated_edges = np.asarray(name_of_models)
        blacklisted_edges = np.asarray([])
        
    return mining_results_loss ,mining_results_acc, miner_names, validated_edges, blacklisted_edges


def start_mining_validator_node_cifar(number_of_edge_clients, number_of_validators, x_val_dict, y_val_dict, model_dict, criterion_dict, device, batch_size, current_round,validation_threshold=0.2, initialization_rounds = 20, config="one_to_all", rgRatio=0.3):
    # TODO test this function
    name_of_x_val_sets = get_x_test_sets_names(x_val_dict)
    name_of_y_val_sets = get_y_test_sets_names(y_val_dict)

    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)

    transform_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4)])

    mining_results_acc = []
    mining_results_loss = []
    miner_names = []

    

    if config=="one_to_all":

        for i in range(number_of_validators): # TODO this is not in parallel, multiprocessing can be implemented here for speed purposes
            val_ds = TensorDataset(x_val_dict[name_of_x_val_sets[i]], y_val_dict[name_of_y_val_sets[i]])
            val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

            mining__acc = []
            mining__loss = []
            
            for j in range(number_of_edge_clients):
                unverified_edge_model = model_dict[name_of_models[j]]
                unverified_edge_criterion = criterion_dict[name_of_criterions[j]]

                val_loss, val_accuracy = validation(unverified_edge_model, val_dl, unverified_edge_criterion, device)

                mining__acc.append(val_accuracy)
                mining__loss.append(val_loss)
            miner_names.append(name_of_y_val_sets[i])
            mining_results_acc.append(mining__acc)
            mining_results_loss.append(mining__loss)

    elif config=="one_to_one":
        
        one2one_validators = np.random.randint(0, number_of_validators, number_of_edge_clients) # generate a vector containing the validator ids for the size of edges
        mining__acc = []
        mining__loss = []
        for j in range(number_of_edge_clients):
            val_ds = TensorDataset(x_val_dict[name_of_x_val_sets[one2one_validators[j]]], y_val_dict[name_of_y_val_sets[one2one_validators[j]]])
            val_dl = DataLoader(val_ds, batch_size=batch_size * 2)
            unverified_edge_model = model_dict[name_of_models[j]]
            unverified_edge_criterion = criterion_dict[name_of_criterions[j]]
            val_loss, val_accuracy = validation(unverified_edge_model, val_dl, unverified_edge_criterion, device)

            mining__acc.append(val_accuracy)
            mining__loss.append(val_loss)
        mining_results_acc.append(mining__acc)
        mining_results_loss.append(mining__loss)
    
    elif config=="one_to_randgr":

        numberOfValidation = np.round(rgRatio*number_of_validators).astype(int)
        validators = np.transpose(np.array([np.random.choice(number_of_validators, size=numberOfValidation, replace=False) for i in range(number_of_edge_clients)]))
        
        for valround in range(numberOfValidation):
            mining__acc = []
            mining__loss = []
        
            for j in range(number_of_edge_clients):
                val_ds = TensorDataset(x_val_dict[name_of_x_val_sets[validators[valround, j]]], y_val_dict[name_of_y_val_sets[validators[valround, j]]])
                val_dl = DataLoader(val_ds, batch_size=batch_size * 2)
                unverified_edge_model = model_dict[name_of_models[j]]
                unverified_edge_criterion = criterion_dict[name_of_criterions[j]]
                val_loss, val_accuracy = validation(unverified_edge_model, val_dl, unverified_edge_criterion, device)
                mining__acc.append(val_accuracy)
                mining__loss.append(val_loss)
        
            mining_results_acc.append(mining__acc)
            mining_results_loss.append(mining__loss)

    else:
        print("ERROR: Configuration is not specified")

        
    if current_round>=initialization_rounds:
        validated_edges, blacklisted_edges = apply_validation_threshold_acc(mining_results_acc, validation_threshold, number_of_edge_clients, name_of_models)        
    else:
        validated_edges = np.asarray(name_of_models)
        blacklisted_edges = np.asarray([])
        
    return mining_results_loss ,mining_results_acc, miner_names, validated_edges, blacklisted_edges

def apply_validation_threshold_acc(mining_results_acc, validation_threshold, number_of_edge_clients, name_of_models):
        # TODO test this function
        mining_results_acc = np.asarray(mining_results_acc) # (num_val x num_client)
        mean_mining_acc = np.mean(mining_results_acc, axis=0)
        is_validated = mean_mining_acc > validation_threshold
        edge_names = np.asarray([name_of_models[j] for j in range(number_of_edge_clients)])
        validated_edges = edge_names[is_validated]
        blacklisted_edges = edge_names[~is_validated]
        return validated_edges, blacklisted_edges


# the functions on the below is for mnist classification

def create_model_optimizer_criterion_dict_for_mnist(number_of_samples, learning_rate, momentum, device, is_cnn=False, weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        if is_cnn:
            model_info = cm.Netcnn()
        else:
            model_info = cm.Net2nn()
        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def start_train_end_node_process_without_print(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch,
                                               device):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    for i in range(number_of_samples):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]

        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer, device)
            test_loss, test_accuracy = validation(model, test_dl, criterion, device)
