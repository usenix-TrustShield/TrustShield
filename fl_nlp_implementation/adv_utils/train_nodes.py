import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import models
from torchvision import transforms
from adv_utils import construct_models as cm
from statistics import NormalDist
from scipy import stats
from adv_utils import dataloader as dl
from adv_utils import distribute_data as dd
from transformers import AdamW


def train(model, train_loader, criterion, optimizer, device):
    """
    Trains the model appropriately.
    """
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data,)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def train_with_clipping(model, train_loader, criterion, optimizer, device, clipping=True, clipping_threshold=10):
    """
    Traind the model while enforching gradient clipping.
    """
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
            torch.nn.utils.clip_grad_value_(
                model.parameters(), clipping_threshold)
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def train_with_augmentation(model, train_loader, criterion, optimizer, device, clipping, clipping_threshold=10, use_augmentation=False, augment=None):
    """
    Augments data and performs model training.
    """
    model.train()
    train_loss = 0
    correct    = 0

    for data, mask, target in train_loader:
        data, mask, target = data.to(device), mask.to(device), target.to(device)

        if use_augmentation:
            data = augment(data)

        model.zero_grad()        
        preds = model(data, mask)
        loss  = criterion(preds, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        prediction  = preds.argmax(dim=1, keepdim=True)
        correct    += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def validation(model, test_loader, criterion, device):
    """
    Performs validation on the test dataset and returns the loss and accuracy.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, mask, target in test_loader:
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            output = model(data, mask)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)


def get_model_names(dict):
    """
    Returns the keys of a dictionary.
    """
    return list(dict.keys())


def create_model_optimizer_criterion_dict_for_cifar_cnn(n, y, number_of_samples, learning_rate, momentum, device, weight_decay=0):
    """
    A function to generate objects for each clients seperately
    n                   : The number of classes
    y                   : Target labels
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
        model_name    = "model" + str(i)
        model_info    = cm.BertModelForClassification(n).to(device[(i%(len(device)-1))+1])
        class_weights = compute_class_weight(class_weight="balanced",
                                             classes=np.unique(y),
                                             y=y.numpy())
        class_weights = dict(zip(np.unique(y), class_weights))
        class_wts     = [class_weights[i] for i in range(n)]
        # convert class weights to tensor
        weights = torch.tensor(class_wts, dtype=torch.float).to(device[(i%(len(device)-1))+1])
        # Define the loss function
        criterion_info = nn.NLLLoss()
        # criterion_info = nn.NLLLoss()
        optimizer_info = AdamW(model_info.parameters(), lr=1e-5)

        model_info = model_info.to(device[(i%(len(device)-1))+1])
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        # optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum,
        #                                  weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        #criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples, device):
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
            model_dict[name_of_models[i]].to(device[(i%(len(device)-1))+1])
    return model_dict


def start_train_end_node_process_cifar(number_of_samples, x_train_dict, mask_train_dict, y_train_dict, x_test_dict, mask_test_dict, y_test_dict, batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, device, clipping=False, clipping_threshold=10):
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

    name_of_x_train_sets    = get_model_names(x_train_dict)
    name_of_mask_train_sets = get_model_names(mask_train_dict)
    name_of_y_train_sets    = get_model_names(y_train_dict)
    name_of_x_test_sets     = get_model_names(x_test_dict)
    name_of_mask_test_sets  = get_model_names(mask_test_dict)
    name_of_y_test_sets     = get_model_names(y_test_dict)
    name_of_models          = get_model_names(model_dict)
    name_of_criterions      = get_model_names(criterion_dict)
    name_of_optimizers      = get_model_names(optimizer_dict)

    transform_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4)])

    # TODO this is not in parallel, multiprocessing can be implemented here for speed purposes
    for i in range(number_of_samples):
        print(i, ": num_samples =", len(x_train_dict[name_of_x_train_sets[i]]))
        train_ds = TensorDataset(
            x_train_dict[name_of_x_train_sets[i]],
            mask_train_dict[name_of_mask_train_sets[i]],
            y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(
            x_test_dict[name_of_x_test_sets[i]],
            mask_test_dict[name_of_mask_test_sets[i]],
            y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size)

        model     = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train_with_augmentation(model,
                                                                 train_dl,
                                                                 criterion,
                                                                 optimizer,
                                                                 device[(i%(len(device)-1))+1],
                                                                 clipping=clipping,
                                                                 clipping_threshold=clipping_threshold,
                                                                 use_augmentation=False,
                                                                 augment=transform_augment)

            test_loss, test_accuracy = validation(model, test_dl, criterion, device[(i%(len(device)-1))+1])
            # print(type(y_train_dict[name_of_y_train_sets[i]]))
            # print("Sample #{} split : ".format(i), torch.sum(y_train_dict[name_of_y_train_sets[i]]) / len(x_train_dict[name_of_x_train_sets[i]]))
            # print("Train set size :", len(x_train_dict[name_of_x_train_sets[i]]))
            # print("Test set size :", len(x_test_dict[name_of_x_test_sets[i]]))
            print("Sample #{} : Epoch #{} : Train Loss = {} : Train Accuracy = {} : Test Loss = {} : Test Accuracy = {}".format(i, epoch, train_loss, train_accuracy, test_loss, test_accuracy))


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
        # get the shape of the weight matrix for each layer parameters
        w_shape = list(parameters[k][1].shape)
        # add the number of edge users as first dimension to the weight dictionary
        w_shape.insert(0, len(model_dict))
        # initialize the matrices with 0 initial value
        weight_info = torch.zeros(w_shape, device=device)
        # append the dictionary layer by layer
        weight_dict.update({name: weight_info})

    weight_names_list = list(weight_dict.keys())
    # following lines takes client weights and writes all weights into single dictionary to take the mean to realize FedAvg
    with torch.no_grad():
        for i in range(len(model_dict)):  # write all clients into one matrix
            sample_param_data_list = list(
                model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        mean_weight_array = []
        for m in range(len(weight_names_list)):  # take the mean of all weights
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
    mean_weight_array = get_averaged_weights_faster(model_dict, device)  # aggregation function
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = mean_weight_array[j]
    return main_model


def start_mining_validator_node_cifar(number_of_edge_clients,
                                      number_of_validators,
                                      x_val_dict,
                                      mask_val_dict,
                                      y_val_dict,
                                      model_dict,
                                      criterion_dict,
                                      device,
                                      batch_size,
                                      current_round,
                                      validation_threshold=0.2,
                                      initialization_rounds=20,
                                      config="one_to_all",
                                      rgRatio=0.3):
    """
    Uses our validation mechanism on the edge node training.
    """
    # TODO test this function
    name_of_x_val_sets    = get_model_names(x_val_dict)
    name_of_mask_val_sets = get_model_names(mask_val_dict)
    name_of_y_val_sets    = get_model_names(y_val_dict)
    name_of_models        = get_model_names(model_dict)
    name_of_criterions    = get_model_names(criterion_dict)

    transform_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4)])

    mining_results_acc = []
    mining_results_loss = []
    miner_names = []

    if config == "one_to_all":

        # TODO this is not in parallel, multiprocessing can be implemented here for speed purposes
        for i in range(number_of_validators):
            val_ds = TensorDataset(
                x_val_dict[name_of_x_val_sets[i]],
                mask_val_dict[name_of_mask_val_sets[i]],
                y_val_dict[name_of_y_val_sets[i]])
            val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

            mining__acc = []
            mining__loss = []

            for j in range(number_of_edge_clients):
                unverified_edge_model = model_dict[name_of_models[j]]
                unverified_edge_criterion = criterion_dict[name_of_criterions[j]]

                val_loss, val_accuracy = validation(unverified_edge_model,
                                                    val_dl,
                                                    unverified_edge_criterion,
                                                    device[j%(len(device)-1)+1])
                #print("validation accuracy for client"+str(i)+" : "+str(val_accuracy))
                mining__acc.append(val_accuracy)
                mining__loss.append(val_loss)
            miner_names.append(name_of_y_val_sets[i])
            mining_results_acc.append(mining__acc)
            mining_results_loss.append(mining__loss)

    elif config == "one_to_one":

        # generate a vector containing the validator ids for the size of edges
        one2one_validators = np.random.randint(
            0, number_of_validators, number_of_edge_clients)
        mining__acc = []
        mining__loss = []
        for j in range(number_of_edge_clients):
            val_ds = TensorDataset(x_val_dict[name_of_x_val_sets[one2one_validators[j]]],
                                   y_val_dict[name_of_y_val_sets[one2one_validators[j]]])
            val_dl = DataLoader(val_ds, batch_size=batch_size * 2)
            unverified_edge_model = model_dict[name_of_models[j]]
            unverified_edge_criterion = criterion_dict[name_of_criterions[j]]
            val_loss, val_accuracy = validation(
                unverified_edge_model, val_dl, unverified_edge_criterion, device)

            mining__acc.append(val_accuracy)
            mining__loss.append(val_loss)
        mining_results_acc.append(mining__acc)
        mining_results_loss.append(mining__loss)

    elif config == "one_to_randgr":

        numberOfValidation = np.round(rgRatio*number_of_validators).astype(int)
        validators = np.transpose(np.array([np.random.choice(
            number_of_validators, size=numberOfValidation, replace=False) for i in range(number_of_edge_clients)]))

        for valround in range(numberOfValidation):
            mining__acc = []
            mining__loss = []

            for j in range(number_of_edge_clients):
                val_ds = TensorDataset(x_val_dict[name_of_x_val_sets[validators[valround, j]]],
                                       y_val_dict[name_of_y_val_sets[validators[valround, j]]])
                val_dl = DataLoader(val_ds, batch_size=batch_size * 2)
                unverified_edge_model = model_dict[name_of_models[j]]
                unverified_edge_criterion = criterion_dict[name_of_criterions[j]]
                val_loss, val_accuracy = validation(
                    unverified_edge_model, val_dl, unverified_edge_criterion, device)
                mining__acc.append(val_accuracy)
                mining__loss.append(val_loss)

            mining_results_acc.append(mining__acc)
            mining_results_loss.append(mining__loss)

    else:
        print("ERROR: Configuration is not specified")

    if current_round >= initialization_rounds:
        validated_edges, blacklisted_edges = apply_validation_threshold_acc(
            mining_results_acc, validation_threshold, number_of_edge_clients, name_of_models)
    else:
        validated_edges = np.asarray(name_of_models)
        blacklisted_edges = np.asarray([])

    return mining_results_loss, mining_results_acc, miner_names, validated_edges, blacklisted_edges


def apply_validation_threshold_acc(mining_results_acc, validation_threshold, number_of_edge_clients, name_of_models):
    """
    Applies the validation threshold over the metrics reported by each
    validator to determine blacklisted edges.
    """
    # TODO test this function
    mining_results_acc = np.asarray(
        mining_results_acc)  # (num_val x num_client)
    print(mining_results_acc)
    mean_mining_acc = np.mean(mining_results_acc, axis=0)
    is_validated = mean_mining_acc > validation_threshold
    edge_names = np.asarray([name_of_models[j]
                            for j in range(number_of_edge_clients)])
    validated_edges = edge_names[is_validated]
    blacklisted_edges = edge_names[~is_validated]
    return validated_edges, blacklisted_edges


def generate_ood_converter_dict(ood_ratio, train_number_classes, validator_num_classes, test_num_classes):
    """
    Performs some OOD changes.
    ood_ratio              percentage of out of distribution data
    validator_num_classes  number of classes in the validator nodes
    train_number_classes   number of classes in the training set
    test_num_classes       number of classes of the cloud test set
    """
    converter_dict = {}
    test_classes = []
    for i in range(test_num_classes):
        test_classes.append(i)
    for i in range(train_number_classes):
        if i < test_num_classes:
            converter_dict[i] = i
        else:
            converter_dict[i] = np.random.choice(
                test_num_classes, size=1, replace=False).tolist()[0]
    # print("Converter Dict", converter_dict)
    return converter_dict
