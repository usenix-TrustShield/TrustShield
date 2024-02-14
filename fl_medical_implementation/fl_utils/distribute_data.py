import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
import requests
import pickle
import gzip
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import seaborn as sn

def load_cifar_data():
    train_data_path ='/home/ob3942/tshield_medical_exp/data_subset/train'
    test_data_path = '/home/ob3942/tshield_medical_exp/data_subset/test'
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225]),])

    test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)

    trainset = torchvision.datasets.ImageFolder(root=train_data_path,transform=train_transforms)
    testset = torchvision.datasets.ImageFolder(root=test_data_path,transform=test_transforms) 
    #print("length of trainset",len(trainset))
    x_train = torch.zeros((len(trainset), 3, 128, 128))
    y_train = torch.zeros(len(trainset))
    ind_train = 0
    ctr=0
    for data, output in trainset:
        x_train[ind_train, :, :, :] = data
        y_train[ind_train] = output
        ind_train = ind_train + 1
        
    x_test = torch.zeros((len(testset), 3, 128, 128))
    y_test = torch.zeros(len(testset))
    ind_test = 0
    for data, output in testset:
        x_test[ind_test, :, :, :] = data
        y_test[ind_test] = output
        ind_test = ind_test + 1

    y_train = y_train.type(torch.LongTensor)
    y_test = y_test.type(torch.LongTensor)
    #print(x_train,y_train)
    return x_train, y_train, x_test, y_test

def split_and_shuffle_labels(y_data, seed, amount):
    """
    A function to take all labels and assign a new position to that one among other positions of the same label
    y_data : the labels of the set
    amount : the number of sample used in the process
    --------
    label_dict: the output dict showing the positions of each label
    """
    y_data = pd.DataFrame(y_data, columns=["labels"])
    y_data["i"] = np.arange(len(y_data))
    label_dict = dict()
    for i in range(3):
        var_name = "label" + str(i)
        label_info = y_data[y_data["labels"] == i]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = label_info[0:amount]
        label_info = pd.DataFrame(label_info, columns=["labels", "i"])
        label_dict.update({var_name: label_info})

    return label_dict
def gaussian(num_validators, test_samples, num_classes, get_dist_plots=True):
    """
    A function to generate gaussian distributions for validators' private data
    num_validators      : the number of validator nodes
    test_samples        : the total number of test samples for each class
    num_classes         : the number of classes
    get_dist_plots      : plot the distribution of each validators' data
    =======
    lbl                 : the labels of corresponding classes
    output_dist         : (n_validators x n_classes) the class distribution of each validator node
    """

    current_mean = 0
    validator_means = []
    shift = num_classes/num_validators
    for i in range(num_validators):
        validator_means.append(current_mean)
        current_mean += shift
    validator_means = np.floor(validator_means).astype(int)
    validator_label_dist = []

    for i in range(num_validators):
        output = np.round(np.random.randn(test_samples)*1.35)
        output = output - validator_means[i]
        output = output % num_classes
        while not np.array_equiv(np.unique(output),np.linspace(0,num_classes-1,num_classes)):
            output = np.round(np.random.randn(test_samples)*1.35)
            output = output - validator_means[i]
            output = output% num_classes
        validator_label_dist.append(output.astype(int))
        if get_dist_plots:
            sn.distplot(output,bins=range(num_classes+1)) 
            plt.xlabel("Class Label")                                                                                                                                                                                                                                                                                               
            #plt.waitforbuttonpress(10)
    validator_label_dist = np.array(validator_label_dist)    
    if get_dist_plots:
        plt.figure()
        sn.distplot([i for i in validator_label_dist], bins=range(num_classes+1))
        plt.xlabel("Class Label")
        #plt.waitforbuttonpress(10)
    lbl, ct = np.unique(validator_label_dist, return_counts=True) 
    normalization_factor = test_samples/np.max(ct)
    output_dist = np.zeros((num_validators, num_classes))
    for i in range(num_validators):
        _, ct_temp = np.unique(validator_label_dist[i,:], return_counts=True)
        ct_temp = np.floor(ct_temp*normalization_factor).astype(int)
        output_dist[i,:] = ct_temp
    return lbl.astype(int), output_dist.astype(int)



def get_info_for_validator_nodes(number_of_validators, n, amount, seed, distribution="uniform"):
    """
    A function to realize the validator nodes and their corresponding private data distributions and each validator will have the same number of data samples
    
    number_of_samples       : the number of validator nodes
    n                       : number of classes
    amount                  : number of data samples in total
    seed                    : the seed of the randomization mechanism
    distribution            : type of the data distribution that the validator nodes have 
    --------
    node_label_info         : (size: # clients x # classes) a dictionary containing a map for which state is which class 
                            for a specific client, -1 implies there is no such class
    total_label_occurences  : (size: 2 x # classes) a table showing the number of clients who have corresponding class and 
                            the number of samples from that class for each client correspondingly
    amount_info_table       : (size: # clients x # classes) a map containing the number of samples from classes for each 
                            client has with the mapping done in node_label_info
    """
    # node label info generation is only for the sake of similarity between other dictionary generation functions
    # following lines are not very significant, just determines the label position in state maps of each client
    node_label_info = np.ones([number_of_validators, n]) * -1
    columns = []
    for j in range(n):
        columns.append("s" + str(j))
    node_label_info = pd.DataFrame(node_label_info, columns=columns, dtype=int)

    np.random.seed(seed)
    seeds = np.random.choice(number_of_validators * n * 5, size=number_of_validators, replace=False) # generate different seed for each client
    for i in range(number_of_validators):
        np.random.seed(seeds[i])
        which_labels = np.random.choice(3, size=3, replace=False)         # select the classes that client will have
        node_label_info.iloc[i, 0:len(which_labels)] = which_labels

    #################################
    #################################
    if distribution == "uniform":
        # TODO test the function
        dist = np.ones((number_of_validators, n))*np.floor(amount/number_of_validators)
    elif distribution == "gaussian": # TODO test the function
        _, dist = gaussian(num_validators=number_of_validators, test_samples=amount, num_classes=n)
    elif distribution == "dirichelet":
        pass
        # TODO be sure about the dirichelet alpha value is a proper one
    else:
        print("There is no such distribution for validator datasets")
    ##################################
    ##################################

    amount_info_table = pd.DataFrame(np.zeros([number_of_validators, n]), dtype=int)
    for a in range(number_of_validators):
        for b in range(n):
            if node_label_info.iloc[a, b] == -1:
                amount_info_table.iloc[a, b] = 0
            else:
                amount_info_table.iloc[a, b] = dist[a, node_label_info.iloc[a, b]]

    return node_label_info, [], amount_info_table


def get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples, n, amount, seed, min_n_each_node=2):
    """
    A function the generate data distributions for different nodes by considering wether the dataset will be non-iid or not.
    number_of_samples : the number of edge clients
    n       :           number of classes
    amount  :           number of data samples in total
    seed    :           the seed of the randomization mechanism
    min_n_each_node :   the number of classes the clients can have at least 
    --------
    node_label_info :           (size: # clients x # classes) a dictionary containing a map for which state is which class 
                                for a specific client, -1 implies there is no such class
    total_label_occurences :    (size: 2 x # classes) a table showing the number of clients who have corresponding class and 
                                the number of samples from that class for each client correspondingly
    amount_info_table :         (size: # clients x # classes) a map containing the number of samples from classes for each 
                                client has with the mapping done in node_label_info
    """
    node_label_info = np.ones([number_of_samples, n]) * -1
    columns = []
    for j in range(n):
        columns.append("s" + str(j))
    node_label_info = pd.DataFrame(node_label_info, columns=columns, dtype=int)

    np.random.seed(seed)
    seeds = np.random.choice(number_of_samples * n * 5, size=number_of_samples, replace=False) # generate different seed for each client
    for i in range(number_of_samples):
        np.random.seed(seeds[i])
        how_many_label_created = np.random.randint( n + 1 - min_n_each_node) + min_n_each_node  # ensures at least one label is created by default
        which_labels = np.random.choice(n, size=how_many_label_created, replace=False)         # select the classes that client will have
        node_label_info.iloc[i, 0:len(which_labels)] = which_labels

    #################################
    #################################

    total_label_occurences = pd.DataFrame()
    for m in range(n):

        total_label_occurences.loc[0, m] = int(np.sum(node_label_info.values == m))
        if total_label_occurences.loc[0, m] == 0:
            total_label_occurences.loc[1, m] = 0
        else:
            total_label_occurences.loc[1, m] = int(amount / np.sum(node_label_info.values == m))
    total_label_occurences = total_label_occurences.astype('int32')

    ##################################
    ##################################

    amount_info_table = pd.DataFrame(np.zeros([number_of_samples, n]), dtype=int)
    for a in range(number_of_samples):
        for b in range(n):
            if node_label_info.iloc[a, b] == -1:
                amount_info_table.iloc[a, b] = 0
            else:
                amount_info_table.iloc[a, b] = total_label_occurences.iloc[1, node_label_info.iloc[a, b]]

    return node_label_info, total_label_occurences, amount_info_table


def distribute_cifar_data_to_participants(label_dict, amount, number_of_samples, n,
                                          x_data, y_data, x_name, y_name, node_label_info,
                                          amount_info_table):
    label_names = list(label_dict)
    label_dict_data = pd.DataFrame(columns=["labels", "i"])

    for a in label_names:
        data = pd.DataFrame.from_dict(label_dict[a])
        label_dict_data = pd.concat([label_dict_data, data], ignore_index=True)

    index_counter = pd.DataFrame(label_names, columns=["labels"])
    index_counter["start"] = np.ones(n, dtype=int) * np.arange(n) * amount
    index_counter["end"] = np.ones(n, dtype=int) * np.arange(n) * amount
    # print("index_counter",index_counter)
    x_data_dict = dict()
    y_data_dict = dict()
    #print(label_dict)
    #print(label_dict_data)
    for i in range(number_of_samples):
        node_data_indices = pd.DataFrame()

        xname = x_name + str(i)
        yname = y_name + str(i)

        for j in range(n):
            label = node_label_info.iloc[i, j]
            if label != -1:
                label_amount = amount_info_table.iloc[i, j]
                index_counter.loc[label, "end"] = index_counter.loc[label, "end"] + label_amount
                node_data_indices = pd.concat([node_data_indices, label_dict_data.loc[
                                                                  index_counter.loc[label, "start"]:(index_counter.loc[label, "end"] - 1),
                                                                  "i"]])

                # index_counter.loc[label, "start"] = index_counter.loc[label, "end"]
        #print("debug",node_data_indices.iloc[:, 0])
        x_info = x_data[node_data_indices.iloc[:, 0].reset_index(drop=True), :]

        x_data_dict.update({xname: x_info})
        #print(x_data_dict)

        y_info = y_data[node_data_indices.iloc[:, 0].reset_index(drop=True)]
        y_data_dict.update({yname: y_info})
    #print("index_counter", index_counter)
    return x_data_dict, y_data_dict


def choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed=90):
    """
    A function to select some clients to be adversary
    
    hostile_node_percentage : the ratio of the adverserial nodes in [0,1]
    number_of_samples       : the number of edge clients
    hostility_seed          : the seed of the randomization mechanism
    ------
    y_dict                  : the output dictionary containing the names of the adverserial nodes
    """
    nodes_list=[]
    np.random.seed(hostility_seed)
    nodes=np.random.choice(number_of_samples, size=int(number_of_samples*hostile_node_percentage), replace=False)
    for node in nodes:
        name="y_train"+str(node)
        nodes_list.append(name)
    return nodes_list

def convert_nodes_to_hostile(y_dict, nodes_list,include_original_class=True,
                             converter_dict={0:9,1:7, 2:5,3:8, 4:6, 5:2, 6:4, 7:1, 8:3, 9:0}):
    
    for node in nodes_list:
        original_data=y_dict[node]
        converted_data=np.ones(y_dict[node].shape, dtype=int)*-1
        #labels_in_node=np.unique(original_data)
        labels_in_node=np.array(original_data)
        
        for i,label in enumerate(labels_in_node):
            #temporary_converter_choices=converter_choices.remove(label)
            #print(temporary_converter_choices)
            # if(include_original_class):
            #     converter_choices=[0,1,2]
            # else:
            #     if(label==1):
            #         converter_choices=[0,2]
            #     elif(label==0):
            #         converter_choices=[1,2]
            #     else:
            #         converter_choices=[0,1]
            #converted_data[original_data==label]=np.random.choice(converter_choices)
            #converted_data[i]=np.random.choice(converter_choices)
            #print(original_data,converted_data[])
            converted_data[original_data==label]=converter_dict[label]
        converted_data=(torch.tensor(converted_data)).type(torch.LongTensor)
        #print("node",node)
        #print(converted_data)
        y_dict.update({node:converted_data})
    #print(y_dict)
    return y_dict

def create_different_converters_for_each_attacker(y_dict, nodes_list, converters_seed):
    converters = dict()
    np.random.seed(converters_seed)
    converter_seeds_array = np.random.choice(5000, size=len(nodes_list), replace=False)

    for i in range(len(nodes_list)):
        unique_labels = np.unique(y_dict[nodes_list[i]])
        np.random.seed([converter_seeds_array[i]])
        subseeds = np.random.choice(1000, len(unique_labels), replace=False)

        conv = dict()
        for j in range(len(unique_labels)):
            choose_from = np.delete(np.arange(10), unique_labels[j])
            np.random.seed(subseeds[j])
            chosen = np.random.choice(choose_from, replace=False)
            conv[unique_labels[j]] = chosen
        converters.update({nodes_list[i]: conv})
    return converters

def convert_nodes_to_hostile_with_different_converters(y_dict, nodes_list, converters_seed=61):
    converters= create_different_converters_for_each_attacker(y_dict, nodes_list, converters_seed)
    y_dict_converted = y_dict.copy()
    for node in nodes_list:
        original_data=y_dict[node]
        converted_data=np.ones(y_dict[node].shape, dtype=int)*-1
        labels_in_node=np.unique(original_data)
        for label in labels_in_node:
            converted_data[original_data==label]=converters[node][label]
        converted_data=(torch.tensor(converted_data)).type(torch.LongTensor)
        y_dict_converted.update({node:converted_data})
    return y_dict_converted