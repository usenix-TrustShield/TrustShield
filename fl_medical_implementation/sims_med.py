# import the necessary packages
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
from fl_utils import distribute_data as dd
from fl_utils import train_nodes as tn
from fl_utils import construct_models as cm
from fl_utils import create_plot as cp
from fl_utils import ARFED_utils as arfu
import torchvision.models as models
import torch
import torch.nn as nn
import json
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler
from PIL import Image
import cv2
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict
import os
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd

def plot_multiple(y_maps):
        print("In this function")
        for item,value in y_maps.items():
            #print(value)
            plt.plot(range(1, len(value) + 1), value, '.-', label=item)
            #plt.plot(x,value,'.-', label=item)
        plt.legend()
        #plt.xlabel('%Adversarial clients')
        plt.xlabel('Communication rounds')
        plt.ylabel('Accuracy')
        plt.title("Training Accuracy vs Communication rounds")
        plt.savefig('arfed2.png')
def plot_multiple_acc(y_maps):
        print("In this function")
        for item,value in y_maps.items():
            #print(value)
            plt.plot(range(1, len(value) + 1), value, '.-', label=item)
            #plt.plot(x,value,'.-', label=item)
        plt.legend()
        #plt.xlabel('%Adversarial clients')
        plt.xlabel('Communication rounds')
        plt.ylabel('Accuracy')
        plt.title("Training Accuracy vs Communication Rounds")
        plt.savefig('arfed2.png')
def plot_single(y_maps,x):
        plt.figure()
        print("In this function")
        for item,value in y_maps.items():
            #plt.plot(range(1, len(value) + 1), value, '.-', label=item)
            plt.plot(x,value,'.-', label=item)
        plt.legend()
        #plt.xlabel('No of comm rounds')
        plt.xlabel('% Adversarial Clients')
        plt.ylabel('Accuracy')
        plt.title("Training Accuracy Curves for various adversarial percentages")
        plt.savefig('accuracyVsAdversarialUsers17.png')

def run_sim(opt, device):
    """
    A function to realize simulations for CIFAR10
    opt: the hyperparameters given by user
    device: the torch device for training and inference
    """
    print("device: ", device)

    number_of_samples = opt.numusers    # number of participants
    print(number_of_samples)
    number_of_validators = opt.numvals  # number of validator miners
    apply_mining = opt.applymining      # a boolean standing for validation will be applied
    validator_dist = opt.valdist        # the data distribution of the validator nodes
    validator_num_classes = opt.valn    # the number of classes validator node has
    is_noniid = opt.isnoniid            # boolean stands for edge users having iid or non iid distribution
    is_organized = opt.isorganized      # is label flipping attack is selects classes randomly or specified manner
    hostile_node_percentage = opt.advratio # malicious participant ratio
    iteration_num = opt.nrounds         # number of communication rounds
    learning_rate = opt.lr              # learning rate of the client "internal" process
    min_lr = opt.minlr                  # minimum learning rate obtained after decay of the learning rate 
    lr_scheduler_factor =opt.lrschfac   # the multiplier of the learning rate decay
    best_threshold=opt.bestth
    best_threshold = opt.bestth         # TODO
    clipping=opt.clipping               # boolean stands for clipping on the weights is going to be applied
    clipping_threshold =opt.clipth      # numerical threshold of the clipping applied

    weight_decay = opt.weightdecay      # the decay multiplier of the weight
    numEpoch = opt.nepoch               # the number of epoch that clients train model on their private data
    batch_size = opt.batchsize          # batch size of the client training
    momentum = opt.momentum             # the momentum of the client training 
    # some randomization seeds
    seed = opt.seed                     
    use_seed = opt.useseed              
    hostility_seed = opt.hostilityseed  
    converters_seed = opt.convertersseed

    train_amount = opt.trainamount      # The number of training samples
    test_amount = opt.testamount        # The number of test samples

    if is_noniid:                       # if the edge data is noniid then choose the num class each device will have at most and at least
        n = 3
        min_n_each_node = 1
    else:                               # o.w. it is 10 class per client and the label distribution is uniform accross different edge users
        n = 3
        min_n_each_node = 3

    x_train, y_train, x_test, y_test = dd.load_cifar_data() # load the central dataset
    #print("step1 done",len(x_train),len(x_test))
    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount) # assign new position to each label
    # the below function gets the mappings for each label for each classes, see the function comment
    #print("step2 done",label_dict_train)

    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    # the below function generates dataset for each client by considering the distributions provided as the outputs of the function above
    #print("step3 done",node_label_info_train,total_label_occurences_train,amount_info_table_train,sep='\n')

    x_train_dict, y_train_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_train,amount=train_amount,number_of_samples=number_of_samples,n=n, x_data=x_train,y_data=y_train,node_label_info=node_label_info_train,amount_info_table=amount_info_table_train,x_name="x_train",y_name="y_train")
    # selects some of the nodes to make them adversary

    if apply_mining:
        label_dict_validators = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
        node_label_info_validators, _, amount_info_table_validators = dd.get_info_for_validator_nodes(number_of_validators=number_of_validators, n=validator_num_classes, amount=test_amount, seed=use_seed, distribution=validator_dist)
        x_val_dict, y_val_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_validators,amount=test_amount,number_of_samples=number_of_validators,n=validator_num_classes, x_data=x_test,y_data=y_test,node_label_info=node_label_info_validators,amount_info_table=amount_info_table_validators,x_name="x_val",y_name="y_val")
    # TODO the validator implementation is valid till here

    nodes_list = dd.choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed)
    #print("step 5 done")
    print("hostile_nodes:", nodes_list)
    # following code provides label flipping attack in either ordered way or completely random
    if is_organized:
        if(hostile_node_percentage<=0.5):
            y_train_dict = dd.convert_nodes_to_hostile(y_train_dict, nodes_list,False,
                                                converter_dict={0: 1,
                                                                1: 2,
                                                                2: 0,
                                                                })
        else:
             y_train_dict = dd.convert_nodes_to_hostile(y_train_dict, nodes_list,True,
                                                converter_dict={0: 1,
                                                                1: 2,
                                                                2: 0,
                                                                })
             
    else:
        y_train_dict = dd.convert_nodes_to_hostile_with_different_converters(y_train_dict, nodes_list,
                                                                            converters_seed=converters_seed)

    ## apply the same functions applied to the training dataset
    label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
    node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples,n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    x_test_dict, y_test_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_test,amount=test_amount,number_of_samples=number_of_samples,n=n, x_data=x_test,y_data=y_test,node_label_info=node_label_info_test,amount_info_table=amount_info_table_test,x_name="x_test",y_name="y_test")
    # generate the datasets and loaders
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    #print("train_dl",train_dl)
    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)
    # x_test_dict['x_test0'] = x_test
    # y_test_dict['y_test0'] = y_test
    #Resnet
    main_model=models.resnet18(pretrained=True)
    num_ftrs = main_model.fc.in_features
    main_model.fc = nn.Linear(num_ftrs, 3)
    #Densenet
    # main_model = models.densenet121(pretrained=True)
    # for param in main_model.parameters():
    #     param.requires_grad = False
    # classifier = nn.Sequential(OrderedDict([
    # ('fc1', nn.Linear(1024, 512)),
    # ('relu', nn.LeakyReLU()),
    # ('fc2', nn.Linear(512, 3)),
    # ('output', nn.LogSoftmax(dim=1))]))
    # main_model.classifier = classifier
    # main_model.to(device)
    # main_criterion = nn.NLLLoss().to(device)
    # main_optimizer = optim.Adam(main_model.classifier.parameters(), lr=0.001)
    
    # model_info = models.densenet121(pretrained=True)
    # for param in model_info.parameters():
    #         param.requires_grad = False
    # classifier = nn.Sequential(OrderedDict([
    #     ('fc1', nn.Linear(1024, 512)),
    #     ('relu', nn.LeakyReLU()),
    #     ('fc2', nn.Linear(512, 3)),
    #     ('output', nn.LogSoftmax(dim=1))]))
    # model_info.classifier = classifier
    # main_model=model_info

    main_model.to(device)
    #main_model.fc = main_model.fc.cuda() if use_cuda else main_model.fc
    #main_model = cm.Cifar10CNN()        # generate model object for cloud server
    cm.weights_init(main_model)         # assign weights to the model to cloud server
    #main_model = main_model.to(device)  # assign model to the device to be processed by the cloud #TODO seperate devices for the cloud and each edge with additional dictionary
    # define hyper-parameters of the model training
    #print(main_model.parameters)
    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    main_criterion = nn.CrossEntropyLoss()
    #scheduler = lr_scheduler.ReduceLROnPlateau(main_optimizer, mode="max", factor=lr_scheduler_factor, patience=10, threshold=best_threshold, verbose=True, min_lr=min_lr)
    # the function below generates individual models for each client on the edge seperately
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_cifar_cnn(number_of_samples, learning_rate, momentum, device, weight_decay)


    test_accuracies_of_each_iteration = np.array([], dtype=float)


    mining_history = {
        "loss_hist" : [],
        "acc_hist"  : [],
        "miner_nodes": [],
        "validated_edges": [],
        "blacklisted_edges": []
    }
    
    # warranty_rounds = 5
    # warranty_cut = 0.1
    warranty_factor = 1.25
    initial_cut = np.round(1/n, 2)
    test_accuracy = 0
    initialization_rounds = 5 # 3 for iid case
    its = np.array([], dtype=int)
    distances = pd.DataFrame()
    thresholds = pd.DataFrame()

    for iteration in tqdm(range(iteration_num)): # communication rounds
        #print("Epoch",iteration)
        # send the cloud's current state to the edge users to initiate the federated learning for current communication round
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples)
        # start internal training of the edge users for current communication round
        tn.start_train_end_node_process_cifar(number_of_samples, x_train_dict, y_train_dict, x_test_dict,y_test_dict, batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, device, clipping, clipping_threshold)

        # perform validation 
        filtered_model_dict = model_dict.copy()
        # print("Iteration: " + str(iteration))
        # print("apply mining"+str(apply_mining))   
        if apply_mining and initialization_rounds<=iteration:
            mining_results_loss, mining_results_acc, miner_names, validated_edges, blacklisted_edges = tn.start_mining_validator_node_cifar(number_of_samples, number_of_validators, x_val_dict, y_val_dict, model_dict, criterion_dict, device, batch_size, iteration,validation_threshold=max(test_accuracy/warranty_factor, initial_cut), initialization_rounds =  initialization_rounds , config="one_to_all")
            #mining_results_loss, mining_results_acc, miner_names, validated_edges, blacklisted_edges = tn.start_mining_validator_node_cifar(number_of_samples, number_of_validators, x_val_dict, y_val_dict, model_dict, criterion_dict, device, batch_size, iteration,validation_threshold=max(test_accuracy/warranty_factor, initial_cut), initialization_rounds =  initialization_rounds , config="one_to_all")
        
            mining_history["loss_hist"].append(mining_results_loss)
            mining_history["acc_hist"].append(mining_results_acc)
            mining_history["miner_nodes"].append(miner_names)
            mining_history["validated_edges"].append(validated_edges.tolist())
            mining_history["blacklisted_edges"].append(blacklisted_edges.tolist())
        
            for adv in blacklisted_edges: #  filter the model_dict to give 0 for poisonous ones
                # TODO should we remove adverserial nodes completely or sparsely
                # currently it is not deleting at all
                _ = filtered_model_dict.pop(adv)
            #print("models being considered for update",filtered_model_dict)    

        # aggregation mechanism is introduced by the function below
        if len(filtered_model_dict):
            main_model = tn.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, filtered_model_dict, device)
        # testing the federated model on central server test dataset
        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device,set_evaluation_flag=False)
        #scheduler.step(test_accuracy)
        new_lr = main_optimizer.param_groups[0]["lr"]
        optimizer_dict = cm.update_learning_rate_decay(optimizer_dict, new_lr) # adjust lr for all clients according to the cloud test set accuracy, TODO can be performed as intervalidator function as well since there will be no central server testing dataset
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))
    print({str(hostile_node_percentage):test_accuracies_of_each_iteration})
    # print("DISTANCES")
    # print(distances)
    # print("THRESHOLDS")
    # print(thresholds)
    #if(apply_mining):
    #plot_multiple_acc({str(hostile_node_percentage):test_accuracies_of_each_iteration})
    #np.save("./histories/FLCIFAR10_mining_"+str(apply_mining)+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_organized_numclients_100_ratio_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)
    with open("./histories/FLMEDICAL/hist_new_new_non_arfed_medfl_numepoch"+str(opt.nepoch)+"_lr_"+str(opt.lr)+"_isnoniid_"+str(opt.isnoniid)+"_numclients_"+str(number_of_samples)+"_ismining_"+str(opt.applymining)+"_valdist_"+opt.valdist+"_ratio_"+str(hostile_node_percentage)+".txt", "w") as fp:
        json.dump(mining_history, fp)  # encode dict into JSON
    print("Simulation is complete!")
    #return test_accuracies_of_each_iteration[-1]
    np.save("./histories/FLMEDICAL/FLMEDICAL_mining_mining_"+str(opt.applymining)+"_noniidedge_"+str(opt.isnoniid)+"_"+opt.valdist+"_ratio_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)



if __name__ == "__main__":

    #Parameters for the simulation

    parser = argparse.ArgumentParser()
    parser.add_argument("--deviceno", type=int, default=2)
    parser.add_argument("--numusers",type=int,default=50)
    parser.add_argument("--isnoniid", type=int,default=1)
    parser.add_argument("--isorganized", type=int, default=1)
    parser.add_argument("--advratio", type=float, default=0.5)
    parser.add_argument("--nrounds", type=int,default=20)
    
    parser.add_argument("--trainamount", type=int,default= 450)
    parser.add_argument("--testamount", type=int,default=50)
    parser.add_argument("--nepoch", type=int, default=1)
    parser.add_argument("--batchsize", type=int,default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--minlr", type=float, default=0.000010)
    parser.add_argument("-lrschfac", type=float, default=0.2)

    #miner parameters
    parser.add_argument("--applymining", type=int, default=1)
    parser.add_argument("--numvals", type=int, default=3)
    parser.add_argument("--valdist", type=str, default="gaussian")
    parser.add_argument("--valn", type=int, default=3)

    parser.add_argument("--bestth", type=float, default=0.0001)
    parser.add_argument("--clipping", type=int, default=1)
    parser.add_argument("--clipth",type=int, default=10)
    parser.add_argument("--weightdecay", type=float, default=0.001)    
    parser.add_argument("--momentum",type=float, default= 0.9)

    parser.add_argument("--seed", type=int,default=11)
    parser.add_argument("--useseed", type=int,default=8)
    parser.add_argument("--hostilityseed", type=int,default=90)
    parser.add_argument("--convertersseed", type=int,default= 51)


    opt = parser.parse_args()

    device = torch.device("cuda:"+str(opt.deviceno) if (torch.cuda.is_available()) else "cpu")
    #adv=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

    #adv=[0.0,0.2,0.4]
    #adv=[0.5,0.8]
    #adv=[0.6]
    #adv=[0.1,0.3,0.5,0.7,0.8]
    #adv=[0.4,0.5,0.6,0.7,]
    #adv=[0.5]
    # y_maps={}
    # final_test_accuracies=[]
    #apply_mining=[0]
    #for flags in apply_mining:
    #opt.applymining=flags
    # final_test_accuracies=[]
    # for elem in adv:
    #         opt.advratio=elem
    #final_test_accuracies.append(run_sim(opt,device))
    #y_maps[flags]=final_test_accuracies.copy()
        #print("y maps",y_maps)
    #plot_multiple(y_maps)
    run_sim(opt,device)
    #print("final_test_accuracies",final_test_accuracies)

    #plot_single(y_maps,adv)

    
    #plot_single({'lr=0.001':final_test_accuracies},adv)