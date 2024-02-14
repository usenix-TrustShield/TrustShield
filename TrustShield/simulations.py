# import the necessary packages
import numpy as np
import torch
import json
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.models as models
import argparse
from TrustShield.fl_utils import distribute_data as dd
from TrustShield.fl_utils import train_nodes as tn
from TrustShield.fl_utils import construct_models as cm
from TrustShield.fl_utils import ARFED_utils as arfu
from tqdm import tqdm
from transformers   import AdamW
import pandas as pd
import copy
import os
from sklearn.utils.class_weight import compute_class_weight

def cifar10_run_sim(opt, device, isdict=False, verbose=False):
    """
    A function to realize simulations for CIFAR10
    opt: the hyperparameters given by user
    device: the torch device for training and inference
    """
    print("device: ", device)

    if isdict:
        # get the hyperparameters from the dictionary
        number_of_samples = opt["numusers"]    # number of participants
        number_of_validators = opt["numvals"]   # number of validator miners
        apply_mining = opt["applymining"]      # a boolean standing for validation will be applied
        validator_dist = opt["valdist"]        # the data distribution of the validator nodes
        validator_num_classes = opt["valn"]     # the number of classes validator node has
        is_noniid = opt["isnoniid"]          # boolean stands for edge users having iid or non iid distribution
        is_organized = opt["isorganized"]      # is label flipping attack is selects classes randomly or specified manner
        hostile_node_percentage = opt["advratio"]  # malicious participant ratio
        iteration_num = opt["nrounds"]          # number of communication rounds
        learning_rate = opt["lr"]          # learning rate of the client "internal" process
        min_lr = opt["minlr"]              # minimum learning rate obtained after decay of the learning rate 
        lr_scheduler_factor =opt["lrschfac"]    # the multiplier of the learning rate decay
        best_threshold = opt["bestth"]       
        clipping=opt["clipping"]            # boolean stands for clipping on the weights is going to be applied
        clipping_threshold =opt["clipth"]       # numerical threshold of the clipping applied
        initialization_rounds = opt["initround"] 
        weight_decay = opt["weightdecay"]       # the decay multiplier of the weight
        numEpoch = opt["nepoch"]                # the number of epoch that clients train model on their private data
        batch_size = opt["batchsize"]           # batch size of the client training
        momentum = opt["momentum"]              # the momentum of the client training 
        # some randomization seeds
        seed = opt["seed"]                      
        use_seed = opt["useseed"]               
        hostility_seed = opt["hostilityseed"]  
        converters_seed = opt["convertersseed"] 
        warranty_factor = opt["warfact"] 
        initial_cut = opt["initcut"] 
        unbias_mechanism = opt["unbiasmech"] 
        train_amount = opt["trainamount"]       # The number of training samples
        test_amount = opt["testamount"]         # The number of test samples
        # 10 class per client and the label distribution is uniform accross different edge users
        n = opt["numofclasses"]  # TODO 10
        min_n_each_node = opt["numofclasses"] 
        attack = opt["attack"]
        applyarfed = opt["applyarfed"]
        
    else:
        # get the hyperparameters from the dictionary
        number_of_samples = opt.numusers    # number of participants
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
        best_threshold = opt.bestth         
        clipping=opt.clipping               # boolean stands for clipping on the weights is going to be applied
        clipping_threshold =opt.clipth      # numerical threshold of the clipping applied
        initialization_rounds = opt.initround
        weight_decay = opt.weightdecay      # the decay multiplier of the weight
        numEpoch = opt.nepoch               # the number of epoch that clients train model on their private data
        batch_size = opt.batchsize          # batch size of the client training
        momentum = opt.momentum             # the momentum of the client training 
        # some randomization seeds
        seed = opt.seed                     
        use_seed = opt.useseed              
        hostility_seed = opt.hostilityseed  
        converters_seed = opt.convertersseed
        warranty_factor = opt.warfact
        initial_cut = opt.initcut
        unbias_mechanism = opt.unbiasmech
        train_amount = opt.trainamount      # The number of training samples
        test_amount = opt.testamount        # The number of test samples
        # 10 class per client and the label distribution is uniform accross different edge users
        n = opt.numofclasses # TODO 10
        min_n_each_node = opt.numofclasses
        attack = opt.attack
        applyarfed = opt.applyarfed
    if attack == "collective":
        if isdict:
            converter_dict = opt["converterdict"]
        else: converter_dict = opt.converterdict
    else:
        converter_dict = {0: np.random.randint(10) ,1: np.random.randint(10),2: np.random.randint(10),3: np.random.randint(10),4: np.random.randint(10),5: np.random.randint(10),6: np.random.randint(10),7: np.random.randint(10),8: np.random.randint(10),9: np.random.randint(10)}
    
    if is_noniid: # if the edge data is noniid then choose the num class each device will have at most and at least
        if isdict:
            edge_sparsity_ratio = opt["sparsityratio"]
        else:                      
            edge_sparsity_ratio = opt.sparsityratio
        min_n_each_node = round(n*edge_sparsity_ratio)
        n = round(n*edge_sparsity_ratio)
    # setup the federated learning environment
    x_train, y_train, x_test, y_test = dd.load_cifar_data() # load the central dataset
    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount) # assign new position to each label
    # the below function gets the mappings for each label for each classes, see the function comment
    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    # the below function generates dataset for each client by considering the distributions provided as the outputs of the function above
    x_train_dict, y_train_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_train,amount=train_amount,number_of_samples=number_of_samples,n=n, x_data=x_train,y_data=y_train,node_label_info=node_label_info_train,amount_info_table=amount_info_table_train,x_name="x_train",y_name="y_train")
    # selects some of the nodes to make them adversary
    nodes_list = dd.choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed)
    # if verbose:
    print("hostile_nodes:", nodes_list)
    # following code provides label flipping attack in either ordered way or completely random
    if is_organized:
        y_train_dict = dd.convert_nodes_to_hostile(y_train_dict, nodes_list, converter_dict = converter_dict)
    else:
        y_train_dict = dd.convert_nodes_to_hostile_with_different_converters(y_train_dict, nodes_list, converters_seed=converters_seed)

    ## apply the same functions applied to the training dataset to form validaton set
    
    label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
    node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples,n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    x_test_dict, y_test_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_test,amount=test_amount,number_of_samples=number_of_samples,n=n, x_data=x_test,y_data=y_test,node_label_info=node_label_info_test,amount_info_table=amount_info_table_test,x_name="x_test",y_name="y_test")
    
    if apply_mining:
        label_dict_validators = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
        node_label_info_validators, _, amount_info_table_validators = dd.get_info_for_validator_nodes(number_of_validators=number_of_validators, n=validator_num_classes, amount=test_amount, seed=use_seed, distribution=validator_dist)
        x_val_dict, y_val_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_validators,amount=test_amount,number_of_samples=number_of_validators,n=validator_num_classes, x_data=x_test,y_data=y_test,node_label_info=node_label_info_validators,amount_info_table=amount_info_table_validators,x_name="x_val",y_name="y_val")
    # generate the datasets and loaders
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    main_model = cm.Cifar10CNN()        # generate model object for cloud server
    cm.weights_init(main_model)         # assign weights to the model to cloud server
    main_model = main_model.to(device)  # assign model to the device to be processed by the cloud #TODO seperate devices for the cloud and each edge with additional dictionary
    # define hyper-parameters of the model training
    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    main_criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(main_optimizer, mode="max", factor=lr_scheduler_factor, patience=10, threshold=best_threshold, verbose=True, min_lr=min_lr)
    # the function below generates individual models for each client on the edge seperately
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_cifar_cnn(number_of_samples, learning_rate, momentum, device, weight_decay)

    mining_history = {
        "loss_hist" : [],
        "acc_hist"  : [],
        "miner_nodes": [],
        "validated_edges": [],
        "blacklisted_edges": []
    }

    test_accuracy = 0
    test_accuracies_of_each_iteration = np.array([], dtype=float)
    its = np.array([], dtype=int)
    distances = pd.DataFrame()
    thresholds = pd.DataFrame()
    # test_accuracy_array = [np.round(2/n, 2)+warranty_cut for i in range(warranty_rounds)]

    for iteration in tqdm(range(iteration_num)): # communication rounds
        # send the cloud's current state to the edge users to initiate the federated learning for current communication round
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples)
        # start internal training of the edge users for current communication round
        tn.start_train_end_node_process_cifar(number_of_samples, x_train_dict, y_train_dict, x_test_dict,y_test_dict, batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, device, clipping, clipping_threshold)
        
        if applyarfed:
            iteration_distance = arfu.calculate_euclidean_distances(main_model, model_dict)
            iteration_distance, iteration_threshold = arfu.get_outlier_situation_and_thresholds_for_layers(iteration_distance)
            thresholds = pd.concat([thresholds, iteration_threshold])
            distances = pd.concat([distances, iteration_distance])
            main_model = arfu.strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, iteration_distance, device)
        
        # perform validation 
        filtered_model_dict = model_dict.copy()
        if apply_mining and initialization_rounds<=iteration:
            if is_noniid:
                if unbias_mechanism: # unbias the models and change the edge models dictionary accordingly
                    for client in range(number_of_samples):
                        model_dict["model" + str(client)] = tn.unbias_noniid_edge_update(main_model, model_dict["model" + str(client)], device)
                else:
                    print("WARNING: There is no unbiasing mechanism used!")
            mining_results_loss, mining_results_acc, miner_names, validated_edges, blacklisted_edges = tn.start_mining_validator_node_cifar(number_of_samples, number_of_validators, x_val_dict, y_val_dict, model_dict, criterion_dict, device, batch_size, iteration,validation_threshold=max(test_accuracy/warranty_factor, initial_cut), initialization_rounds =  initialization_rounds , config="one_to_all")
        
            mining_history["loss_hist"].append(mining_results_loss)
            mining_history["acc_hist"].append(mining_results_acc)
            mining_history["miner_nodes"].append(miner_names)
            mining_history["validated_edges"].append(validated_edges.tolist())
            mining_history["blacklisted_edges"].append(blacklisted_edges.tolist())
        
            for adv in blacklisted_edges: #  filter the model_dict to give 0 for poisonous ones
                # TODO should we remove adverserial nodes completely or sparsely based on the behavior history
                # currently it is not deleting at all
                _ = filtered_model_dict.pop(adv)    

        # aggregation mechanism is introduced by the function below
        if len(filtered_model_dict) and not applyarfed:
                main_model = tn.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, filtered_model_dict, device)
        # testing the federated model on central server test dataset
        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)
        #scheduler.step(test_accuracy)
        # test_accuracy_array.append(test_accuracy) if iteration>initialization_rounds else None
        new_lr = main_optimizer.param_groups[0]["lr"]
        optimizer_dict = cm.update_learning_rate_decay(optimizer_dict, new_lr) # adjust lr for all clients according to the cloud test set accuracy, TODO can be performed as intervalidator function as well since there will be no central server testing dataset
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        if verbose:
            print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

    if not os.path.exists("histories"):
        os.makedirs("histories")
    if apply_mining:
        method = "tshield"
    elif applyarfed:
        method = "arfed"
    else:
        method = "vanillafl"
    np.save("./histories/FLCIFAR10_"+method+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_ratio_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)
    with open("./histories/hist_FLCIFAR10_"+method+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_initround_"+str(initialization_rounds)+"_initcut_"+str(initial_cut)+"_warfact_"+str(warranty_factor)+"_internalepoch_"+str(numEpoch)+"_unbiasing_"+str(unbias_mechanism)+"_minernum_"+str(number_of_validators)+"_numclients_"+str(number_of_samples)+"_ratio_"+str(hostile_node_percentage)+".txt", "w") as fp:
        json.dump(mining_history, fp)  # encode dict into JSON
    print("Simulation is complete!")


def mnist_run_sim(opt, device, isdict=False, verbose=False):
    """
    A function to realize simulations for MNIST
    opt: the hyperparameters given by user
    device: the torch device for training and inference
    """
    print("device: ", device)

    if isdict:
        # get the hyperparameters from the dictionary
        number_of_samples = opt["numusers"]    # number of participants
        number_of_validators = opt["numvals"]   # number of validator miners
        apply_mining = opt["applymining"]      # a boolean standing for validation will be applied
        validator_dist = opt["valdist"]        # the data distribution of the validator nodes
        validator_num_classes = opt["valn"]     # the number of classes validator node has
        is_noniid = opt["isnoniid"]          # boolean stands for edge users having iid or non iid distribution
        is_organized = opt["isorganized"]      # is label flipping attack is selects classes randomly or specified manner
        hostile_node_percentage = opt["advratio"]  # malicious participant ratio
        iteration_num = opt["nrounds"]          # number of communication rounds
        learning_rate = opt["lr"]          # learning rate of the client "internal" process
        initialization_rounds = opt["initround"] 
        # the decay multiplier of the weight
        numEpoch = opt["nepoch"]                # the number of epoch that clients train model on their private data
        batch_size = opt["batchsize"]           # batch size of the client training
        momentum = opt["momentum"]              # the momentum of the client training 
        # some randomization seeds
        seed = opt["seed"]                      
        use_seed = opt["useseed"]               
        hostility_seed = opt["hostilityseed"]  
        converters_seed = opt["convertersseed"] 
        warranty_factor = opt["warfact"] 
        initial_cut = opt["initcut"] 
        unbias_mechanism = opt["unbiasmech"] 
        train_amount = opt["trainamount"]       # The number of training samples
        test_amount = opt["testamount"]         # The number of test samples
        # 10 class per client and the label distribution is uniform accross different edge users
        is_cnn = opt["iscnn"] # TODO in mnist multiple architectures are possible
        n = opt["numofclasses"]  # TODO 10
        min_n_each_node = opt["numofclasses"]
        attack = opt["attack"]
        applyarfed = opt["applyarfed"]
        
    else:
        number_of_samples = opt.numusers    # number of participants
        number_of_validators = opt.numvals  # number of validator miners
        apply_mining = opt.applymining      # a boolean standing for validation will be applied
        validator_dist = opt.valdist        # the data distribution of the validator nodes
        validator_num_classes = opt.valn    # the number of classes validator node has
        is_noniid = opt.isnoniid            # boolean stands for edge users having iid or non iid distribution
        is_organized = opt.isorganized      # is label flipping attack is selects classes randomly or specified manner
        hostile_node_percentage = opt.advratio # malicious participant ratio
        iteration_num = opt.nrounds         # number of communication rounds
        learning_rate = opt.lr              # learning rate of the client "internal" process
        initialization_rounds = opt.initround
        numEpoch = opt.nepoch               # the number of epoch that clients train model on their private data
        batch_size = opt.batchsize          # batch size of the client training
        momentum = opt.momentum             # the momentum of the client training 
        # some randomization seeds
        seed = opt.seed                     
        use_seed = opt.useseed              
        hostility_seed = opt.hostilityseed  
        converters_seed = opt.convertersseed
        warranty_factor = opt.warfact
        initial_cut = opt.initcut
        unbias_mechanism = opt.unbiasmech
        train_amount = opt.trainamount      # The number of training samples
        test_amount = opt.testamount        # The number of test samples
        is_cnn = opt.iscnn
        n = opt.numofclasses
        min_n_each_node = opt.num_of_classes
        attack = opt.attack
        applyarfed = opt.applyarfed
    
    if attack == "collective":
        if isdict:
            converter_dict = opt["converterdict"]
        else: converter_dict = opt.converterdict
    else:
        converter_dict = {0: np.random.randint(10) ,1: np.random.randint(10),2: np.random.randint(10),3: np.random.randint(10),4: np.random.randint(10),5: np.random.randint(10),6: np.random.randint(10),7: np.random.randint(10),8: np.random.randint(10),9: np.random.randint(10)}
 
    # converter_dict = {0: 9, 1: 7,2: 5, 3: 8, 4: 6, 5: 2, 6: 4, 7: 1, 8: 3,9: 0}

    if is_noniid: # if the edge data is noniid then choose the num class each device will have at most and at least
        if isdict:
            edge_sparsity_ratio = opt["sparsityratio"]
        else:                      
            edge_sparsity_ratio = opt.sparsityratio
        min_n_each_node = round(n*edge_sparsity_ratio)
        n = round(n*edge_sparsity_ratio)

    x_train, y_train, x_valid, y_valid, x_test, y_test = dd.load_mnist_data() # load the central dataset

    # The following two lines are necessary?
    x_test, y_test = dd.get_equal_size_test_data_from_each_label(x_test, y_test, min_amount=test_amount) # TODO
    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor,(x_train, y_train, x_valid, y_valid, x_test, y_test)) # TODO

    ##train
    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount)# assign new position to each label
    # the below function gets the mappings for each label for each classes, see the function comment
    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    # the below function generates dataset for each client by considering the distributions provided as the outputs of the function above
    x_train_dict, y_train_dict = dd.distribute_mnist_data_to_participants(label_dict=label_dict_train,amount=train_amount,number_of_samples=number_of_samples,n=n, x_data=x_train,y_data=y_train,node_label_info=node_label_info_train,amount_info_table=amount_info_table_train,x_name="x_train",y_name="y_train",is_cnn=is_cnn)
    # selects some of the nodes to make them adversary
    nodes_list = dd.choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed)
    print("hostile_nodes:", nodes_list)
    # following code provides label flipping attack in either ordered way or completely random
    if is_organized:
        y_train_dict = dd.convert_nodes_to_hostile(y_train_dict, nodes_list, converter_dict=converter_dict)
    else:
        y_train_dict = dd.convert_nodes_to_hostile_with_different_converters(y_train_dict, nodes_list, converters_seed=converters_seed)


    ## apply the same functions applied to the training dataset to form validaton set
    label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
    node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples,n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    x_test_dict, y_test_dict = dd.distribute_mnist_data_to_participants(label_dict=label_dict_test,amount=test_amount,number_of_samples=number_of_samples,n=n, x_data=x_test,y_data=y_test,node_label_info=node_label_info_test,amount_info_table=amount_info_table_test,x_name="x_test",y_name="y_test", is_cnn=is_cnn)


    if apply_mining:
        label_dict_validators = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
        node_label_info_validators, _, amount_info_table_validators = dd.get_info_for_validator_nodes(number_of_validators=number_of_validators, n=validator_num_classes, amount=test_amount, seed=use_seed, distribution=validator_dist)
        x_val_dict, y_val_dict = dd.distribute_mnist_data_to_participants(label_dict=label_dict_validators,amount=test_amount,number_of_samples=number_of_validators,n=validator_num_classes, x_data=x_test,y_data=y_test,node_label_info=node_label_info_validators,amount_info_table=amount_info_table_validators,x_name="x_val",y_name="y_val")

    
    # generate the datasets and loaders
    if is_cnn:
        reshape_size = int(np.sqrt(x_train.shape[1]))
        x_train = x_train.view(-1, 1, reshape_size, reshape_size)
        x_valid = x_valid.view(-1, 1, reshape_size, reshape_size)
        x_test = x_test.view(-1, 1, reshape_size, reshape_size)

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    # generate model object for cloud server
    if is_cnn:
        main_model = cm.Netcnn()
    else:
        main_model = cm.Net2nn()
    # assign weights to the model to cloud server
    cm.weights_init(main_model)
    main_model = main_model.to(device) # assign model to the device to be processed by the cloud
    # define hyper-parameters of the model training
    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=0.9)
    main_criterion = nn.CrossEntropyLoss()
    # the function below generates individual models for each client on the edge seperately
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_mnist(number_of_samples, learning_rate,momentum, device, is_cnn)

    test_accuracy = 0
    test_accuracies_of_each_iteration = np.array([], dtype=float)
    its = np.array([], dtype=int)
    distances = pd.DataFrame()
    thresholds = pd.DataFrame()
    
    mining_history = {
        "loss_hist" : [],
        "acc_hist"  : [],
        "miner_nodes": [],
        "validated_edges": [],
        "blacklisted_edges": []
    }

    test_accuracy = 0

    for iteration in  tqdm(range(iteration_num)):  # communication rounds
        # send the cloud's current state to the edge users to initiate the federated learning for current communication round
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict,number_of_samples)
        # start internal training of the edge users for current communication round
        tn.start_train_end_node_process_without_print(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict, batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, device)
        if applyarfed:
            iteration_distance = arfu.calculate_euclidean_distances(main_model, model_dict)
            iteration_distance, iteration_threshold = arfu.get_outlier_situation_and_thresholds_for_layers(iteration_distance)
            thresholds = pd.concat([thresholds, iteration_threshold])
            distances = pd.concat([distances, iteration_distance])
            main_model = arfu.strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(main_model,
                                                                        model_dict, iteration_distance, device)
        # perform validation 
        filtered_model_dict = model_dict.copy()
        if apply_mining and initialization_rounds<=iteration:
            if is_noniid:
                if unbias_mechanism: # unbias the models and change the edge models dictionary accordingly
                    for client in range(number_of_samples):
                        model_dict["model" + str(client)] = tn.unbias_noniid_edge_update(main_model, model_dict["model" + str(client)], device)
                else:
                    print("WARNING: There is no unbiasing mechanism used!")
            mining_results_loss, mining_results_acc, miner_names, validated_edges, blacklisted_edges = tn.start_mining_validator_node_mnist(number_of_samples, number_of_validators, x_val_dict, y_val_dict, model_dict, criterion_dict, device, batch_size, iteration,validation_threshold=max((test_accuracy-0.05)*warranty_factor, initial_cut), initialization_rounds =  initialization_rounds , config="one_to_all")
            #mining_results_loss, mining_results_acc, miner_names, validated_edges, blacklisted_edges = tn.start_mining_validator_node_mnist(number_of_samples, number_of_validators, x_val_dict, y_val_dict, model_dict, criterion_dict, device, batch_size, iteration,validation_threshold=initial_cut, initialization_rounds =  initialization_rounds , config="one_to_all")
            #print(validated_edges)
            mining_history["loss_hist"].append(mining_results_loss)
            mining_history["acc_hist"].append(mining_results_acc)
            mining_history["miner_nodes"].append(miner_names)
            mining_history["validated_edges"].append(validated_edges.tolist())
            mining_history["blacklisted_edges"].append(blacklisted_edges.tolist())
        
        
            for adv in blacklisted_edges: #  filter the model_dict to give 0 for poisonous ones
                # TODO should we remove adverserial nodes completely or sparsely
                # currently it is not deleting at all
                _ = filtered_model_dict.pop(adv)  

        # aggregation mechanism is introduced by the function below
        if len(filtered_model_dict) and not applyarfed:
            main_model = tn.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, filtered_model_dict, device)
        # testing the federated model on central server test dataset
        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        if verbose:
            print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

    if not os.path.exists("histories"):
        os.makedirs("histories")
    if apply_mining:
        method = "tshield"
    elif applyarfed:
        method = "arfed"
    else:
        method = "vanillafl"
    np.save("./histories/MNIST_"+method+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_initround_"+str(initialization_rounds)+"_initcut_"+str(initial_cut)+"_warfact_"+str(warranty_factor)+"_unbiasing_"+str(unbias_mechanism)+"_minernum_"+str(number_of_validators)+"_ratio_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)
    with open("./histories/hist_MNIST_"+method+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_initround_"+str(initialization_rounds)+"_initcut_"+str(initial_cut)+"_warfact_"+str(warranty_factor)+"_unbiasing_"+str(unbias_mechanism)+"_minernum_"+str(number_of_validators)+"_ratio_"+str(hostile_node_percentage)+".txt", "w") as fp:
        json.dump(mining_history, fp)  # encode dict into JSON
    print("Simulation is complete!")


def nlp_run_sim(opt, device, isdict=False, verbose=False):
    """
    A function to realize simulations for any classification task
    opt: the hyperparameters given by user
    device: the torch device for training and inference
    """
    print("device: ", device)

    if isdict:
        # get the hyperparameters from the dictionary
        number_of_samples = opt["numusers"]    # number of participants
        number_of_validators = opt["numvals"]   # number of validator miners
        apply_mining = opt["applymining"]      # a boolean standing for validation will be applied
        validator_dist = opt["valdist"]        # the data distribution of the validator nodes
        validator_num_classes = opt["valn"]     # the number of classes validator node has
        is_noniid = opt["isnoniid"]          # boolean stands for edge users having iid or non iid distribution
        is_organized = opt["isorganized"]      # is label flipping attack is selects classes randomly or specified manner
        hostile_node_percentage = opt["advratio"]  # malicious participant ratio
        iteration_num = opt["nrounds"]          # number of communication rounds
        learning_rate = opt["lr"]          # learning rate of the client "internal" process
        # min_lr = opt["minlr"]              # minimum learning rate obtained after decay of the learning rate 
        # lr_scheduler_factor =opt["lrschfac"]    # the multiplier of the learning rate decay
        # best_threshold = opt["bestth"]       
        clipping=opt["clipping"]            # boolean stands for clipping on the weights is going to be applied
        clipping_threshold =opt["clipth"]       # numerical threshold of the clipping applied
        initialization_rounds = opt["initround"] 
        weight_decay = opt["weightdecay"]       # the decay multiplier of the weight
        numEpoch = opt["nepoch"]                # the number of epoch that clients train model on their private data
        batch_size = opt["batchsize"]           # batch size of the client training
        momentum = opt["momentum"]              # the momentum of the client training 
        # some randomization seeds
        seed = opt["seed"]                      
        use_seed = opt["useseed"]               
        hostility_seed = opt["hostilityseed"]  
        converters_seed = opt["convertersseed"] 
        warranty_factor = opt["warfact"] 
        initial_cut = opt["initcut"] 
        unbias_mechanism = opt["unbiasmech"] 
        train_amount = opt["trainamount"]       # The number of training samples
        test_amount = opt["testamount"]         # The number of test samples
        n = opt["numofclasses"]  
        min_n_each_node = opt["numofclasses"] 
        attack = opt["attack"]
        applyarfed = opt["applyarfed"]
        # ood_ratio               = opt["ood_ratio"]         # percentage of out of distribution data
        train_number_classes    = opt["train_num_classes"] # number of classes in the training set
        test_num_classes        = opt["test_num_classes"]  # number of classes of the cloud test set
    else:
        number_of_samples       = opt.numusers          # number of participants
        number_of_validators    = opt.numvals           # number of validator miners
        apply_mining            = opt.applymining       # a boolean standing for validation will be applied
        validator_dist          = opt.valdist           # the data distribution of the validator nodes
        validator_num_classes   = opt.valn              # the number of classes validator node has
        is_noniid               = opt.isnoniid          # boolean stands for edge users having iid or non iid distribution
        is_organized            = opt.isorganized       # is label flipping attack is selects classes randomly or specified manner
        hostile_node_percentage = opt.advratio          # malicious participant ratio
        iteration_num           = opt.nrounds           # number of communication rounds
        learning_rate           = opt.lr                # learning rate of the client "internal" process
        # min_lr                  = opt.minlr             # minimum learning rate obtained after decay of the learning rate 
        # lr_scheduler_factor     = opt.lrschfac          # the multiplier of the learning rate decay
        # best_threshold          = opt.bestth         
        clipping                = opt.clipping          # boolean stands for clipping on the weights is going to be applied
        clipping_threshold      = opt.clipth            # numerical threshold of the clipping applied
        initialization_rounds   = opt.initround
        weight_decay            = opt.weightdecay       # the decay multiplier of the weight
        numEpoch                = opt.nepoch            # the number of epoch that clients train model on their private data
        batch_size              = opt.batchsize         # batch size of the client training
        momentum                = opt.momentum          # the momentum of the client training 
        #ood_ratio               = opt.ood_ratio         # percentage of out of distribution data
        train_number_classes    = opt.train_num_classes # number of classes in the training set
        test_num_classes        = opt.test_num_classes  # number of classes of the cloud test set
        n = opt.numofclasses  
        min_n_each_node = opt.numofclasses
        attack = opt.attack
        applyarfed = opt.applyarfed
        # some randomization seeds
        seed             = opt.seed                     
        use_seed         = opt.useseed              
        hostility_seed   = opt.hostilityseed  
        converters_seed  = opt.convertersseed
        warranty_factor  = opt.warfact
        initial_cut      = opt.initcut
        unbias_mechanism = opt.unbiasmech
        train_amount     = opt.trainamount      # The number of training samples
        test_amount      = opt.testamount       # The number of test samples

    if attack == "collective":
        if isdict:
            converter_dict = opt["converterdict"] 
        else:
            converter_dict = opt.converterdict
    else:
        converter_dict = {0: np.random.randint(4) ,1: np.random.randint(4),2: np.random.randint(4),3: np.random.randint(4)} 

    if is_noniid: # if the edge data is noniid then choose the num class each device will have at most and at least
        if isdict:
            edge_sparsity_ratio = opt["sparsityratio"]
        else:                      
            edge_sparsity_ratio = opt.sparsityratio
            
        min_n_each_node = round(min_n_each_node*edge_sparsity_ratio)
        n = round(n*edge_sparsity_ratio) # 4 x 0.5


    # # Set number of classes
    # n = 4
    # if is_noniid:                       # if the edge data is noniid then choose the num class each device will have at most and at least
    #     min_n_each_node = 2
    # else:                               # o.w. it is 10 class per client and the label distribution is uniform accross different edge use
    #     min_n_each_node = 4
    # {0: 1,1: 0,2: 3,3: 2}

    file = './data/nlpdata/ecommerceDataset.csv'

    x_train, mask_train, y_train, x_val, mask_val, y_val, x_test, mask_test, y_test = dd.load_nlp_data(file, new=True) # load the central dataset
    label_dict_train = dd.split_and_shuffle_labels_nlp(y_data=y_train,seed=seed,amount=train_amount,n=n) # assign new position to each label
    # the below function gets the mappings for each label for each classes, see the function comment
    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples,n=n,amount=train_amount,seed=use_seed,min_n_each_node=min_n_each_node)
    x_train_dict, mask_train_dict, y_train_dict = dd.distribute_nlp_data_to_participants(label_dict=label_dict_train,amount=train_amount,number_of_samples=number_of_samples,n=train_number_classes,x_data=x_train,mask_data=mask_train,y_data=y_train,node_label_info=node_label_info_train,amount_info_table=amount_info_table_train,x_name="x_train",mask_name="mask_train",y_name="y_train",randomization_flag=True)
    # selects some of the nodes to make them adversary
    nodes_list = dd.choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed)
    print("Hostile_nodes:", nodes_list)
    # following code provides label flipping attack in either ordered way or completely random
    if is_organized:
        y_train_dict = dd.convert_nodes_to_hostile(y_train_dict,nodes_list,converter_dict=converter_dict)
    else:
        y_train_dict = dd.convert_nodes_to_hostile_with_different_converters(y_train_dict, nodes_list, converters_seed=converters_seed)
    ## apply the same functions applied to the training dataset to form validaton set
    label_dict_test = dd.split_and_shuffle_labels_nlp(y_data=y_test, seed=seed, amount=test_amount,n=test_num_classes)
    
    node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples,   n=test_num_classes,   amount=test_amount,   seed=use_seed,   min_n_each_node=test_num_classes)
        
    x_test_dict, mask_test_dict, y_test_dict = dd.distribute_nlp_data_to_participants(label_dict=label_dict_test,   amount=test_amount,   number_of_samples=number_of_samples,   n=test_num_classes,   x_data=x_test,   mask_data=mask_test,   y_data=y_test,   node_label_info=node_label_info_test,   amount_info_table=amount_info_table_test,   x_name="x_test",   mask_name="mask_test",   y_name="y_test",   randomization_flag=False)

    if apply_mining:
        label_dict_validators = dd.split_and_shuffle_labels_nlp(y_data=y_test, seed=seed, amount=test_amount, n=n)
        node_label_info_validators, _, amount_info_table_validators = dd.get_info_for_validator_nodes(number_of_validators=number_of_validators,    n=validator_num_classes,    amount=test_amount,    seed=use_seed,    distribution=validator_dist)
        x_val_dict, mask_val_dict, y_val_dict = dd.distribute_nlp_data_to_participants(label_dict=label_dict_validators,   amount=test_amount,   number_of_samples=number_of_validators,   n=validator_num_classes,   x_data=x_test,   mask_data=mask_test,   y_data=y_test,   node_label_info=node_label_info_validators,   amount_info_table=amount_info_table_validators,   x_name="x_val",   mask_name="mask_val",   y_name="y_val",   randomization_flag=False)

    train_ds = TensorDataset(x_train, mask_train, y_train)
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)

    test_ds = TensorDataset(x_test, mask_test, y_test)
    test_dl = DataLoader(test_ds, batch_size = batch_size * 2)
    #Bert Model
    main_model    = cm.BertModelForClassification(n).to(device[0])
    print(y_train, np.unique(y_train))
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train.numpy())
    class_weights = dict(zip(np.unique(y_train), class_weights))
    class_wts     = [class_weights[i] for i in range(n)]
    # convert class weights to tensor
    weights = torch.tensor(class_wts, dtype=torch.float).to(device[0])
    main_criterion = nn.NLLLoss()
    main_optimizer = AdamW(main_model.parameters(), lr=5e-5)
    main_model = main_model.to(device[0])  # assign model to the device to be processed by the cloud #TODO seperate devices for the cloud and each edge with additional dictionary

    # define hyper-parameters of the model training
    # main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # main_criterion = nn.CrossEntropyLoss()
    # scheduler = lr_scheduler.ReduceLROnPlateau(main_optimizer, mode="max", factor=lr_scheduler_factor, patience=10, threshold=best_threshold, verbose=True, min_lr=min_lr)
    # the function below generates individual models for each client on the edge seperately
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_nlp(n,   y_train,   number_of_samples,   learning_rate,   momentum,   device,   weight_decay)

    test_accuracy = 0
    test_accuracies_of_each_iteration = np.array([], dtype=float)
    its = np.array([], dtype=int)
    distances = pd.DataFrame()
    thresholds = pd.DataFrame()
    
    mining_history = {
        "loss_hist" : [],
        "acc_hist"  : [],
        "miner_nodes": [],
        "validated_edges": [],
        "blacklisted_edges": []
    }

    test_accuracy = 0

    for iteration in tqdm(range(iteration_num)): # communication rounds
        # send the cloud's current state to the edge users to initiate the federated learning for current communication round
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict_nlp(main_model,  model_dict,  number_of_samples, device)
        # start internal training of the edge users for current communication round
        tn.start_train_end_node_process_nlp(number_of_samples,  x_train_dict,  mask_train_dict,  y_train_dict,  x_test_dict,  mask_test_dict,  y_test_dict,  batch_size,  model_dict,  criterion_dict,  optimizer_dict,  numEpoch,  device,  clipping,  clipping_threshold)

        # # perform validation 
        filtered_model_dict = copy.deepcopy(model_dict)
        if applyarfed:
            iteration_distance = arfu.calculate_euclidean_distances(main_model, model_dict)
            iteration_distance, iteration_threshold = arfu.get_outlier_situation_and_thresholds_for_layers(iteration_distance)
            thresholds = pd.concat([thresholds, iteration_threshold])
            distances  = pd.concat([distances, iteration_distance])
            main_model = arfu.strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, iteration_distance, device[0])

        if apply_mining and initialization_rounds <= iteration:
            if is_noniid:
                if unbias_mechanism: # unbias the models and change the edge models dictionary accordingly
                    for client in range(number_of_samples):
                        model_dict["model" + str(client)] = tn.unbias_noniid_edge_update(main_model, model_dict["model" + str(client)], device)
                else:
                    print("WARNING: There is no unbiasing mechanism used!")
            mining_results_loss, mining_results_acc, miner_names, validated_edges, blacklisted_edges = tn.start_mining_validator_node_nlp(number_of_samples, number_of_validators, x_val_dict, mask_val_dict, y_val_dict, model_dict, criterion_dict, device, batch_size, iteration, validation_threshold=max(test_accuracy/warranty_factor, initial_cut), initialization_rounds=initialization_rounds, config="one_to_all")
        
            mining_history["loss_hist"].append(mining_results_loss)
            mining_history["acc_hist"].append(mining_results_acc)
            mining_history["miner_nodes"].append(miner_names)
            mining_history["validated_edges"].append(validated_edges.tolist())
            mining_history["blacklisted_edges"].append(blacklisted_edges.tolist())
            for adv in blacklisted_edges: #  filter the model_dict to give 0 for poisonous ones
                # TODO should we remove adverserial nodes completely or sparsely
                # currently it is not deleting at all
                _ = filtered_model_dict.pop(adv)    
        #with open("./mining/nlp_init_{}_rounds_{}_adv_ratio_{}_ood.txt".format(initialization_rounds, iteration_num, hostile_node_percentage), "w") as fp:
                #json.dump(mining_history, fp)  # encode dict into JSON

        # aggregation mechanism is introduced by the function below
        if len(filtered_model_dict) and not applyarfed:
            if verbose:
                print("Updating main model weights!")
            main_model = tn.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, filtered_model_dict, device[0])
        # testing the federated model on central server test dataset
        test_loss, test_accuracy = tn.validation_nlp(main_model, test_dl, main_criterion, device[0])
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        new_lr = main_optimizer.param_groups[0]["lr"]
        #optimizer_dict = cm.update_learning_rate_decay(optimizer_dict, new_lr) # adjust lr for all clients according to the cloud test set accuracy, TODO can be performed as intervalidator function as well since there will be no central server testing dataset
        if verbose:                
            print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

    print(test_accuracies_of_each_iteration)
    
    if not os.path.exists("histories"):
        os.makedirs("histories")
    if apply_mining:
        method = "tshield"
    elif applyarfed:
        method = "arfed"
    else:
        method = "vanillafl"
    np.save("./histories/FLNLP_{}_{}_noniidedge_{}_{}_ratio_{}.npy".format(method,apply_mining, is_noniid, validator_dist, hostile_node_percentage), test_accuracies_of_each_iteration)

    with open("./histories/hist_FLNLP_"+method+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_initround_"+str(initialization_rounds)+"_initcut_"+str(initial_cut)+"_warfact_"+str(warranty_factor)+"_unbiasing_"+str(unbias_mechanism)+"_minernum_"+str(number_of_validators)+"_ratio_"+str(hostile_node_percentage)+".txt", "w") as fp:
        json.dump(mining_history, fp)  # encode dict into JSON
    print("Simulation is complete!")
    return test_accuracies_of_each_iteration

def medical_run_sim(opt, device, isdict=False, verbose=False):
    """
    A function to realize simulations for CIFAR10
    opt: the hyperparameters given by user
    device: the torch device for training and inference
    """
    print("device: ", device)

    if isdict:  
        number_of_samples = opt["numusers"]    # number of participants
        number_of_validators = opt["numvals"]  # number of validator miners
        apply_mining = opt["applymining"]      # a boolean standing for validation will be applied
        validator_dist = opt["valdist"]        # the data distribution of the validator nodes
        validator_num_classes = opt["valn"]    # the number of classes validator node has
        is_noniid = opt["isnoniid"]            # boolean stands for edge users having iid or non iid distribution
        is_organized = opt["isorganized"]      # is label flipping attack is selects classes randomly or specified manner
        hostile_node_percentage = opt["advratio"] # malicious participant ratio
        iteration_num = opt["nrounds"]         # number of communication rounds
        learning_rate = opt["lr"]              # learning rate of the client "internal" process
        clipping=opt["clipping"]               # boolean stands for clipping on the weights is going to be applied
        clipping_threshold =opt["clipth"]      # numerical threshold of the clipping applied
        weight_decay = opt["weightdecay"]      # the decay multiplier of the weight
        numEpoch = opt["nepoch"]               # the number of epoch that clients train model on their private data
        batch_size = opt["batchsize"]          # batch size of the client training
        momentum = opt["momentum"]             # the momentum of the client training 
        # some randomization seeds
        seed = opt["seed"]                     
        use_seed = opt["useseed"]              
        hostility_seed = opt["hostilityseed"]  
        converters_seed = opt["convertersseed"]
        train_amount = opt["trainamount"]      # The number of training samples
        test_amount = opt["testamount"]        # The number of test samples
        n = opt["numofclasses"] # TODO 10
        min_n_each_node = opt["numofclasses"]
        attack = opt["attack"]
        applyarfed = opt["applyarfed"]
        initialization_rounds = opt["initround"]
        warranty_factor =opt["warfact"]
    else:
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
        n = opt.numofclasses  # TODO 10
        min_n_each_node = opt.numofclasses
        attack = opt.attack
        applyarfed = opt.applyarfed
        initialization_rounds = opt.initround
        warranty_factor =opt.warfact
    if attack == "collective":
        if isdict:
            converter_dict = opt["converterdict"] #         {0: 1,1: 2,2: 0,}
        else: converter_dict = opt.converterdict
    else:
        converter_dict = {0: np.random.randint(3) ,1: np.random.randint(3),2: np.random.randint(3)}

    if is_noniid:                       # if the edge data is noniid then choose the num class each device will have at most and at least
        n = 3
        min_n_each_node = 1
    else:                               # o.w. it is 10 class per client and the label distribution is uniform accross different edge users
        n = 3
        min_n_each_node = 3

    x_train, y_train, x_test, y_test = dd.load_medical_data(loc="./data/datasubset") # load the central dataset
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
        y_train_dict = dd.convert_nodes_to_hostile(y_train_dict, nodes_list,converter_dict=converter_dict)
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
    #Resnet
    main_model=models.resnet18(pretrained=True)
    num_ftrs = main_model.fc.in_features
    main_model.fc = nn.Linear(num_ftrs, 3)

    main_model.to(device)
    cm.weights_init(main_model)         # assign weights to the model to cloud server
    
    # define hyper-parameters of the model training

    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    main_criterion = nn.CrossEntropyLoss()
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_medical_cnn(number_of_samples, learning_rate, momentum, device, weight_decay)


    test_accuracy = 0
    test_accuracies_of_each_iteration = np.array([], dtype=float)
    its = np.array([], dtype=int)
    distances = pd.DataFrame()
    thresholds = pd.DataFrame()

    mining_history = {
        "loss_hist" : [],
        "acc_hist"  : [],
        "miner_nodes": [],
        "validated_edges": [],
        "blacklisted_edges": []
    }
    
    # warranty_rounds = 5
    # warranty_cut = 0.1
    #warranty_factor = 1.25
    initial_cut = np.round(1/n, 2)
    #initialization_rounds = 5 # 3 for iid case

    for iteration in tqdm(range(iteration_num)): # communication rounds
        #print("Epoch",iteration)
        # send the cloud's current state to the edge users to initiate the federated learning for current communication round
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples)
        # start internal training of the edge users for current communication round
        tn.start_train_end_node_process_cifar(number_of_samples, x_train_dict, y_train_dict, x_test_dict,y_test_dict, batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, device, clipping, clipping_threshold)

        if applyarfed:         
            iteration_distance = arfu.calculate_euclidean_distances(main_model, model_dict)
            iteration_distance, iteration_threshold = arfu.get_outlier_situation_and_thresholds_for_layers(iteration_distance)
            thresholds = pd.concat([thresholds, iteration_threshold])
            distances = pd.concat([distances, iteration_distance])
            main_model = arfu.strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(
            main_model, model_dict, iteration_distance, device)
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
        if len(filtered_model_dict) and not applyarfed:
            main_model = tn.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, filtered_model_dict, device)
        # testing the federated model on central server test dataset
        test_loss, test_accuracy = tn.validation_medical(main_model, test_dl, main_criterion, device,set_evaluation_flag=False)
        #scheduler.step(test_accuracy)
        new_lr = main_optimizer.param_groups[0]["lr"]
        optimizer_dict = cm.update_learning_rate_decay(optimizer_dict, new_lr) # adjust lr for all clients according to the cloud test set accuracy, TODO can be performed as intervalidator function as well since there will be no central server testing dataset
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        if verbose:
            print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))
    print({str(hostile_node_percentage):test_accuracies_of_each_iteration})
    if not os.path.exists("histories"):
        os.makedirs("histories")
    if apply_mining:
        method = "tshield"
    elif applyarfed:
        method = "arfed"
    else:
        method = "vanillafl"
    with open("./histories/hist_flmedical_"+method+"_"+str(numEpoch)+"_lr_"+str(learning_rate)+"_isnoniid_"+str(is_noniid)+"_numclients_"+str(number_of_samples)+"_ismining_"+str(apply_mining)+"_valdist_"+validator_dist+"_ratio_"+str(hostile_node_percentage)+".txt", "w") as fp:
        json.dump(mining_history, fp)  # encode dict into JSON
    print("Simulation is complete!")
    #return test_accuracies_of_each_iteration[-1]
    np.save("./histories/FLMEDICAL_"+method+"_noniidedge_"+str(is_noniid)+"_"+validator_dist+"_ratio_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)


