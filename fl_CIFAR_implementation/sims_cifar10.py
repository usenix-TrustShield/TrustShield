# import the necessary packages
import numpy as np
import torch
import json
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
from fl_utils import distribute_data as dd
from fl_utils import train_nodes as tn
from fl_utils import construct_models as cm
from tqdm import tqdm

def run_sim(opt, device):
    """
    A function to realize simulations for CIFAR10
    opt: the hyperparameters given by user
    device: the torch device for training and inference
    """
    print("device: ", device)

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

    if is_noniid:                       # if the edge data is noniid then choose the num class each device will have at most and at least
        n = 5
        min_n_each_node = 5
    else:                               # o.w. it is 10 class per client and the label distribution is uniform accross different edge users
        n = 10
        min_n_each_node = 10

    x_train, y_train, x_test, y_test = dd.load_cifar_data() # load the central dataset

    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount) # assign new position to each label
    # the below function gets the mappings for each label for each classes, see the function comment
    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    # the below function generates dataset for each client by considering the distributions provided as the outputs of the function above
    x_train_dict, y_train_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_train,amount=train_amount,number_of_samples=number_of_samples,n=n, x_data=x_train,y_data=y_train,node_label_info=node_label_info_train,amount_info_table=amount_info_table_train,x_name="x_train",y_name="y_train")
    # selects some of the nodes to make them adversary
    nodes_list = dd.choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed)
    print("hostile_nodes:", nodes_list)
    # following code provides label flipping attack in either ordered way or completely random
    if is_organized:
        y_train_dict = dd.convert_nodes_to_hostile(y_train_dict, nodes_list,
                                                converter_dict={0: 2,
                                                                1: 9,
                                                                2: 0,
                                                                3: 5,
                                                                4: 7,
                                                                5: 3,
                                                                6: 8,
                                                                7: 4,
                                                                8: 6,
                                                                9: 1})
    else:
        y_train_dict = dd.convert_nodes_to_hostile_with_different_converters(y_train_dict, nodes_list,
                                                                            converters_seed=converters_seed)

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

    test_accuracies_of_each_iteration = np.array([], dtype=float)

    mining_history = {
        "loss_hist" : [],
        "acc_hist"  : [],
        "miner_nodes": [],
        "validated_edges": [],
        "blacklisted_edges": []
    }
    


    test_accuracy = 0
    
    # test_accuracy_array = [np.round(2/n, 2)+warranty_cut for i in range(warranty_rounds)]

    for iteration in tqdm(range(iteration_num)): # communication rounds
        # send the cloud's current state to the edge users to initiate the federated learning for current communication round
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples)
        # start internal training of the edge users for current communication round
        tn.start_train_end_node_process_cifar(number_of_samples, x_train_dict, y_train_dict, x_test_dict,y_test_dict, batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, device, clipping, clipping_threshold)
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
                # TODO should we remove adverserial nodes completely or sparsely
                # currently it is not deleting at all
                _ = filtered_model_dict.pop(adv)    

        # aggregation mechanism is introduced by the function below
        if len(filtered_model_dict):
                main_model = tn.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, filtered_model_dict, device)
        # testing the federated model on central server test dataset
        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)
        #scheduler.step(test_accuracy)
        # test_accuracy_array.append(test_accuracy) if iteration>initialization_rounds else None
        new_lr = main_optimizer.param_groups[0]["lr"]
        optimizer_dict = cm.update_learning_rate_decay(optimizer_dict, new_lr) # adjust lr for all clients according to the cloud test set accuracy, TODO can be performed as intervalidator function as well since there will be no central server testing dataset
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

    #print(test_accuracies_of_each_iteration)
    np.save("./histories/FLCIFAR10_mining_"+str(apply_mining)+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_ratio_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)
    with open("./histories/hist_FLCIFAR10_mining_"+str(apply_mining)+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_initround_"+str(initialization_rounds)+"_initcut_"+str(initial_cut)+"_warfact_"+str(warranty_factor)+"_internalepoch_"+str(numEpoch)+"_unbiasing_"+str(unbias_mechanism)+"_minernum_"+str(number_of_validators)+"_organized_numclients_100_ratio_"+str(hostile_node_percentage)+".txt", "w") as fp:
        json.dump(mining_history, fp)  # encode dict into JSON
    print("Simulation is complete!")


if __name__ == "__main__":
    # Adjust parameters for the simulation

    parser = argparse.ArgumentParser()
    parser.add_argument("--deviceno", type=int, default=0)
    parser.add_argument("--nrounds", type=int,default=500)
    
    # edge user parameters
    parser.add_argument("--numusers",type=int,default=100)
    parser.add_argument("--isnoniid", type=int,default=0)

    # adverserial parameters
    parser.add_argument("--advratio", type=float, default=0.5)
    parser.add_argument("--isorganized", type=int, default=1)
    
    # miner parameters
    parser.add_argument("--initround", type=int, default=20)
    parser.add_argument("--unbiasmech", type=int, default=0)
    parser.add_argument("--applymining", type=int, default=1)
    parser.add_argument("--numvals", type=int, default=4)
    parser.add_argument("--valdist", type=str, default="uniform")
    parser.add_argument("--valn", type=int, default=10)
    parser.add_argument('--warfact', type=float, default=2.0)
    parser.add_argument('--initcut', type=float, default=0.2)
    # validation_threshold=0.2, initialization_rounds = 20, config="one_to_all"

    # learning parameters
    parser.add_argument("--trainamount", type=int,default= 5000)
    parser.add_argument("--testamount", type=int,default=1000)
    parser.add_argument("--nepoch", type=int, default=10)
    parser.add_argument("--batchsize", type=int,default=256)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--minlr", type=float, default=0.000010)
    parser.add_argument("--lrschfac", type=float, default=0.2)
    

    parser.add_argument("--bestth", type=float, default=0.0001)
    parser.add_argument("--clipping", type=int, default=1)
    parser.add_argument("--clipth",type=int, default=10)
    parser.add_argument("--weightdecay", type=float, default=0.0001)    
    parser.add_argument("--momentum",type=float, default= 0.9)

    # randomization parameters
    parser.add_argument("--seed", type=int,default=11)
    parser.add_argument("--useseed", type=int,default=8)
    parser.add_argument("--hostilityseed", type=int,default=90)
    parser.add_argument("--convertersseed", type=int,default= 51)


    opt = parser.parse_args()

    device = torch.device("cuda:"+str(opt.deviceno) if (torch.cuda.is_available()) else "cpu")
    
    run_sim(opt,device)