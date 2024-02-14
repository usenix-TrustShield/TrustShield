# import the necessary packages
import numpy as np
import copy
from sklearn.utils.class_weight import compute_class_weight
import torch
import json
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse

from adv_utils.distribute_data import distribute_data_to_participants

from adv_utils import distribute_data as dd
from adv_utils import train_nodes as tn
from adv_utils import construct_models as cm
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers   import AdamW
import pandas as pd
from adv_utils import ARFED_utils as arfu

def plot_multiple(y_maps, filename):
    """
    Takes in a map of all the y-axis values & plots them against
    communication rounds
    """
    for item, value in y_maps.items():
        #print(value)
        plt.plot(range(1, len(value) + 1), value, '.-', label=item)
        #plt.plot(x,value,'.-', label=item)
    plt.legend()
    #plt.xlabel('%Adversarial clients')
    plt.xlabel('Communication rounds')
    plt.ylabel('Accuracy')
    plt.title("Training Accuracy vs Communication rounds")
    plt.savefig(filename)

def run_sim(opt, device):
    """
    A function to realize simulations for any classification task
    opt: the hyperparameters given by user
    device: the torch device for training and inference
    """
    print("device: ", device)

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
    min_lr                  = opt.minlr             # minimum learning rate obtained after decay of the learning rate 
    lr_scheduler_factor     = opt.lrschfac          # the multiplier of the learning rate decay
    best_threshold          = opt.bestth         
    clipping                = opt.clipping          # boolean stands for clipping on the weights is going to be applied
    clipping_threshold      = opt.clipth            # numerical threshold of the clipping applied
    initialization_rounds   = opt.initround
    weight_decay            = opt.weightdecay       # the decay multiplier of the weight
    numEpoch                = opt.nepoch            # the number of epoch that clients train model on their private data
    batch_size              = opt.batchsize         # batch size of the client training
    momentum                = opt.momentum          # the momentum of the client training 
    ood_ratio               = opt.ood_ratio         # percentage of out of distribution data
    train_number_classes    = opt.train_num_classes # number of classes in the training set
    test_num_classes        = opt.test_num_classes  # number of classes of the cloud test set

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
    # Set number of classes
    n = 4
    if is_noniid:                       # if the edge data is noniid then choose the num class each device will have at most and at least
        min_n_each_node = 2
    else:                               # o.w. it is 10 class per client and the label distribution is uniform accross different edge use
        min_n_each_node = 4

    file = './ecommerceDataset.csv'

    x_train, mask_train, y_train, x_val, mask_val, y_val, x_test, mask_test, y_test = dd.load_data(file, new=True) # load the central dataset
    print("step 1")
    # print("xTRAIN",len(x_train))
    # print("YTrain",len(y_train))
    # print("xTEST",len(x_test))
    # print("Ytest",len(y_test))
    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train,
                                                   seed=seed,
                                                   amount=train_amount,
                                                   n=n) # assign new position to each label
    # the below function gets the mappings for each label for each classes, see the function comment
    node_label_info_train, total_label_occurences_train, amount_info_table_train \
        = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples,
                                                                         n=n,
                                                                         amount=train_amount,
                                                                         seed=use_seed,
                                                                         min_n_each_node=min_n_each_node)
    print("step 2")
    # the below function generates dataset for each client by considering the distributions provided as the outputs of the function above
    # print("node_label_info_train", node_label_info_train)
    # print("amount_info_table_train", amount_info_table_train)
    # print("THIS IS N : ", n)
    x_train_dict, mask_train_dict, y_train_dict = \
        distribute_data_to_participants(label_dict=label_dict_train,
                                        amount=train_amount,
                                        number_of_samples=number_of_samples,
                                        n=train_number_classes,
                                        x_data=x_train,
                                        mask_data=mask_train,
                                        y_data=y_train,
                                        node_label_info=node_label_info_train,
                                        amount_info_table=amount_info_table_train,
                                        x_name="x_train",
                                        mask_name="mask_train",
                                        y_name="y_train",
                                        randomization_flag=True)
    # selects some of the nodes to make them adversary
    print("step 3")
    nodes_list = dd.choose_nodes_randomly_to_convert_hostile(hostile_node_percentage,
                                                             number_of_samples,
                                                             hostility_seed)
    print("Hostile_nodes:", nodes_list)
    # following code provides label flipping attack in either ordered way or completely random
    if is_organized:
        y_train_dict = dd.convert_nodes_to_hostile(y_train_dict,
                                                   nodes_list,
                                                   converter_dict={0: 1,
                                                                   1: 0,
                                                                   2: 3,
                                                                   3: 2})
    else:
        y_train_dict = dd.convert_nodes_to_hostile_with_different_converters(y_train_dict,
                                                                             nodes_list,
                                                                             converters_seed=converters_seed)
    # print("Y train dict", y_train_dict)
    ## apply the same functions applied to the training dataset to form validaton set
    label_dict_test = \
        dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount,n=test_num_classes)
    print("step 4")
    # print("label dict",label_dict_test)
    
    node_label_info_test, total_label_occurences_test, amount_info_table_test = \
        dd.get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples=number_of_samples,
                                                                       n=test_num_classes,
                                                                       amount=test_amount,
                                                                       seed=use_seed,
                                                                       min_n_each_node=test_num_classes)
    
    # print("node label info", node_label_info_test)
    # print("amount info table", amount_info_table_test)
    
    x_test_dict, mask_test_dict, y_test_dict = \
        dd.distribute_data_to_participants(label_dict=label_dict_test,
                                           amount=test_amount,
                                           number_of_samples=number_of_samples,
                                           n=test_num_classes,
                                           x_data=x_test,
                                           mask_data=mask_test,
                                           y_data=y_test,
                                           node_label_info=node_label_info_test,
                                           amount_info_table=amount_info_table_test,
                                           x_name="x_test",
                                           mask_name="mask_test",
                                           y_name="y_test",
                                           randomization_flag=False)
    print("step 5")


    # generate the datasets and loaders
    train_ds = TensorDataset(x_train, mask_train, y_train)
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)

    test_ds = TensorDataset(x_test, mask_test, y_test)
    test_dl = DataLoader(test_ds, batch_size = batch_size * 2)
    print("step 6")
    #Bert Model
    #main_model = nn.DataParallel(cm.BertModelForClassification(n), [0, 1, 2])
    main_model    = cm.BertModelForClassification(n).to(device[0])
    print(y_train, np.unique(y_train))
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=np.unique(y_train),
                                         y=y_train.numpy())
    class_weights = dict(zip(np.unique(y_train), class_weights))
    class_wts     = [class_weights[i] for i in range(n)]
    # convert class weights to tensor
    weights = torch.tensor(class_wts, dtype=torch.float).to(device[0])
    main_criterion = nn.NLLLoss()
    main_optimizer = AdamW(main_model.parameters(), lr=5e-5)
    main_model = main_model.to(device[0])  # assign model to the device to be processed by the cloud #TODO seperate devices for the cloud and each edge with additional dictionary
    print("step 7")
    # define hyper-parameters of the model training
    # main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # main_criterion = nn.CrossEntropyLoss()
    # scheduler = lr_scheduler.ReduceLROnPlateau(main_optimizer, mode="max", factor=lr_scheduler_factor, patience=10, threshold=best_threshold, verbose=True, min_lr=min_lr)
    # the function below generates individual models for each client on the edge seperately
    model_dict, optimizer_dict, criterion_dict = \
        tn.create_model_optimizer_criterion_dict_for_cifar_cnn(n,
                                                               y_train,
                                                               number_of_samples,
                                                               learning_rate,
                                                               momentum,
                                                               device,
                                                               weight_decay)

    test_accuracies_of_each_iteration = np.array([], dtype=float)

    test_accuracy = 0
    
    # test_accuracy_array = [np.round(2/n, 2)+warranty_cut for i in range(warranty_rounds)]
    its        = np.array([], dtype=int)
    distances  = pd.DataFrame()
    thresholds = pd.DataFrame()

    for iteration in tqdm(range(iteration_num)): # communication rounds
        # send the cloud's current state to the edge users to initiate the federated learning for current communication round
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model,
                                                              model_dict,
                                                              number_of_samples, device)
        print("step 8")
        # start internal training of the edge users for current communication round
        tn.start_train_end_node_process_cifar(number_of_samples,
                                              x_train_dict,
                                              mask_train_dict,
                                              y_train_dict,
                                              x_test_dict,
                                              mask_test_dict,
                                              y_test_dict,
                                              batch_size,
                                              model_dict,
                                              criterion_dict,
                                              optimizer_dict,
                                              numEpoch,
                                              device,
                                              clipping,
                                              clipping_threshold)
        iteration_distance = arfu.calculate_euclidean_distances(main_model, model_dict)
        iteration_distance, iteration_threshold = arfu.get_outlier_situation_and_thresholds_for_layers(iteration_distance)

        thresholds = pd.concat([thresholds, iteration_threshold])
        distances  = pd.concat([distances, iteration_distance])
        main_model = arfu.strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, iteration_distance, device[0])
        # testing the federated model on central server test dataset
        # test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)
        # scheduler.step(test_accuracy)
        # test_accuracy_array.append(test_accuracy) if iteration>initialization_rounds else None
        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device[0])
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        new_lr = main_optimizer.param_groups[0]["lr"]
        #optimizer_dict = cm.update_learning_rate_decay(optimizer_dict, new_lr) # adjust lr for all clients according to the cloud test set accuracy, TODO can be performed as intervalidator function as well since there will be no central server testing dataset
                
        print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

    # save_models(model_dict, initialization_rounds, iteration_num, hostile_node_percentage)
    #save_models({"main_model" : main_model}, initialization_rounds, iteration_num, hostile_node_percentage)
    print(test_accuracies_of_each_iteration)
    # np.save("./histories/textcls_mining_"+str(apply_mining)+"_noniidedge_"+str(is_noniid)+"_"+str(validator_dist)+"_initround_"+str(initialization_rounds)+"_initcut_"+str(initial_cut)+"_warfact_"+str(warranty_factor)+"_internalepoch_"+str(numEpoch)+"_unbiasing_"+str(unbias_mechanism)+"_minernum_"+str(number_of_validators)+"_organized_numclients_100_ratio_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)
    np.save("./histories/NLP/FLNLP_ARFED_isnoniid_{}_{}.npy".format(initialization_rounds, iteration_num, is_noniid, apply_mining, validator_dist, initial_cut,hostile_node_percentage), test_accuracies_of_each_iteration)
    #plot_multiple({str(hostile_node_percentage) : test_accuracies_of_each_iteration},"./plots/nlp_rounds_{}_adv_ratio_{}.png".format(iteration_num, hostile_node_percentage))
    # plot_multiple({str(hostile_node_percentage) : test_accuracies_of_each_iteration},
                #   "./plots/arfed_final_plots_validation_init_{}_ood.png".format(initialization_rounds))

    print("Simulation is complete!")
    return test_accuracies_of_each_iteration

def save_models(model_dict, init, rounds, adv_ratio):
    """
    Saves the model within model_dict appropriately
    """
    for name, model in model_dict.items():
        torch.save(model.state_dict(), "./models/new_nlp_{}_init_{}_rounds_{}_adv_ratio_{}.pt".format(name, init, rounds, adv_ratio))

if __name__ == "__main__":
    # Adjust parameters for the simulation

    parser = argparse.ArgumentParser()
    parser.add_argument("--deviceno", type=int, default=1)
    parser.add_argument("--nrounds", type=int,default=20)
    
    # edge user parameters
    parser.add_argument("--numusers",type=int,default=20)
    parser.add_argument("--isnoniid", type=int,default=1)

    # adverserial parameters
    parser.add_argument("--advratio", type=float, default=0.0)
    parser.add_argument("--isorganized", type=int, default=1)
    
    # miner parameters
    parser.add_argument("--initround", type=int, default=5)
    parser.add_argument("--unbiasmech", type=int, default=0)
    parser.add_argument("--applymining", type=int, default=0)
    parser.add_argument("--numvals", type=int, default=4)
    parser.add_argument("--valdist", type=str, default="uniform")
    parser.add_argument("--valn", type=int, default=4)
    parser.add_argument('--warfact', type=float, default=1.2)
    parser.add_argument('--initcut', type=float, default=0.3)
    # validation_threshold=0.2, initialization_rounds = 20, config="one_to_all"

    # learning parameters
    parser.add_argument("--trainamount", type=int,default=200)
    parser.add_argument("--testamount", type=int,default=100)
    parser.add_argument("--nepoch", type=int, default=1)
    parser.add_argument("--batchsize", type=int,default=8)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--minlr", type=float, default=0.00005)
    parser.add_argument("--lrschfac", type=float, default=0.2)
    

    parser.add_argument("--bestth", type=float, default=0.0001)
    parser.add_argument("--clipping", type=int, default=0.1)
    parser.add_argument("--clipth",type=int, default=10)
    parser.add_argument("--weightdecay", type=float, default=0.0001)    
    parser.add_argument("--momentum",type=float, default= 0.9)

    # randomization parameters
    parser.add_argument("--seed", type=int,default=11)
    parser.add_argument("--useseed", type=int,default=8)
    parser.add_argument("--hostilityseed", type=int,default=90)
    parser.add_argument("--convertersseed", type=int,default= 51)

    #ood parameters
    parser.add_argument("--ood_ratio", type=int,default=0.5)
    parser.add_argument("--test_num_classes",type=int, default=4)
    parser.add_argument("--train_num_classes",type=int, default=4)



    opt = parser.parse_args()
    adv = [0.45, 0.65]
    device1 = torch.device("cuda:"+str(0) if (torch.cuda.is_available()) else "cpu")
    device2 = torch.device("cuda:"+str(1) if (torch.cuda.is_available()) else "cpu")
    device3 = torch.device("cuda:"+str(2) if (torch.cuda.is_available()) else "cpu")
    device4 = torch.device("cuda:"+str(3) if (torch.cuda.is_available()) else "cpu")
    final_test_accuracies = []
    #run_sim(opt, [device1, device2, device3, device4])
    for elem in adv:
         opt.advratio = elem
         final_test_accuracies.append(run_sim(opt, [device1, device2, device3, device4]))
    
    print(final_test_accuracies)
    #np.save("adv_ratio_0.05_run.npy", final_test_accuracies)
    #run_sim(opt,device)