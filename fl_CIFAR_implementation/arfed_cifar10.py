import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler
from fl_utils import distribute_data as dd
from fl_utils import train_nodes as tn
from fl_utils import construct_models as cm
from fl_utils import ARFED_utils as arfu
import argparse
from tqdm import tqdm

def run_sim(opt, device):
    """
    A function to realize simulations for CIFAR10
    opt: the hyperparameters given by user
    device: the torch device for training and inference
    """
    print("device: ", device)

    number_of_samples = opt.numusers #number of participants
    is_noniid = opt.isnoniid
    is_organized = opt.isorganized 
    hostile_node_percentage =opt.advratio #malicious participant ratio
    iteration_num = opt.nrounds  #number of communication rounds
    learning_rate = opt.lr 
    min_lr = opt.minlr  
    lr_scheduler_factor =opt.lrschfac
    best_threshold = opt.bestth  
    clipping=opt.clipping
    clipping_threshold =opt.clipth
    weight_decay = opt.weightdecay 
    numEpoch = opt.nepoch 
    batch_size = opt.batchsize 
    momentum =opt.momentum 
    seed = opt.seed                     
    use_seed = opt.useseed              
    hostility_seed = opt.hostilityseed  
    converters_seed = opt.convertersseed
    train_amount = opt.trainamount      # The number of training samples
    test_amount = opt.testamount        # The number of test samples

    if is_noniid:
        n = 5
        min_n_each_node = 5
    else:
        n = 10
        min_n_each_node = 10



    x_train, y_train, x_test, y_test = dd.load_cifar_data()



    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount)
    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)

    x_train_dict, y_train_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_train,
                                                                        amount=train_amount,
                                                                        number_of_samples=number_of_samples,
                                                                        n=n, x_data=x_train,
                                                                        y_data=y_train,
                                                                        node_label_info=node_label_info_train,
                                                                        amount_info_table=amount_info_table_train,
                                                                        x_name="x_train",
                                                                        y_name="y_train")

    nodes_list = dd.choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed)
    print("hostile_nodes:", nodes_list)

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

    ## test
    label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
    node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples,
        n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    x_test_dict, y_test_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_test,
                                                                        amount=test_amount,
                                                                        number_of_samples=number_of_samples,
                                                                        n=n, x_data=x_test,
                                                                        y_data=y_test,
                                                                        node_label_info=node_label_info_test,
                                                                        amount_info_table=amount_info_table_test,
                                                                        x_name="x_test",
                                                                        y_name="y_test")

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    main_model = cm.Cifar10CNN()
    cm.weights_init(main_model)
    main_model = main_model.to(device)

    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    main_criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(main_optimizer, mode="max", factor=lr_scheduler_factor,
                                            patience=10, threshold=best_threshold, verbose=True, min_lr=min_lr)

    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_cifar_cnn(number_of_samples, learning_rate,
                                                                                                            momentum, device, weight_decay)


    test_accuracies_of_each_iteration = np.array([], dtype=float)
    its = np.array([], dtype=int)
    distances = pd.DataFrame()
    thresholds = pd.DataFrame()

    for iteration in tqdm(range(iteration_num)):
        its = np.concatenate((its, np.ones(number_of_samples, dtype=int) * (iteration + 1)))

        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict,
                                                                    number_of_samples)

        tn.start_train_end_node_process_cifar(number_of_samples, x_train_dict, y_train_dict, x_test_dict,
                                            y_test_dict, batch_size, model_dict, criterion_dict,
                                            optimizer_dict, numEpoch, device, clipping, clipping_threshold)

        iteration_distance = arfu.calculate_euclidean_distances(main_model, model_dict)
        iteration_distance, iteration_threshold = arfu.get_outlier_situation_and_thresholds_for_layers(
            iteration_distance)

        thresholds = pd.concat([thresholds, iteration_threshold])
        distances = pd.concat([distances, iteration_distance])

        main_model = arfu.strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(
            main_model, model_dict, iteration_distance, device)
        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)
        scheduler.step(test_accuracy)
        new_lr = main_optimizer.param_groups[0]["lr"]
        optimizer_dict = cm.update_learning_rate_decay(optimizer_dict, new_lr)

        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))
    # TODO comment out the line below after getting tsne results   
    np.save("./histories/FLCIFAR10_ARFED_isnoniid_"+str(is_noniid)+"_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)


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