import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from fl_utils import distribute_data as dd
from fl_utils import train_nodes as tn
from fl_utils import construct_models as cm
from fl_utils import ARFED_utils as arfu
import argparse

def run_sim(opt, device):
    print("device: ", device)

    number_of_samples = opt.numusers # number of participantss
    is_noniid = opt.isnoniid
    is_cnn = opt.iscnn
    is_organized = opt.isorganized
    hostile_node_percentage = opt.advratio # malicious participant ratio

    iteration_num = opt.nrounds  ## number of communication rounds

    learning_rate = opt.lr
    numEpoch = opt.nepoch
    batch_size = opt.batchsize
    momentum = opt.momentum

    seed = opt.seed
    use_seed = opt.useseed
    hostility_seed = opt.hostilityseed
    converters_seed = opt.convertersseed

    train_amount = opt.trainamount
    test_amount = opt.testamount

    if is_noniid:
        n = 2
        min_n_each_node = 2
    else:
        n = 10
        min_n_each_node = 10


    x_train, y_train, x_valid, y_valid, x_test, y_test = dd.load_mnist_data()
    x_test, y_test = dd.get_equal_size_test_data_from_each_label(x_test, y_test, min_amount=test_amount)

    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor,
                                                            (x_train, y_train, x_valid, y_valid, x_test, y_test))

    ##train
    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount)
    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)

    x_train_dict, y_train_dict = dd.distribute_mnist_data_to_participants(label_dict=label_dict_train,
                                                                                        amount=train_amount,
                                                                                        number_of_samples=number_of_samples,
                                                                                        n=n, x_data=x_train,
                                                                                        y_data=y_train,
                                                                                        node_label_info=node_label_info_train,
                                                                                        amount_info_table=amount_info_table_train,
                                                                                        x_name="x_train",
                                                                                        y_name="y_train",
                                                                                        is_cnn=is_cnn)


    nodes_list = dd.choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed)
    print("hostile_nodes:", nodes_list)


    if is_organized:
        y_train_dict = dd.convert_nodes_to_hostile(y_train_dict, nodes_list,
                                                converter_dict={0: 9, 1: 7, 2: 5, 3: 8, 4: 6, 5: 2, 6: 4, 7: 1, 8: 3,
                                                                9: 0})
    else:
        y_train_dict = dd.convert_nodes_to_hostile_with_different_converters(y_train_dict, nodes_list,
                                                                            converters_seed=converters_seed)


    ## test
    label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
    node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples,
        n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    x_test_dict, y_test_dict = dd.distribute_mnist_data_to_participants(label_dict=label_dict_test,
                                                                                    amount=test_amount,
                                                                                    number_of_samples=number_of_samples,
                                                                                    n=n, x_data=x_test,
                                                                                    y_data=y_test,
                                                                                    node_label_info=node_label_info_test,
                                                                                    amount_info_table=amount_info_table_test,
                                                                                    x_name="x_test",
                                                                                    y_name="y_test", is_cnn=is_cnn)

    if is_cnn:
        reshape_size = int(np.sqrt(x_train.shape[1]))
        x_train = x_train.view(-1, 1, reshape_size, reshape_size)
        x_valid = x_valid.view(-1, 1, reshape_size, reshape_size)
        x_test = x_test.view(-1, 1, reshape_size, reshape_size)

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    if is_cnn:
        main_model = cm.Netcnn()
    else:
        main_model = cm.Net2nn()

    cm.weights_init(main_model)
    main_model = main_model.to(device)

    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=0.9)
    main_criterion = nn.CrossEntropyLoss()
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_mnist(number_of_samples, learning_rate,
                                                                                                    momentum, device, is_cnn)


    test_accuracies_of_each_iteration = np.array([], dtype=float)
    its = np.array([], dtype=int)
    distances = pd.DataFrame()
    thresholds = pd.DataFrame()


    for iteration in range(iteration_num):
        its = np.concatenate((its, np.ones(number_of_samples, dtype=int) * (iteration + 1)))

        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict,
                                                                    number_of_samples)

        tn.start_train_end_node_process_without_print(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                                    batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, device)

        iteration_distance = arfu.calculate_euclidean_distances(main_model, model_dict)
        iteration_distance, iteration_threshold = arfu.get_outlier_situation_and_thresholds_for_layers(
            iteration_distance)

        thresholds = pd.concat([thresholds, iteration_threshold])
        distances = pd.concat([distances, iteration_distance])


        main_model = arfu.strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(main_model,
                                                                        model_dict, iteration_distance, device)

        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))


    distances["iteration"] = its
    thresholds["iteration"] = np.arange(1, iteration_num + 1)
    np.save("./histories/MNIST/FLMNIST_ARFED_isnoniid_"+str(is_noniid)+"_"+str(hostile_node_percentage)+".npy", test_accuracies_of_each_iteration)

if __name__ == "__main__":
    # Adjust parameters for the simulation

    parser = argparse.ArgumentParser()
    parser.add_argument("--deviceno", type=int, default=0)
    parser.add_argument("--nrounds", type=int,default=20)
    
    # edge user parameters
    parser.add_argument("--numusers",type=int,default=100)
    parser.add_argument("--isnoniid", type=int,default=1)

    # adverserial parameters
    parser.add_argument("--advratio", type=float, default=0.0)
    parser.add_argument("--isorganized", type=int, default=1)
    
    # miner parameters # TODO add these parameters afterwards
    parser.add_argument("--initround", type=int, default=5)
    parser.add_argument("--unbiasmech", type=int, default=0)
    parser.add_argument("--applymining", type=int, default=0)
    parser.add_argument("--numvals", type=int, default=4)
    parser.add_argument("--valdist", type=str, default="gaussian")
    parser.add_argument("--valn", type=int, default=10)
    parser.add_argument('--warfact', type=float, default=1.00)
    parser.add_argument('--initcut', type=float, default=0.1)
    # validation_threshold=0.2, initialization_rounds = 20, config="one_to_all"

    # learning parameters
    parser.add_argument("--trainamount", type=int,default= 4000)
    parser.add_argument("--testamount", type=int,default=890)
    parser.add_argument("--nepoch", type=int, default=50)
    parser.add_argument("--batchsize", type=int,default=256)
    parser.add_argument("--lr", type=float, default=0.001)  
    parser.add_argument("--momentum",type=float, default= 0.9)
    parser.add_argument("--iscnn", type=int, default=0) # TODO investigate the effect
    # randomization parameters
    parser.add_argument("--seed", type=int,default=1)
    parser.add_argument("--useseed", type=int,default=23)
    parser.add_argument("--hostilityseed", type=int,default=77)
    parser.add_argument("--convertersseed", type=int,default= 51)


    opt = parser.parse_args()

    device = torch.device("cuda:"+str(opt.deviceno) if (torch.cuda.is_available()) else "cpu")
    
    run_sim(opt,device)
