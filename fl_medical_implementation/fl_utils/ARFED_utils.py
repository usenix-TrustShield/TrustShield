import torch
import numpy as np
import pandas as pd

def calculate_euclidean_distances(main_model, model_dict):
    calculated_parameter_names = []

    for parameters in main_model.named_parameters():
        if "bias" not in parameters[0]:
            calculated_parameter_names.append(parameters[0])

    columns = ["model"] + calculated_parameter_names
    distances = pd.DataFrame(columns=columns)
    model_names = list(model_dict.keys())

    main_model_weight_dict = {}
    for parameter in main_model.named_parameters():
        name = parameter[0]
        weight_info = parameter[1]
        main_model_weight_dict.update({name: weight_info})

    with torch.no_grad():
        for i in range(len(model_names)):
            distances.loc[i, "model"] = model_names[i]
            sample_node_parameter_list = list(model_dict[model_names[i]].named_parameters())
            for j in sample_node_parameter_list:
                if j[0] in calculated_parameter_names:
                    distances.loc[i, j[0]] = round(
                        np.linalg.norm(main_model_weight_dict[j[0]].cpu().data - j[1].cpu().data), 4)

    return distances

def get_outlier_situation_and_thresholds_for_layers(distances, factor=1.5):
    layers = list(distances.columns)
    layers.remove("model")
    threshold_columns = []
    for layer in layers:
        threshold_columns.append((layer + "_lower"))
        threshold_columns.append((layer + "_upper"))
    thresholds = pd.DataFrame(columns=threshold_columns)

    include_calculation_result = True
    for layer in layers:
        data = distances[layer]
        lower, upper = calculate_lower_and_upper_limit(data, factor)
        lower_name = layer + "_lower"
        upper_name = layer + "_upper"
        thresholds.loc[0, lower_name] = lower
        thresholds.loc[0, upper_name] = upper
        name = layer + "_is_in_ci"

        distances[name] = (distances[layer] > lower) & (distances[layer] < upper)
        print(distances[layer],lower,upper)
        include_calculation_result = include_calculation_result & distances[name]

    distances["include_calculation"] = include_calculation_result
    return distances, thresholds

def get_averaged_weights_without_outliers_strict_condition(model_dict, iteration_distance, device):
    chosen_clients = iteration_distance[iteration_distance["include_calculation"] == True].index
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    print("CHOSEN CLIENTS", chosen_clients)
    print("NAME OF MODELS", name_of_models)
    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(chosen_clients))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})

    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(chosen_clients)):
            sample_param_data_list = list(model_dict[name_of_models[chosen_clients[i]]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        mean_weight_array = []
        for m in range(len(weight_names_list)):
            mean_weight_array.append(torch.mean(weight_dict[weight_names_list[m]], 0))

    return mean_weight_array


def strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(main_model,
                                                                                                       model_dict,
                                                                                                       iteration_distance,
                                                                                                       device):
    mean_weight_array = get_averaged_weights_without_outliers_strict_condition(model_dict, iteration_distance, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = mean_weight_array[j]
    return main_model


def calculate_lower_and_upper_limit(data, factor):
    quantiles = data.quantile(q=[0.25, 0.50, 0.75]).values
    q1 = quantiles[0]
    q2 = quantiles[1]
    q3 = quantiles[2]
    iqr = q3 - q1
    print("QUARTILES",q1,q2,q3)
    lower_limit = q1 - factor * iqr
    upper_limit = q3 + factor * iqr
    return lower_limit, upper_limit