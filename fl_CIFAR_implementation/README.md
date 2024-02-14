# INFOCOM-2024 TrustShield
This is a repository for evaluation and mitigation of Federated Learning (FL) for Label Flipping (LF) attacks conducted by a variety of adversarial user ratio in the system. We address the LF attacks with TrustShield algorithm that is explained in the following paper.  

Link to paper: Will be inserted.

## Table of Contents
- [TLDR](#TLDR)
- [Results](#results)
- [Installation](#installation)
  - [Packages](#packages)
  - [Datasets](#dataset)
- [File Structure](#file-structure)
- [Usage](#usage)

## TLDR
We open-source a toolbox to characterize and test FL systems for LF attacks and propose a solution that significantly mitigates such attacks.

## Abstract
<details>
<summary>Click to expand</summary>
Federated Learning (FL) is widely embraced by today's distributed learning applications due to its privacy-preserving nature. However, as we have analyzed in this paper, privacy comes at the cost of security. FL is highly susceptible to label-flipping attacks due to the cloud's lack of access to private data on edge nodes, such as mobile phones. Hence, we have open-sourced TrustShield to enhance the security of FL systems through a layer of validators that evaluate the edge updates. These validators operate on a blockchain infrastructure, tracking the node evaluations. This framework mitigates label-flipping attacks and enables secure training process within FL systems, regardless of the difficulty of the setting, such as whether edge devices have independent and identically distributed (IID) datasets or not. We show the effectiveness of TrustShield by conducting experiments on four datasets: MNIST, CIFAR-10, Chest X-Ray, and a Natural Language Processing dataset. In IID settings, our algorithm improves performance by 52%, 40%, 38%, and 35% across the respective datasets. In the Non-IID settings, we observe improvements of 21%, 56%, 26%, and 39%, respectively. These improvements are compared to state-of-the-art benchmarks and are achieved when half of the nodes are adversarial.
</details>

## FL Scheme and The System Architecture:
After obtaining the weights $w_c^t$ for round $t$ from the cloud, edge node $e_i$ (green) trains a model using its private dataset $\mathcal{D}_{e_i}$ and generates optimal gradients $g_{e_i}^{t*}$. Optimal gradients of all nodes are aggregated to update the cloud's model for the next round $w_c^{t+1}$. However, malicious node $\overline{e_i}$ (red) might perform label flipping to form an adversarial dataset $\overline{\mathcal{D}}_{e_i}$. Then, it can calculate a poisonous gradient $\overline{g}_{e_i}^{t*}$ over the dataset $\overline{\mathcal{D}}_{e_i}$. Sending $\overline{g}_{e_i}^{t*}$ for aggregation can perturb the performance of the cloud model. We propose using a layer of validators, called {\tt TrustShield} (blue region), to certify non-poisonous gradients $g_{e_i}^{t\#}$ for clean aggregation $\overline{g}_{e_i}^{t*}$ by validator $v_j$ (yellow) performing inference over its private dataset $\mathcal{D}_{v_j}$.

## Results
### Vulnerability of Vanilla FL to Poisoning Attacks
![results](./results/ "Vulnerability of Vanilla FL to Poisoning Attacks")

We conducted experiments to present  test accuracies of CIFAR-10 FL model training as the ratio of adversarial users increases in the system. Each curve represents a different round of FL communication. The results clearly indicate a significant degradation in performance as the malicious user activity intensifies, particularly in later rounds.  Remarkably, the shape of the curves for increasing adversarial users bears a striking resemblance to the inverse S-curve commonly observed in the susceptible trajectory in SIR pandemic modeling. The findings underscore the susceptibility of Vanilla FL to poisoning.

### TrustShield outperforms the benchmark Vanilla FL and provides a higher level of robustness compared to the state-of-the-art mechanism ARFED
![results](./results_qa.png "Question Answering Results")

Each row represents the results for 4 different datasets.
Column 1: When edge devices have IID class distributions TrustShield, effectively mitigates the LF attack, reducing its effect to almost zero. In contrast, ARFED fails to counteract high adversarial user ratios and exhibits a performance similar to Vanilla FL.
Column 2: TrustShield, demonstrates high accuracy improvements in any communication round compared to the baseline when half of the participants are adversarial. This holds true regardless of the class distribution of the validator nodes.
Column 3: Training an FL model with edge devices having Non-IID class distributions is challenging. However, TrustShield demonstrates significantly better performance for high adversarial user ratios in the system compared to other methods. 
Column 4: When edge devices have sparse datasets, the FL training with TrustShield, is more stable compared to other methods. TrustShield effectively mitigates the impact of adversarial users regardless of the validator class distributions, leading to a more stable and robust FL training process.

### Sample level performance of the defense mechanisms 
We analyzed the CIFAR-10 dataset by creating a 2D t-SNE map from the CNN embedding to visualize different classes represented by distinct colors. Next, we examined the sample-level performance of defense mechanisms under a 30% adversarial user ratio. Green and purple points represent data samples correctly classified by both mechanisms, while yellow points indicate samples correctly classified only by our algorithm. The notable density of yellow points serves as compelling evidence of the superior efficiency of our algorithm compared to ARFED.

### The Detection Performance of the Algorithm per Round
We simulated an FL experiment using eCommerce NLP dataset with a 70% adversarial user ratio over 20 communication rounds. Our algorithm was activated during the 5th round. The plot illustrates the ratio of undetected malicious edge nodes in red and the ratio of detected collaborative edge nodes in blue. Evidently, the activation of our algorithm effectively reduces the ratio of adversarial nodes used in aggregation among all adversarial nodes. Simultaneously, it maintains a consistent 100\% ratio for collaborative nodes used in aggregation among all collaborative nodes throughout each 

### OoD Detection Functionality of TrustShield
In our CIFAR-10 FL experiments, we simulated different ratios of adversarial users, each represented by different colors. The OoD ratio (proportion of unknown classes) was set at 30%. Activating TrustShield's Poisoning Filtering during the 10th communication round significantly improved model performance for adversarial user ratios higher than 0. At the 35th round, TrustShield's OoD detection further enhanced model performance for all adversarial user ratios by excluding an unknown class that caused confusion during training.

## Installation
### Packages
To run the necessary packages, build a conda environment from the [environment.yml]([environment.yml]) file as in:
```bash
pip install -r requirements.txt
```

### Dataset
#### MNIST Dataset
#### CIFAR-10 Dataset
#### Chest X-Ray Dataset

#### eCommerce Dataset
This is an openly available dataset found on [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification). It a sample dataset chosen for text classification, it contains over 27,000 unique samples of a short description of a product, labelled by its product category amongst Books, Clothing & Accessories, Fashion and Electronics.


## File-Structure

fl_CIFAR_implementation/data - This is the dataset location for CIFAR-10 dataset.

fl_CIFAR_implementation/arfed_cifar10.py - This is the main file to run ARFED against LF attacks with a given adversarial user ratio as described in the [Attack Resistant Federated Learning](https://github.com/eceisik/ARFED).

fl_CIFAR_implementation/sims_cifar10.py - This is the main file used to run the FL for CIFAR10 with/without TrustShield with a given validator (Gaussian/Uniform) & edge data (IID/Non-IID) class distributions.

fl_CIFAR_implementation/fl_utils/ARFED_utils.py - This contains the module to induce protection against poisoning attacks as described in the [Attack Resistant Federated Learning](https://github.com/eceisik/ARFED) paper.

fl_CIFAR_implementation/fl_utils/construct_models.py - This contains the base classes that implement the DNN models.

fl_CIFAR_implementation/fl_utils/distribute_data.py - This contains primitives used to govern the data distribution of edge clients / validators in our FL system, as well as primitives to induce label flipping.

fl_CIFAR_implementation/fl_utils/plotting_utils.py - This contains the functions used for the figure plotting.

fl_CIFAR_implementation/fl_utils/textfile_utils.py - This contains the functions to adjust the text fonts and types in the figures.

fl_CIFAR_implementation/fl_utils/train_nodes.py - This contains the primitives to train and test the FL models (both edge and cloud).

fl_CIFAR_implementation/fl_utils/tsneCIFAR10_training.py - 

fl_CIFAR_implementation/fl_utils/tsneCIFAR10_training_classes.py - 

fl_CIFAR_implementation/fl_utils/tsne_estimationCIFAR10.py - 

fl_MNIST_implementation/data - This is the dataset location for MNIST dataset.

fl_MNIST_implementation/arfed_mnist.py - This is the main file to run ARFED against LF attacks with a given adversarial user ratio as described in the [Attack Resistant Federated Learning](https://github.com/eceisik/ARFED).

fl_MNIST_implementation/sims_mnist.py - This is the main file used to run the FL for MNIST with/without TrustShield with a given validator (Gaussian/Uniform) & edge data (IID/Non-IID) class distributions.

fl_MNIST_implementation/fl_utils/ARFED_utils.py - This contains the module to induce protection against poisoning attacks as described in the [Attack Resistant Federated Learning](https://github.com/eceisik/ARFED) paper.

fl_MNIST_implementation/fl_utils/construct_models.py - This contains the base classes that implement the DNN models.

fl_MNIST_implementation/fl_utils/distribute_data.py - This contains primitives used to govern the data distribution of edge clients / validators in our FL system, as well as primitives to induce label flipping.

fl_MNIST_implementation/fl_utils/plotting_utils.py - This contains the functions used for the figure plotting.

fl_MNIST_implementation/fl_utils/textfile_utils.py - This contains the functions to adjust the text fonts and types in the figures.

fl_MNIST_implementation/fl_utils/train_nodes.py - This contains the primitives to train and test the FL models (both edge and cloud).

fl_medical_implementation/data - This is the location for Chest X-Ray dataset to be downloaded.

fl_medical_implementation/sims_med.py - This is the main file used to run the FL for Chest X-Ray with/without TrustShield with a given validator (Gaussian/Uniform) & edge data (IID/Non-IID) class distributions.

fl_medical_implementation/sims_med_ARFED.py - This is the main file to run ARFED against LF attacks with a given adversarial user ratio as described in the [Attack Resistant Federated Learning](https://github.com/eceisik/ARFED).

fl_medical_implementation/fl_utils/ARFED_utils.py - This contains the module to induce protection against poisoning attacks as described in the [Attack Resistant Federated Learning](https://github.com/eceisik/ARFED) 

fl_medical_implementation/fl_utils/construct_models.py - This contains the base classes that implement the DNN models.

fl_medical_implementation/fl_utils/distribute_data.py - This contains primitives used to govern the data distribution of edge clients / validators in our FL system, as well as primitives to induce label flipping.

fl_medical_implementation/fl_utils/plotting_utils.py - This contains the functions used for the figure plotting.

fl_medical_implementation/fl_utils/textfile_utils.py - This contains the functions to adjust the text fonts and types in the figures.

fl_medical_implementation/fl_utils/train_nodes.py - This contains the primitives to train and test the FL models (both edge and cloud).

fl_nlp_implementation/adv_utils/ARFED_utils.py - This contains the module to induce protection against poisoning attacks as described in the [Attack Resistant Federated Learning](https://github.com/eceisik/ARFED) paper.

fl_nlp_implementation/adv_utils/construct_models.py - This contains the base classes that implement the BERT models from Huggingface transformers.

fl_nlp_implementation/adv_utils/dataloader.py - This contains the base classes to load data for both the classification and question answering tasks.

fl_nlp_implementation/adv_utils/distribute_data.py - This contains primitives used to govern the data distribution of edge clients / validators in our FL system (for text classification), as well as primitives to induce label flipping.

fl_nlp_implementation/adv_utils/train_nodes.py - This contains the primitives to train and test the FL models (both edge and cloud).

fl_nlp_implementation/sims_adversarial.py - This is the main file used to run the FL system for the Text Classification task.

fl_nlp_implementation/ARFED_sims.py - This is the main file to run ARFED against LF attacks with a given adversarial user ratio as described in the [Attack Resistant Federated Learning](https://github.com/eceisik/ARFED).


## Usage

### Training a BERT Model for Classification
Run the following snippet to simulate FL - with / without poisoning attacks and with / without validation.
```
python3 sims_adversarial.py --test_num_classes <> --train_num_classes <> --trainamount <> --testamount <> --nepoch <> --nrounds <> --numusers <> --applymining <> --initround <> --valn <> --device <>
```

Run the following to understand the input arguments -
```
python3 sims_adversarial.py --help
```
