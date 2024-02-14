from TrustShield import simulations as ts
import yaml
import torch

# Read YAML data from a file
with open("./TrustShield/mnistconfig.yml", "r") as file:
    opt = yaml.load(file, Loader=yaml.FullLoader)
# yaml.load() returns a dictionary containing the YAML data
print(opt)
device=torch.device("cuda:"+str(opt["deviceno"]) if (torch.cuda.is_available()) else "cpu")
ts.mnist_run_sim(opt, device, isdict=True, verbose=False)