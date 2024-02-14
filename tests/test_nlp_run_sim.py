from TrustShield import simulations as ts
import yaml
import torch

# Read YAML data from a file
with open("./TrustShield/nlpconfig.yml", "r") as file:
    opt = yaml.load(file, Loader=yaml.FullLoader)
# yaml.load() returns a dictionary containing the YAML data
print(opt)
device = []
devices = opt["devices"]
for d in devices:
   device.append(torch.device("cuda:"+str(d) if (torch.cuda.is_available()) else "cpu"))
ts.nlp_run_sim(opt, device, isdict=True, verbose=False)