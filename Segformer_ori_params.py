import argparse
import json
import os
import torch

from model import SCNN
from model_segformer import segformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
while exp_dir[-1]=='/':
    exp_dir = exp_dir[:-1]
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
device = torch.device(exp_cfg['device'])

# ------------ preparation ------------
model_config = exp_cfg['MODEL_CONFIG']
net = segformer(model_config, pretrained=True)
net = net.to(device)

total_params = sum(param.numel() for param in net.parameters() if param.requires_grad)
print(total_params)