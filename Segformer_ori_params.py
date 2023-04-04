import argparse
import json
import os
import torch

from model_segformer import segformer
from model_default_config import LDoptions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Tusimple")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return args
args = parse_args()

device = torch.device(args.device)

model_config = LDoptions()
dataset = args.dataset_name
net = segformer(model_config, dataset, pretrained=True)
net = net.to(device)

total_params = sum(param.numel() for param in net.parameters() if param.requires_grad)
print("{}M".format(total_params/1e6))