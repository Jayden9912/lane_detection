import os
import time
import torch
import json
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from utils.prob2lines import getLane
from model_segformer import segformer
from options import LDoptions

# cuDnn configurations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

opts = LDoptions()
args = opts.parse()
exp_dir = args.exp_dir
iteration = 1000
dataset = args.dataset_name
assert dataset in ["Tusimple", "CULane"]

model = segformer(args, dataset, pretrained=True)

# load the pretrained weight of the model
save_name = os.path.join(exp_dir, exp_dir.split("/")[-1] + "_best.pth")
save_dict = torch.load(save_name, map_location="cpu")
print("\nloading", save_name, "...... From Epoch: ", save_dict["epoch"])
model.load_state_dict(save_dict["net"])
model = model.to("cuda:2")
model.eval()

# prepare random input
name = "Segformer"
print("     + {} Speed testing... ...".format(name))
if dataset == "CULane":
    random_input = torch.randn(1,3,288, 800).to("cuda:2")
else:
    random_input = torch.randn(1,3,288, 512).to("cuda:2")


time_list = []
for i in tqdm(range(iteration)):
    torch.cuda.synchronize()
    tic = time.time()
    seg_pred, exist_pred = model(random_input)[:2]
    seg_pred = F.softmax(seg_pred, dim=1)
    seg_pred = seg_pred.detach().cpu().numpy()
    exist_pred = exist_pred.detach().cpu().numpy()

    for b in range(len(seg_pred)):
        seg = seg_pred[b]
        exist = [1 if exist_pred[b, i] > 0.5 else 0 for i in range(4)]
        if dataset == "CULane":
            lane_coords, lane_idx = getLane.prob2lines_CULane(seg, exist, resize_shape=(590, 1640), y_px_gap=20, pts=18)
        else:
            lane_coords, lane_idx = getLane.prob2lines_tusimple(
                seg, exist, resize_shape=(720, 1280), y_px_gap=10, pts=56
            )
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])
    torch.cuda.synchronize()
    time_list.append(time.time()-tic)
# the first iteration time cost much higher, so exclude the first iteration
time_list = time_list[1:]
print("     + Done {} iterations inference !".format(iteration))
print("     + Total time cost: {}s".format(sum(time_list)))
print("     + Average time cost: {}s".format(sum(time_list)/(iteration-1)))
print("     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/(iteration-1))))