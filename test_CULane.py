import argparse
import json
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from config import *
from model import SCNN
from utils.prob2lines import getLane, curve_fitting_CULane
from utils.transforms import *
from model_segformer import segformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp10")
    args = parser.parse_args()
    return args


# ------------ config ------------
args = parse_args()
exp_dir = args.exp_dir
exp_name = exp_dir.split("/")[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg["dataset"]["resize_shape"])
device = torch.device("cuda")


def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders


# ------------ data and model ------------
# # CULane mean, std
mean = (0.3598, 0.3653, 0.3662)
std = (0.2573, 0.2663, 0.2756)
# Imagenet mean, std
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
dataset_name = exp_cfg["dataset"].pop("dataset_name")
Dataset_Type = getattr(dataset, dataset_name)
transform = Compose(
    Resize(resize_shape, dataset_name), ToTensor(), Normalize(mean=mean, std=std)
)
test_dataset = Dataset_Type(Dataset_Path[dataset_name], "test", transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, collate_fn=test_dataset.collate, num_workers=4
)
model_config = exp_cfg["MODEL_CONFIG"]
net = segformer(model_config, dataset_name,pretrained=True)
# net = SCNN(resize_shape, pretrained=False)
save_name = os.path.join(exp_dir, exp_dir.split("/")[-1] + "_best.pth")
save_dict = torch.load(save_name, map_location="cpu")
print("\nloading", save_name, "...... From Epoch: ", save_dict["epoch"])
net.load_state_dict(save_dict["net"])
net = net.to(device)
net.eval()

# ------------ test ------------
out_path = os.path.join(exp_dir, "coord_output")
out_idx_path = os.path.join(exp_dir, "lane_idx_output")
evaluation_path = os.path.join(exp_dir, "evaluate")
if not os.path.exists(out_path):
    os.mkdir(out_path)
if not os.path.exists(out_idx_path):
    os.mkdir(out_idx_path)
if not os.path.exists(evaluation_path):
    os.mkdir(evaluation_path)

progressbar = tqdm(range(len(test_loader)))
with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
        img = sample["img"].to(device)
        img_name = sample["img_name"]

        seg_pred, exist_pred = net(img)[:2]
        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()

        for b in range(len(seg_pred)):
            seg = seg_pred[b]
            exist = [1 if exist_pred[b, i] > 0.5 else 0 for i in range(4)]
            lane_coords, lane_idx = getLane.prob2lines_CULane(
                seg, exist, resize_shape=(590, 1640), y_px_gap=20, pts=18
            )

            # # RANSAC curve fitting --- start
            preprocessed_lane_coords = curve_fitting_CULane.lane_coords_preprocess(lane_coords)
            predicted_lane_coords = curve_fitting_CULane.ransac_fitting(
                preprocessed_lane_coords
            )
            lane_coords = curve_fitting_CULane.lane_coords_postprocess(predicted_lane_coords)
            # # RANSAC curve fitting --- end
            path_tree = split_path(img_name[b])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_idx_dir, save_idx_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(out_path, *save_dir)
            save_idx_dir = os.path.join(out_idx_path, *save_idx_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_idx_name = save_idx_name[:-3] + "lines.txt"
            save_name = os.path.join(save_dir, save_name)
            save_idx_name = os.path.join(save_idx_dir, save_idx_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(save_idx_dir):
                os.makedirs(save_idx_dir)
            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)
            with open(save_idx_name, "w") as f:
                for i in range(len(lane_idx)):
                    print(lane_idx[i], end=" ", file=f)

        progressbar.update(1)
progressbar.close()

# ---- evaluate ----
os.system("sh utils/lane_evaluation/CULane/Run.sh " + exp_name)