import argparse
import json
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from config import *
from utils.prob2lines import getLane
from utils.transforms import *
from trt.trt_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
    parser.add_argument("--trt_dir", type=str, help="directory to trt engine file")
    parser.add_argument("--batch_size", type=int, help="batch size of inputs for the engine file")
    args = parser.parse_args()
    return args



# ------------ config ------------
args = parse_args()
exp_dir = args.exp_dir
exp_name = exp_dir.split("/")[-1]
trt_dir = args.trt_dir
input_batch_size = args.batch_size

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg["dataset"]["resize_shape"])

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
dataset_name = "CULane"
Dataset_Type = getattr(dataset, dataset_name)
transform = Compose(
    Resize(resize_shape, dataset_name), ToTensor(), Normalize(mean=mean, std=std)
)
test_dataset = Dataset_Type(Dataset_Path[dataset_name], "test", transform)
test_loader = DataLoader(
    test_dataset, batch_size=input_batch_size, collate_fn=test_dataset.collate, num_workers=4, drop_last=True
)

# load trt engine
encoder_context, encoder_inputs,encoder_outputs,encoder_bindings,encoder_stream = initialize_trt_engine(trt_dir)

# ------------ test ------------
out_path = os.path.join(exp_dir, "trt_output", "coord_output")
evaluation_path = os.path.join(exp_dir, "trt_output", "evaluate")
out_idx_path = os.path.join(exp_dir, "trt_output", "lane_idx_output")
if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(out_idx_path):
    os.makedirs(out_idx_path)
if not os.path.exists(evaluation_path):
    os.makedirs(evaluation_path)

progressbar = tqdm(range(len(test_loader)))

for batch_idx, sample in enumerate(test_loader):
    img = sample["img"]
    batch, channel, height, width = img.shape
    encoder_inputs[0].host = np.ascontiguousarray(img, dtype=np.int8)
    img_name = sample["img_name"]

    e_out = do_inference(encoder_context,
                        bindings=encoder_bindings,
                        inputs=encoder_inputs,
                        outputs=encoder_outputs,
                        stream=encoder_stream,
                        batch_size=batch
                        )

    # output reshape
    e_out[0] = e_out[0].reshape((batch, 5, height, width))
    e_out[1] = e_out[1].reshape((batch, 4))

    # output postprocessing
    seg_pred, exist_pred = e_out[:2]
    seg_pred = torch.from_numpy(seg_pred).type(torch.float)
    seg_pred = F.softmax(seg_pred, dim=1)
    seg_pred = seg_pred.detach().cpu().numpy()
    
    for b in range(len(seg_pred)):
        seg = seg_pred[b]
        exist = [1 if exist_pred[b, i] > 0.5 else 0 for i in range(4)]
        lane_coords, lane_idx = getLane.prob2lines_CULane(
            seg, exist, resize_shape=(590, 1640), y_px_gap=20, pts=18
        )
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
