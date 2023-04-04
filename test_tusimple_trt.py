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

resize_shape = (512,288)

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
# Imagenet mean, std
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
dataset_name = "Tusimple"
transform = Compose(
    Resize(resize_shape, dataset_name), ToTensor(), Normalize(mean=mean, std=std)
)
Dataset_Type = getattr(dataset, dataset_name)
test_dataset = Dataset_Type(Dataset_Path["Tusimple"], "test", transform)
test_loader = DataLoader(
    test_dataset, batch_size=input_batch_size, collate_fn=test_dataset.collate, num_workers=4
)

# load trt engine
encoder_context, encoder_inputs,encoder_outputs,encoder_bindings,encoder_stream = initialize_trt_engine(trt_dir)

# ------------ test ------------
out_path = os.path.join(exp_dir, "trt_output", "coord_output")
evaluation_path = os.path.join(exp_dir, "trt_output", "evaluate")
if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(evaluation_path):
    os.makedirs(evaluation_path)
dump_to_json = []

progressbar = tqdm(range(len(test_loader)))
for batch_idx, sample in enumerate(test_loader):
    img = sample["img"]
    batch, channel, height, width = img.shape
    encoder_inputs[0].host = np.ascontiguousarray(img, dtype=np.float32)
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
        lane_coords, lane_idx = getLane.prob2lines_tusimple(
            seg, exist, resize_shape=(720, 1280), y_px_gap=10, pts=56
        )
        for i in range(len(lane_coords)):
            lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])
        # # RANSAC curve fitting --- start
        # preprocessed_lane_coords = curve_fitting.lane_coords_preprocess(lane_coords)
        # predicted_lane_coords = curve_fitting.ransac_fitting(
        #     preprocessed_lane_coords
        # )
        # lane_coords = curve_fitting.lane_coords_postprocess(predicted_lane_coords)
        # # RANSAC curve fitting --- end
        path_tree = split_path(img_name[b])
        save_dir, save_name = path_tree[-3:-1], path_tree[-1]
        save_dir = os.path.join(out_path, *save_dir)
        save_name = save_name[:-3] + "lines.txt"
        save_name = os.path.join(save_dir, save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with open(save_name, "w") as f:
            for l in lane_coords:
                for (x, y) in l:
                    print("{} {}".format(x, y), end=" ", file=f)
                print(file=f)
            for i in range(len(lane_idx)):
                print(lane_idx[i], end=" ", file=f)

        json_dict = {}
        json_dict["lanes"] = []
        json_dict["h_sample"] = []
        json_dict["raw_file"] = os.path.join(*path_tree[-4:])
        json_dict["run_time"] = 0
        json_dict["lane_idx"] = lane_idx
        for l in lane_coords:
            if len(l) == 0:
                continue
            json_dict["lanes"].append([])
            for (x, y) in l:
                json_dict["lanes"][-1].append(int(x))
        for (x, y) in lane_coords[0]:
            json_dict["h_sample"].append(y)
        dump_to_json.append(json.dumps(json_dict))

    progressbar.update(1)
progressbar.close()

with open(os.path.join(out_path, "predict_test.json"), "w") as f:
    for line in dump_to_json:
        print(line, end="\n", file=f)

# ---- evaluate ----
from utils.lane_evaluation.tusimple.lane import LaneEval

eval_result = LaneEval.bench_one_submit(
    os.path.join(out_path, "predict_test.json"),
    os.path.join(Dataset_Path["Tusimple"], "test_label.json"),
)
print(eval_result)
with open(os.path.join(evaluation_path, "evaluation_result.txt"), "w") as f:
    print(eval_result, file=f)
