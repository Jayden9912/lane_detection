import os
import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from trt.trt_utils import *
import torch.nn.functional as F
from utils.prob2lines import getLane
from utils.visualisation import result_visualisation
from torchvision.transforms import Normalize as Normalize_th



def img_preprocessing(img, dataset):
    if dataset == "CULane":
        mean=(0.3598, 0.3653, 0.3662)
        std=(0.2573, 0.2663, 0.2756)
        img = cv2.resize(img, (800, 288))
    else: #Tusimple
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        img = cv2.resize(img, (512, 288))
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).type(torch.float) / 255.0

    transform = Normalize_th(mean, std)
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.numpy()
    return img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt_dir", type=str, help = "full directory to the engine file")
    parser.add_argument("--iteration", type=int, default=200, help = "iteration number")
    parser.add_argument("--dataset", type=str, help = "dataset name (Tusimple or CULane)")
    parser.add_argument("--saving_path", type=str, default="/home/automan/wuguanjie/SCNN_Pytorch/trt", help="saving path for the visualization of the image")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    trt_dir = args.trt_dir
    iteration = args.iteration
    dataset = args.dataset
    assert dataset in ["Tusimple", "CULane"]
    
    saving_path = args.saving_path
    saving_path = saving_path
    
    encoder_context, encoder_inputs,encoder_outputs,encoder_bindings,encoder_stream = initialize_trt_engine(trt_dir)
    
    if dataset == "CULane":
        pred_img = cv2.imread("/home/automan/wuguanjie/CULane_path/driver_37_30frame/05191358_0443.MP4/00060.jpg")
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    else: #Tusimple
        pred_img = cv2.imread("/home/automan/wuguanjie/TuSimple/test/clips/0531/1492626253262712112/20.jpg")
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    vis_img = pred_img.copy()

    # data preprocessing
    pred_img = img_preprocessing(pred_img, dataset)
    encoder_inputs[0].host = np.ascontiguousarray(pred_img, dtype=np.float32)

    time_list = []

    # inference
    for i in tqdm(range(iteration)):
        tic = time.time()
        e_out = do_inference(
            encoder_context,
            bindings=encoder_bindings,
            inputs=encoder_inputs,
            outputs=encoder_outputs,
            stream=encoder_stream,
        )

        # output reshape
        if dataset == "CULane":
            e_out[0] = e_out[0].reshape((1, 5, 288, 800))
        else:
            e_out[0] = e_out[0].reshape((1, 5, 288, 512))
        e_out[1] = e_out[1].reshape((1, 4))

        # output postprocessing
        seg_pred, exist_pred = e_out[:2]
        seg_pred = torch.from_numpy(seg_pred).type(torch.float)
        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.detach().cpu().numpy()
        for b in range(len(seg_pred)):
            seg = seg_pred[b]
            exist = [1 if exist_pred[b, i] > 0.5 else 0 for i in range(4)]
            if dataset == "CULane":
                lane_coords, lane_idx = getLane.prob2lines_CULane(seg, exist, resize_shape=(590, 1640), y_px_gap=20, pts=18)
            else:
                lane_coords, lane_idx = getLane.prob2lines_tusimple(seg, exist, resize_shape=(720, 1280), y_px_gap=10, pts=56)
        time_list.append(time.time()-tic)
    time_list = time_list[1:]
    print("Done {} iterations inference !".format(iteration))
    print("Total time cost: {}s".format(sum(time_list)))
    print("Average time cost: {}s".format(sum(time_list)/(iteration-1)))
    print("Frame Per Second: {:.2f}".format(1/(sum(time_list)/(iteration-1))))

    vis_img = result_visualisation(vis_img,lane_idx,lane_coords)
    cv2.imwrite(os.path.join(saving_path,"result_{}.jpg".format(dataset)), vis_img)
    






