import argparse
import cv2
import torch

from model_SCNN import SCNN
from utils.prob2lines import getLane
from utils.transforms import *
from model_segformer import segformer
from model_default_config import LDoptions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/demo.jpg", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, help="Path to model weights")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    parser.add_argument("--dataset_name", type=str, default = "CULane", help="Tusimple or CULane")
    parser.add_argument("--resize_shape", nargs="+", type=int, default=[800,288])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    weight_path = args.weight_path

    model_config = LDoptions()
    dataset = args.dataset_name
    assert dataset in ["Tusimple", "CULane"]
    rs = args.resize_shape
    net = segformer(model_config, dataset, pretrained=True)
    # net = SCNN(input_size=(800, 288), pretrained=False)
    mean=(0.3598, 0.3653, 0.3662) # CULane mean, std
    std=(0.2573, 0.2663, 0.2756)
    transform_img = Resize(rs, dataset)
    transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform_img({'img': img})['img']
    x = transform_to_net({'img': img})['img']
    x.unsqueeze_(0)

    save_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(save_dict['net'])
    net.eval()

    seg_pred, exist_pred = net(x)[:2]
    seg_pred = seg_pred.detach().cpu().numpy()
    exist_pred = exist_pred.detach().cpu().numpy()
    seg_pred = seg_pred[0]
    exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, 4):
        if exist_pred[0, i] > 0.5:
            lane_img[coord_mask == (i + 1)] = color[i]
    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    saving_path = "demo/demo_result.jpg"
    cv2.imwrite(saving_path, img)
    print("img is saved in {}".format(saving_path))

    # for x in getLane.prob2lines_CULane(seg_pred, exist):
    #     print(x)

    if args.visualize:
        print([1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)])
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
