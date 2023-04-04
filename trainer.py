import argparse
import json
import os
import shutil

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from config import *
import dataset
from options import LDoptions
from model_segformer import segformer
from utils.tensorboard import TensorBoard
from utils.transforms import *
from utils.lr_scheduler import PolyLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
import wandb

class_labels = {
    0: "background pixels",
    1: "left lane",
    2: "ego left",
    3: "ego right",
    4: "right lane",
}


class Trainer:
    def __init__(self, options):

        self.opts = options
        # ------------ config ------------
        self.exp_dir = self.opts.exp_dir
        while self.exp_dir[-1] == "/":
            self.exp_dir = self.exp_dir[:-1]
        self.exp_name = self.exp_dir.split("/")[-1]

        self.device = torch.device(f"cuda:{self.opts.device_id}")
        self.max_epoch = self.opts.max_epochs
        # self.tensorboard = TensorBoard(exp_dir)

        dataset_name = self.opts.dataset_name
        # ------------ train data ------------
        if dataset_name == "CULane":
            # CULane mean, std
            mean = (0.3598, 0.3653, 0.3662)
            std = (0.2573, 0.2663, 0.2756)
        elif dataset_name == "Tusimple":
            # Imagenet mean, std
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        transform_train = Compose(
            Resize(tuple(self.opts.resize_shape), dataset_name),
            ColorJitter(),
            RandomFlip(self.opts.flip_prob),
            Rotation(2),
            ToTensor(),
            Normalize(mean=mean, std=std),
        )

        Dataset_Type = getattr(dataset, dataset_name)
        train_dataset = Dataset_Type(
            Dataset_Path[dataset_name], "train", transform_train
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate,
            num_workers=8,
        )

        # ------------ val data ------------
        self.transform_val_img = Resize(tuple(self.opts.resize_shape), dataset_name)
        transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
        transform_val = Compose(self.transform_val_img, transform_val_x)
        val_dataset = Dataset_Type(Dataset_Path[dataset_name], "val", transform_val)
        self.val_loader = DataLoader(
            val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=4
        )

        # ------------ preparation ------------
        self.net = segformer(self.opts, dataset_name, pretrained=True)

        self.net = self.net.to(self.device)

        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=self.opts.learning_rate,
            weight_decay=self.opts.weight_decay,
        )
        # self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[6, 9], gamma=0.1, verbose=True)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, patience=5, verbose=True
        )  # 3.6833
        self.best_val_loss = 1e6
        self.start_epoch = 0

        if self.opts.resume:
            save_dict = torch.load("/home/automan/wuguanjie/SCNN_Pytorch/experiments/exp_CU_baseline_flip_sr2211_GELU_se/exp_CU_baseline_flip_sr2211_GELU_se_best.pth")
            self.net.load_state_dict(save_dict["net"])
            self.lr_scheduler.load_state_dict(save_dict["lr_scheduler"])
            self.start_epoch = save_dict['epoch'] + 1
            self.best_val_loss = save_dict.get("best_val_loss", 1e6)

    def train(self):
        # wandb.watch(self.net, log="all", log_freq=100)
        for epoch in range(self.start_epoch, self.max_epoch):
            print("Train Epoch: {}".format(epoch))
            self.net.train()
            train_loss = 0
            train_loss_seg = 0
            train_loss_exist = 0
            progressbar = tqdm(range(len(self.train_loader)))

            for batch_idx, sample in enumerate(self.train_loader):
                img = sample["img"].to(self.device)
                segLabel = sample["segLabel"].to(self.device)
                exist = sample["exist"].to(self.device)

                self.optimizer.zero_grad()
                seg_pred, exist_pred, loss_seg, loss_exist, loss = self.net(
                    img, segLabel, exist
                )
                loss.backward()
                self.optimizer.step()
                # lr_scheduler.step()

                iter_idx = epoch * len(self.train_loader) + batch_idx
                train_loss = loss.item()
                train_loss_seg = loss_seg.item()
                train_loss_exist = loss_exist.item()
                progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
                progressbar.update(1)

                lr = self.optimizer.param_groups[0]["lr"]

                if batch_idx % 100 == 0:
                    wandb.log(
                        {
                            "train/loss": train_loss,
                            "train/loss_seg": train_loss_seg,
                            "train/loss_exist": train_loss_exist,
                        },
                        step=iter_idx,
                    )
                    # self.tensorboard.scalar_summary(self.exp_name + "/train_loss", train_loss, iter_idx)
                    # self.tensorboard.scalar_summary(self.exp_name + "/train_loss_seg", train_loss_seg, iter_idx)
                    # self.tensorboard.scalar_summary(self.exp_name + "/train_loss_exist", train_loss_exist, iter_idx)
                    # self.tensorboard.scalar_summary(self.exp_name + "/learning_rate", lr, iter_idx)
            progressbar.close()
            # self.tensorboard.writer.flush()

            if epoch % 1 == 0:
                save_dict = {
                    "epoch": epoch,
                    "net": self.net.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "best_val_loss": self.best_val_loss,
                }
                save_name = os.path.join(self.exp_dir, self.exp_name + ".pth")
                torch.save(save_dict, save_name)
                print("model is saved: {}".format(save_name))
            wandb.log({"learning_rate": lr}, step=epoch)
            print("------------------------\n")
            val_loss = self.val(epoch)
            self.lr_scheduler.step(val_loss)

    def wb_mask(self, val_img, pred_mask, true_mask):
        return wandb.Image(
            val_img,
            masks={
                "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
                "ground truth": {"mask_data": true_mask, "class_labels": class_labels},
            },
        )

    def val(self, epoch):
        print("Val Epoch: {}".format(epoch))

        self.net.eval()
        val_loss = 0
        val_loss_seg = 0
        val_loss_exist = 0
        progressbar = tqdm(range(len(self.val_loader)))

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                img = sample["img"].to(self.device)
                segLabel = sample["segLabel"].to(self.device)
                exist = sample["exist"].to(self.device)

                seg_pred, exist_pred, loss_seg, loss_exist, loss = self.net(
                    img, segLabel, exist
                )

                # visualize validation every 5 frame, 50 frames in all
                gap_num = 5
                if batch_idx % gap_num == 0 and batch_idx < 50 * gap_num:
                    origin_imgs = []
                    mask_list = []
                    seg_pred = seg_pred.detach().cpu().numpy()
                    exist_pred = exist_pred.detach().cpu().numpy()

                    for b in range(len(img)):
                        img_name = sample["img_name"][b]
                        img = cv2.imread(img_name)
                        img = self.transform_val_img({"img": img})["img"]

                        lane_img = np.zeros_like(img)
                        color = np.array(
                            [[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]],
                            dtype="uint8",
                        )

                        coord_mask = np.argmax(seg_pred[b], axis=0)
                        for i in range(0, 4):
                            if exist_pred[b, i] > 0.5:
                                lane_img[coord_mask == (i + 1)] = color[i]

                        img = cv2.addWeighted(
                            src1=lane_img, alpha=0.8, src2=img, beta=1.0, gamma=0.0
                        )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
                        cv2.putText(
                            lane_img,
                            "{}".format(
                                [1 if exist_pred[b, i] > 0.5 else 0 for i in range(4)]
                            ),
                            (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.1,
                            (255, 255, 255),
                            2,
                        )
                        origin_imgs.append(img)
                        origin_imgs.append(lane_img)
                        # mask_list.append(self.wb_mask(sample["img"][b],coord_mask,sample["segLabel"][b].numpy()))
                    # wandb.log({"prediction{}".format(batch_idx) : mask_list})
                    # self.tensorboard.image_summary("img_{}".format(batch_idx), origin_imgs, epoch)

                val_loss += loss.item()
                val_loss_seg += loss_seg.item()
                val_loss_exist += loss_exist.item()

                progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
                progressbar.update(1)

        progressbar.close()
        iter_idx = (epoch + 1) * len(
            self.train_loader
        )  # keep align with training process iter_idx
        wandb.log(
            {
                "val/loss": val_loss,
                "val/loss_seg": val_loss_seg,
                "val/loss_exist": val_loss_exist,
            },
            step=iter_idx,
        )
        # self.tensorboard.scalar_summary("val_loss", val_loss, iter_idx)
        # self.tensorboard.scalar_summary("val_loss_seg", val_loss_seg, iter_idx)
        # self.tensorboard.scalar_summary("val_loss_exist", val_loss_exist, iter_idx)
        # self.tensorboard.writer.flush()

        print("------------------------\n")
        print("best val loss", self.best_val_loss)
        print("current val loss", val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            wandb.run.summary["lowest_validation_loss"] = self.best_val_loss
            save_name = os.path.join(self.exp_dir, self.exp_name + ".pth")
            copy_name = os.path.join(self.exp_dir, self.exp_name + "_best.pth")
            shutil.copyfile(save_name, copy_name)

        return val_loss


if __name__ == "__main__":
    # parsing options
    options_obj = LDoptions()
    opts = options_obj.parse()
    print(opts)
    # setting seed to preserve reproducibility
    seed = opts.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ablation study",
        name="resume + CULane baseline + flip + sr2211 + GELU + se",
        tags=["baseline", "flip", "sr2211", "GELU", "se"],
        notes="resume + CULane baseline + flip + sr2211 + GELU + se",
        # track hyperparameters and run metadata
        config=vars(opts),
    )
    trainer = Trainer(opts)
    trainer.train()
