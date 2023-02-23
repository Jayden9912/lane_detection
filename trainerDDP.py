import argparse
import json
import os
import shutil

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter

from config import *
import dataset
from model_segformer import segformer
from utils.tensorboard import TensorBoard
from utils.transforms import *
from utils.lr_scheduler import PolyLR
from torch_poly_lr_decay import PolynomialLRDecay
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    init_process_group(backend="nccl")


class Trainer:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        while self.exp_dir[-1] == "/":
            self.exp_dir = self.exp_dir[:-1]
        self.exp_name = self.exp_dir.split("/")[-1]
        with open(os.path.join(self.exp_dir, "cfg.json")) as f:
            self.exp_cfg = json.load(f)

        self.resize_shape = tuple(self.exp_cfg["dataset"]["resize_shape"])
        # self.device = torch.device(self.exp_cfg["device"])
        self.device = int(os.environ["LOCAL_RANK"])
        self.max_epoch = self.exp_cfg["MAX_EPOCHES"]
        self.dataset_name = self.exp_cfg["dataset"].pop("dataset_name")
        self.snapshot_path = os.path.join(self.exp_dir, self.exp_name + ".pth")
        self.start_epoch = 0

        # mean and std
        if self.dataset_name == "CULane":
            # CULane mean, std
            mean = (0.3598, 0.3653, 0.3662)
            std = (0.2573, 0.2663, 0.2756)
        elif self.dataset_name == "Tusimple":
            # Imagenet mean, std
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        transform_train = Compose(
            Resize(self.resize_shape, self.dataset_name),
            ColorJitter(),
            RandomFlip(),
            Rotation(2),
            ToTensor(),
            Normalize(mean=mean, std=std),
        )
        Dataset_Type = getattr(dataset, self.dataset_name)
        train_dataset = Dataset_Type(
            Dataset_Path[self.dataset_name], "train", transform_train
        )

        # self.train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=self.exp_cfg["dataset"]["batch_size"],
        #     shuffle=True,
        #     collate_fn=train_dataset.collate,
        #     num_workers=8,
        # )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.exp_cfg["dataset"]["batch_size"],
            shuffle=False,
            collate_fn=train_dataset.collate,
            num_workers=8,
            sampler=DistributedSampler(train_dataset),
        )

        # ------------ val data ------------
        self.transform_val_img = Resize(self.resize_shape, self.dataset_name)
        transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
        transform_val = Compose(self.transform_val_img, transform_val_x)
        val_dataset = Dataset_Type(
            Dataset_Path[self.dataset_name], "val", transform_val
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=4
        )

        # ------------ preparation ------------
        model_config = self.exp_cfg["MODEL_CONFIG"]
        # parallel_training = self.exp_cfg["MODEL_CONFIG"].pop("parallel_training")
        self.net = segformer(model_config, self.dataset_name, pretrained=True)
        self.net = self.net.to(self.device)
        self.net = DDP(self.net, device_ids=[self.device], find_unused_parameters=True)

        self.optimizer = optim.AdamW(self.net.parameters(), **self.exp_cfg["optim"])
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=[3, 6, 9], gamma=0.1, verbose=True
        )
        # self.lr_scheduler = PolynomialLRDecay(
        #     self.optimizer, max_decay_steps=362600, end_learning_rate=0.0, power=1.0
        # )
        self.best_val_loss = 1e6

        # tensorboard
        self.writers = SummaryWriter(log_dir=self.exp_dir)

        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self.load_snapshot()

    def load_snapshot(self):
        snapshot = torch.load(self.snapshot_path)
        self.net.module.load_state_dict(snapshot["net"])
        self.optimizer.load_state_dict(snapshot["optim"])
        self.lr_scheduler.load_state_dict(snapshot["lr_scheduler"])
        self.start_epoch = snapshot["epoch"] + 1
        self.best_val_loss = snapshot.get("best_val_loss", 1e6)
        print(f"Resuming training from snapshot at epoch {self.start_epoch}")

    def set_train(self):
        """convert the model to training mode"""
        self.net.train()

    def set_eval(self):
        """convert the model to evaluation mode"""
        self.net.eval()

    def train(self):
        """run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()

    def run_epoch(self):
        """Run a single epoch of training and validation"""
        self.set_train()
        self.training_losses = {}
        progressbar = tqdm(range(len(self.train_loader)))
        print("Train Epoch: {}".format(self.epoch))
        self.train_loader.sampler.set_epoch(self.epoch)
        for batch_idx, sample in enumerate(self.train_loader):
            # print("learning_rate", self.lr_scheduler.get_lr())
            img = sample["img"].to(self.device)
            segLabel = sample["segLabel"].to(self.device)
            exist = sample["exist"].to(self.device)

            self.optimizer.zero_grad()
            seg_pred, exist_pred, loss_seg, loss_exist, loss = self.net(
                img, segLabel, exist
            )
            loss.backward()
            self.optimizer.step()
            # self.lr_scheduler.step()
            progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
            progressbar.update(1)

            self.training_losses["train_loss"] = loss.item()
            self.training_losses["train_loss_seg"] = loss_seg.item()
            self.training_losses["train_loss_exist"] = loss_exist.item()

            early_phase = self.step < 1500
            late_phase = self.step % 100 == 0
            if early_phase or late_phase:
                self.log(self.training_losses)
            self.step += 1
        if self.device == 0:
            self.save_snapshot()
        progressbar.close()
        self.lr_scheduler.step()
        self.val()

    def val(self):
        self.set_eval()
        print("Val Epoch: {}".format(self.epoch))
        self.validation_losses = {}
        self.validation_losses["val_loss"] = 0
        self.validation_losses["val_loss_seg"] = 0
        self.validation_losses["val_loss_exist"] = 0
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
                    self.log_img(origin_imgs, batch_idx)

                self.validation_losses["val_loss"] += loss.item()
                self.validation_losses["val_loss_seg"] += loss_seg.item()
                self.validation_losses["val_loss_exist"] += loss_exist.item()

                progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
                progressbar.update(1)

        progressbar.close()
        self.log(self.validation_losses)

        print("------------------------\n")
        print("best val loss", self.best_val_loss)
        print("current val loss", self.validation_losses["val_loss"])
        if self.validation_losses["val_loss"] < self.best_val_loss:
            self.best_val_loss = self.validation_losses["val_loss"]
            copy_name = os.path.join(self.exp_dir, self.exp_name + "_best.pth")
            shutil.copyfile(self.snapshot_path, copy_name)

    def log(self, loss_dict):
        for l, v in loss_dict.items():
            self.writers.add_scalar("{}".format(l), v, self.step)

    def log_img(self, img_list, log_batch_idx):
        for i, img in enumerate(img_list):
            self.writers.add_image(
                "img_{}/{}".format(str(log_batch_idx), str(i)),
                img,
                self.epoch,
                dataformats="HWC",
            )

    def save_snapshot(self):
        save_dict = {
            "epoch": self.epoch,
            "net": self.net.module.state_dict()
            if isinstance(self.net, DDP)
            else self.net.state_dict(),
            "optim": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(save_dict, self.snapshot_path)
        print("model is saved: {}".format(self.snapshot_path))


def main(args):
    ddp_setup()
    trainer = Trainer(args.exp_dir)
    trainer.train()
    destroy_process_group


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    main(args)

    trainer = Trainer(args.exp_dir)
    trainer.train()
