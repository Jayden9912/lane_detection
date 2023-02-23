import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class combined_dataloader(Dataset):
    def __init__(self, file_path, image_set, dataset_name, transforms=None):
        super(combined_dataloader, self).__init__()
        assert image_set in ("train", "val", "test"), "image_set is not valid!"
        assert dataset_name in (
            "TuSimple",
            "CuLane",
            "combined",
        ), "dataset_name is not valid!"
        self.transforms = transforms
        self.file_path = file_path
        self.image_set = image_set
        self.dataset_name = dataset_name
        if image_set == "test" and dataset_name == "CuLane":
            self.createIndex_test()
        else:
            self.createIndex()

    def createIndex(self):
        self.img_list = []
        self.segLabel_list = []
        self.exist_list = []
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(l[0])
                self.segLabel_list.append(l[1])
                self.exist_list.append([int(x) for x in l[2:]])

    def createIndex_test(self):
        self.img_list = []
        with open(self.file_path) as f:
            for line in f:
                line = line.strip()
                self.img_list.append(line)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_set != "test" and self.dataset_name != "CuLane":
            segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            exist = np.array(self.exist_list[idx])
        else:
            segLabel = None
            exist = None

        sample = {
            "img": img,
            "segLabel": segLabel,
            "exist": exist,
            "img_name": self.img_list[idx],
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]["img"], torch.Tensor):
            img = torch.stack([b["img"] for b in batch])
        else:
            img = [b["img"] for b in batch]

        if batch[0]["segLabel"] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]["segLabel"], torch.Tensor):
            segLabel = torch.stack([b["segLabel"] for b in batch])
            exist = torch.stack([b["exist"] for b in batch])
        else:
            segLabel = [b["segLabel"] for b in batch]
            exist = [b["exist"] for b in batch]

        samples = {
            "img": img,
            "segLabel": segLabel,
            "exist": exist,
            "img_name": [x["img_name"] for x in batch],
        }

        return samples
