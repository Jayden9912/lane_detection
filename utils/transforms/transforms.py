import cv2

import numpy as np
import torch
from torchvision.transforms import Normalize as Normalize_th
from torchvision import transforms
from PIL import Image


class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t

        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """

    def __init__(self, *transforms):
        self.transforms = [*transforms]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t


class Resize(CustomTransform):
    def __init__(self, size, dataset_name):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  # (W, H)
        self.dataset_name = dataset_name

    def __call__(self, sample):
        img = sample.get("img")
        segLabel = sample.get("segLabel", None)
        if self.dataset_name == "combined":
            output_size = (512, 288)
            if img.shape == (590, 1640, 3):  # CuLane image
                img = cv2.resize(img, (800, 288), interpolation=cv2.INTER_CUBIC)

                # random cropping
                img_size = img.shape
                width_diff = img_size[1] - output_size[0]
                x1 = np.random.randint(width_diff)
                x2 = x1 + output_size[0]
                img = img[:, x1:x2, :]
                if segLabel is not None:
                    segLabel = cv2.resize(
                        segLabel, (800, 288), interpolation=cv2.INTER_NEAREST
                    )
                    segLabel = segLabel[:, x1:x2]
            else:  # Tusimple image
                img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)
                if segLabel is not None:
                    segLabel = cv2.resize(
                        segLabel, output_size, interpolation=cv2.INTER_NEAREST
                    )
        else:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
            if segLabel is not None:
                segLabel = cv2.resize(
                    segLabel, self.size, interpolation=cv2.INTER_NEAREST
                )

        _sample = sample.copy()
        _sample["img"] = img
        _sample["segLabel"] = segLabel
        return _sample

    def reset_size(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size


class RandomResize(Resize):
    """
    Resize to (w, h), where w randomly samples from (minW, maxW) and h randomly samples from (minH, maxH)
    """

    def __init__(self, minW, maxW, minH=None, maxH=None, batch=False):
        if minH is None or maxH is None:
            minH, maxH = minW, maxW
        super(RandomResize, self).__init__((minW, minH))
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH
        self.batch = batch

    def random_set_size(self):
        w = np.random.randint(self.minW, self.maxW + 1)
        h = np.random.randint(self.minH, self.maxH + 1)
        self.reset_size((w, h))


class Rotation(CustomTransform):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, sample):
        img = sample.get("img")
        segLabel = sample.get("segLabel", None)

        u = np.random.uniform()
        degree = (u - 0.5) * self.theta
        R = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), degree, 1)
        img = cv2.warpAffine(
            img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR
        )
        if segLabel is not None:
            segLabel = cv2.warpAffine(
                segLabel,
                R,
                (segLabel.shape[1], segLabel.shape[0]),
                flags=cv2.INTER_NEAREST,
            )

        _sample = sample.copy()
        _sample["img"] = img
        _sample["segLabel"] = segLabel
        return _sample

    def reset_theta(self, theta):
        self.theta = theta


class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.transform = Normalize_th(mean, std)

    def __call__(self, sample):
        img = sample.get("img")

        img = self.transform(img)

        _sample = sample.copy()
        _sample["img"] = img
        return _sample


class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float):
        self.dtype = dtype

    def __call__(self, sample):
        img = sample.get("img")
        segLabel = sample.get("segLabel", None)
        exist = sample.get("exist", None)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype) / 255.0
        if segLabel is not None:
            segLabel = torch.from_numpy(segLabel).type(torch.long)
        if exist is not None:
            exist = torch.from_numpy(exist).type(
                torch.float32
            )  # BCEloss requires float tensor

        _sample = sample.copy()
        _sample["img"] = img
        _sample["segLabel"] = segLabel
        _sample["exist"] = exist
        return _sample


class RHFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def forward(self, img, gt_mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return cv2.flip(img, 1), cv2.flip(gt_mask, 1)
        return img, gt_mask


class ColorJitter(CustomTransform):
    """
    Apply color jitter on the image
    """

    def __init__(
        self,
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=(-0.1, 0.1),
    ):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, sample):
        img = sample.get("img")

        # convert image from cv2 to pil
        img_cv2_to_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_cv2_to_pil)
        segLabel = sample.get("segLabel", None)

        # apply color jitter on the image
        aug_img = self.color_jitter(pil_image)

        # convert pil image back to cv2
        np_img = np.array(aug_img)
        cv2_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        _sample = sample.copy()
        _sample["img"] = cv2_img
        return _sample
