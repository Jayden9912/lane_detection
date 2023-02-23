import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from loss.focal_loss import FocalLoss
from mmseg.models.backbones import MixVisionTransformer
from mmseg.models.decode_heads import SegFormerHead


class segformer(nn.Module):
    def __init__(self, model_config, dataset_name, pretrained=True):
        super(segformer, self).__init__()
        self.pretrained = pretrained
        self.dataset_name = dataset_name
        # pretrained_model = model_config.pop("pretrained_model")
        self.pooling_method = model_config.pop("pooling_method")
        self.dropout = model_config.pop("dropout")

        self.encoder, self.decoder, self.pooling_layer, self.fc = self.initialize_model()

        if self.pretrained:
            self.load_pretrained()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()
        # testing focal loss
        # self.alpha = torch.tensor([0.4, 1, 1, 1, 1])
        # self.gamma = 2
        # self.focal_loss = FocalLoss(self.alpha, self.gamma)

    def forward(self, img, seg_gt=None, exist_gt=None):
        # encode decode
        x = self.encoder(img)
        x = self.decoder(x)

        # get the segmentation map
        seg_pred = (F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)).contiguous()

        # get the probabilities of the existence of a lane predicted
        x = self.pooling_layer(x)
        B, C, H, W = x.shape
        x = x.view(-1, C*H*W) # TuSimple mit_b0: (-1, 11520), CULane mit_b0: (-1, 18000)
        exist_pred = self.fc(x)

        # calculate losses
        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss

    def load_pretrained(self):
        encoder_weight = torch.load("/home/automan/wuguanjie/SegFormer/pretrained/mit_b0.pth")
        encoder_weight.pop("head.weight")
        encoder_weight.pop("head.bias")
        self.encoder.load_state_dict(encoder_weight)
    
    def initialize_model(self):
        norm_cfg = dict(type="BN", requires_grad=True)
        encoder = MixVisionTransformer(patch_size=4,
                                        embed_dims=[32, 64, 160, 256],
                                        num_heads=[1, 2, 5, 8],
                                        mlp_ratios=[4, 4, 4, 4],
                                        qkv_bias=True,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                        depths=[2, 2, 2, 2],
                                        sr_ratios=[8, 4, 2, 1],
                                        drop_rate=0.0,
                                        drop_path_rate=0.1)
        decoder = SegFormerHead(in_channels=[32, 64, 160, 256],
                                in_index=[0, 1, 2, 3],
                                feature_strides=[4, 8, 16, 32],
                                channels=128,
                                dropout_ratio=0.1,
                                num_classes=5,
                                norm_cfg=norm_cfg,
                                align_corners=False,
                                decoder_params=dict(embed_dim=256),
                                loss_decode=dict(
                                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                                ),)

        if self.dataset_name == "Tusimple":
            fc = nn.Sequential(nn.Linear(11520, 1280),
                                nn.LeakyReLU(),
                                nn.Linear(1280, 128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 4),
                                nn.Sigmoid(),
                                )  # TuSimple mit_b0
        else: #CULane
            fc = nn.Sequential(nn.Linear(18000, 1280),
                                nn.LeakyReLU(),
                                nn.Linear(1280, 128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 4),
                                nn.Sigmoid(),
                                )  # CuLane mit_b0

        if self.pooling_method == "avg":
            pooling_layer = nn.Sequential(nn.Softmax(dim=1),  # (nB, 5, 72, 128)
                                          nn.AvgPool2d(2, 2),  # (nB, 5, 36, 64)
                                          )
        elif self.pooling_method == "max":
            pooling_layer = nn.Sequential(nn.Softmax(dim=1),  # (nB, 5, 72, 128)
                                          nn.MaxPool2d(2, 2),  # (nB, 5, 36, 64)
                                          )
        return encoder, decoder, pooling_layer, fc
