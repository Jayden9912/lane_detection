import argparse

class LDoptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SegFormer LaneDetection Options")

        # path config
        self.parser.add_argument("--exp_dir", type=str, help="path of the experiment folder")

        # reproducibility config
        self.parser.add_argument("--seed", type=int, default=0)
        # training config
        self.parser.add_argument("--resume", "-r", action="store_true")
        self.parser.add_argument("--dataset_name", "-dn", type=str, default="Tusimple", choices=["Tusimple", "CULane"])
        self.parser.add_argument("--device_id", type=int, default=0)
        self.parser.add_argument("--max_epochs", type=int, default=30)
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--resize_shape", nargs="+", type=int, default=[512, 288])
        self.parser.add_argument("--flip_prob", type=float, default=0.0)

        # optimization config
        self.parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
        self.parser.add_argument("--weight_decay", type=float, default=0.01)

        # MixVisionTransformer config
        self.parser.add_argument("--pooling_method", type=str, default="avg", choices=["avg","max"])
        self.parser.add_argument("--embed_dims", nargs="+", type=int, default=[32, 64, 160, 256])
        self.parser.add_argument("--num_heads", nargs="+", type=int, default=[1, 2, 5, 8])
        self.parser.add_argument("--mlp_ratios", nargs="+", type=int, default=[4, 4, 4, 4])
        self.parser.add_argument("--without_qkv_bias", action="store_false")
        self.parser.add_argument("--encoder_depths", nargs="+", default=[2, 2, 2, 2])
        self.parser.add_argument("--sr_ratio", nargs="+", type=int, default=[8, 4, 2, 1])
        self.parser.add_argument("--drop_rate", type=float, default=0.0, help="dropout rate used for MLP, product of attention and values vector")
        self.parser.add_argument("--attention_drop_rate", type=float, default=0.0)
        self.parser.add_argument("--drop_path_rate", type=int, default=0.1)
        self.parser.add_argument("--se_layer", action="store_true")
    
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options




