class LDoptions:
    def __init__(self):
        self.pooling_method = "avg"
        self.embed_dims = [32, 64, 160, 256]
        self.num_heads = [1, 2, 5, 8]
        self.mlp_ratios = [4, 4, 4, 4]
        self.without_qkv_bias = True
        self.encoder_depths = [2, 2, 2, 2]
        self.sr_ratio = [2, 2, 1, 1]
        self.attention_drop_rate = 0.0
        self.drop_path_rate = 0.1
        self.se_layer = True