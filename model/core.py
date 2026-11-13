import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encode import DualStreamNetwork
from .Decode import Decoder
from .Fuse import FusionMLLABlock

class Model(nn.Module):

    def __init__(self, num_classes=6, full_in_chans=3, light_in_chans=1, embed_dim=64, model_use='all'):
        super().__init__()
        self.model_use = model_use
        self._cache = {'encoder_back': None, 'decoder_all': None}
        self.fuse = nn.ModuleList([FusionMLLABlock(dim=embed_dim * 4, input_resolution=(512 // 16, 512 // 16), num_heads=8, drop_rate=0.0, drop_path=0.2), FusionMLLABlock(dim=embed_dim * 2, input_resolution=(512 // 8, 512 // 8), num_heads=4, drop_rate=0.0, drop_path=0.2), FusionMLLABlock(dim=embed_dim, input_resolution=(512 // 4, 512 // 4), num_heads=2, drop_rate=0.0, drop_path=0.2)])
        self.encode = DualStreamNetwork(pretrained_path=None, img_size=512, full_in_chans=full_in_chans, light_in_chans=light_in_chans, drop_rate=0.0, drop_path_rate=0.2, model_use=model_use, fuselayer=self.fuse[0])
        self.decode = Decoder(num_classes=num_classes, drop_rate=0.0, drop_path_rate=0.2, model_use=model_use, fuselist=self.fuse[1:])

    def forward(self, rgb, dsm):
        rgb_features, dsm_features, last_features = self.encode(rgb, dsm)
        return self.decode(rgb_features, dsm_features, last_features)
