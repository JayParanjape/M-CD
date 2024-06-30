import torch
import torch.nn as nn
import sys
sys.path.append('../..')
sys.path.append('..')
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from ..net_utils import FeatureFusionModule as FFM
# from ..net_utils import FeatureRectifyModule as FRM
import math
import time
from engine.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock
from deepspeed.profiling.flops_profiler import FlopsProfiler

logger = get_logger()


class RGBXTransformer(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()
        
        self.ape = ape

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )
        
        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        
        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_A, x_B):
        """
        x_A: B x C x H x W
        """
        B = x_A.shape[0]
        outs_fused = []
        
        outs_A = self.vssm(x_A) # B x C x H x W
        outs_B = self.vssm(x_B) # B x C x H x W
        
        for i in range(4):
            if self.ape:
                # this has been discarded
                out_A = self.absolute_pos_embed[i].to(outs_A[i].device) + outs_A[i]
                out_B = self.absolute_pos_embed_x[i].to(outs_B[i].device) + outs_B[i]
            else:
                out_A = outs_A[i]
                out_B = outs_B[i]
            
            x_fuse = self.channel_attn_mamba[i](out_A.permute(0, 2, 3, 1).contiguous(), out_B.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            # x_fuse = (out_A - out_B)
            outs_fused.append(x_fuse)        
        return outs_fused

    def forward(self, x_A, x_B):
        out = self.forward_features(x_A, x_B)
        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2], 
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )
        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))


class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))

if __name__ == "__main__":
    print("model small")
    model_small = vssm_small().cuda()
    prof = FlopsProfiler(model_small, None)

    dummy1 = torch.ones((1,3,256,256)).cuda()
    dummy2 = torch.ones((1,3,256,256)).cuda()
    prof.start_profile()
    dummy_out = model_small(dummy1, dummy2)
    prof.stop_profile()
    flops = prof.get_total_flops()
    print("Small GFlops: ", flops/(10**9))
    prof.end_profile()
