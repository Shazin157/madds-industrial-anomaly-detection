import torch
import torch.nn as nn
from torchvision import models
import timm

N_PATCHES       = 196
LOCAL_FEAT_DIM  = 384
GLOBAL_FEAT_DIM = 768


class LocalStream(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(*list(backbone.features.children())[:6])
        for p in self.net.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x):
        f = self.net(x)                          # (B, 384, 14, 14)
        B, C, H, W = f.shape
        return f.permute(0,2,3,1).reshape(B, H*W, C)


class GlobalStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = timm.create_model(
            "vit_base_patch16_224.dino", pretrained=True, num_classes=0
        )
        for p in self.dino.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x):
        return self.dino.forward_features(x)[:, 1:, :]  # (B, 196, 768)


class IRBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.local   = LocalStream()
        self.global_ = GlobalStream()
        self.fused_dim = LOCAL_FEAT_DIM + GLOBAL_FEAT_DIM  # 1152

    @torch.inference_mode()
    def forward(self, x):
        l = self.local(x)    # (B, 196, 384)
        g = self.global_(x)  # (B, 196, 768)
        return {"fused": torch.cat([l,g], dim=-1), "local": l, "global": g}