
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import timm.models.vision_transformer

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)

class HashLayer(nn.Module):

    def threshold_function(self, tensor):

        return torch.where(tensor >= 0, torch.tensor(1).to(tensor), torch.tensor(-1).to(tensor))

    def forward(self, x):
        return self.threshold_function(x)

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.embeddings = L2Norm()
        self.hash_layer = HashLayer()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        outcome = x[:, 0]

        return outcome
        
    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x) #FC
        x = self.embeddings(x)
        x = self.hash_layer(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model