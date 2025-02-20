import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from torchvision.models import resnet18
from timm.models.vision_transformer import Block
from scipy.optimize import linear_sum_assignment

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        x = x.reshape((x.shape[0],) + self.shape)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) * 2 / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.shape[1], :]

class Model(nn.Module):
    def __init__(self,
                patch_size,
                patch_num,
                embed_dim = 384,
                tdep = 6,
                nhead = 8,
                mlp_ratio = 4.,
                norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_size = patch_size
        self.patch_num = patch_num

        self.embed_dim = embed_dim
        # self.embedding = nn.Sequential(
        #     nn.Conv2d(3, embed_dim, patch_size, patch_size),
        #     Reshape((embed_dim,)),
        #     # nn.ReLU(),
        #     # nn.Linear(embed_dim, embed_dim),
        # )
        self.embedding = nn.Sequential(*[
            resnet18(), 
            nn.Linear(1000, embed_dim)
        ])
        # self.positional_encoding = PositionalEncoding(embed_dim, patch_num * patch_num)

        # self.cls_token = nn.Parameter(torch.zeros(embed_dim))
        self.transformer = nn.ModuleList([
            Block(embed_dim, nhead, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) 
            for _ in range(tdep)])
        
        self.norm = norm_layer(embed_dim)
        self.jigsaw = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.patch_num * self.patch_num)
        )
        
        self.loss = nn.CrossEntropyLoss()
        self.initialize_weights()

    def print_gradient(self):
        for i, module in enumerate(self.embedding):
            print(f"Layer {i}:")
            if(i == 0):
                for j, mm in enumerate(module.parameters()):
                    if hasattr(mm, 'weight'):
                        print(f"Weight gradient: {mm.weight.grad}")
                    elif hasattr(mm, 'bias'):
                        print(f"Bias mm: {mm.bias.grad}")
            else:
                if hasattr(module, 'weight'):
                    print(f"Weight gradient: {module.weight.grad}")
                elif hasattr(module, 'bias'):
                    print(f"Bias gradient: {module.bias.grad}")

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_jigsaw(self, x):
        # (B, L, C, H, W)
        device = next(self.parameters()).device
        x = x.to(device).to(torch.float32)
        B, L, C, H, W = x.shape
        # embed patches
        x = x.view(B * L, C, H, W)
        x = self.embedding(x)
        x = x.view(B, L, self.embed_dim)
        # (B, L, embed_dim)

        # append cls token
        # cls_tokens = self.cls_token.view(1, 1, -1).repeat(B, 1, 1)
        # x = torch.cat((cls_tokens, x), dim=1)
        #print(x.shape)
        # (B, L, embed_dim + 1)

        # apply Transformer blocks
        for blk in self.transformer:
            x = blk(x)
        
        x = self.norm(x)
        # x = x[:, 1:]
        x = self.jigsaw(x)
        # (B, n * n)
        return x

    def predict(self, imgs):
        device = next(self.parameters()).device
        x = self.forward_jigsaw(imgs)
        return torch.max(x, dim=2)[1]

    def predict_hungarian(self, imgs):
        device = next(self.parameters()).device
        x = self.forward_jigsaw(imgs).detach().cpu().numpy()
        perm = torch.zeros((x.shape[0], self.patch_num * self.patch_num), dtype=int)
        for index in range(x.shape[0]):
            row_ind, col_ind = linear_sum_assignment(x[index], maximize = True)
            for i in range(0, self.patch_num * self.patch_num):
                perm[index][row_ind[i]] = col_ind[i]
        perm = perm.to(device)
        return perm

    def get_loss(self, imgs, perms):
        device = next(self.parameters()).device
        perms = perms.to(device)

        res = self.forward_jigsaw(imgs)
        res = res.reshape(-1, self.patch_num * self.patch_num)
        perms = perms.reshape(-1)
        return self.loss(res, perms)


if __name__ == '__main__':
    p, n = PARAMS["patch_size"], PARAMS["patch_num"]
    
    net = Model(patch_size=p,
        patch_num=n,
        embed_dim=384,
        tdep=12,
        nhead=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    
    net = net.cuda()
    img = torch.cuda.FloatTensor(1, n * n, 
            3, p, p)
    perm = torch.randperm(n).unsqueeze(0).to("cuda")

    mask_ratio = 0.75
    with torch.no_grad():
        loss = net.forward_jigsaw(img)