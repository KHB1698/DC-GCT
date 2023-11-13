import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops.einops import rearrange

from model.block.gcn_conv import Gcn_block
from model.block.graph_frames import Graph
from model.block.transformer import Attention, Mlp


class Local(nn.Module):
    def __init__(self, dim,h_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.graph = Graph('hm36_gt', 'spatial', pad=0)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda()  
        kernel_size = self.A.size(0) 
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gcn1 = Gcn_block(dim, h_dim, kernel_size, residual=False)
        self.norm_gcn1 = norm_layer(dim)
        self.gcn2 =Gcn_block(h_dim, dim, kernel_size, residual=False)
        self.norm_gcn2 = norm_layer(dim)

    def forward(self, x):
        res = x
        x, A = self.gcn1(self.norm_gcn1(x),self.A)
        x, A = self.gcn2(x,self.A)
        x = res + self.drop_path(self.norm_gcn2(x))
        return x
    

class Global(nn.Module):
    def __init__(self, dim, num_heads,  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, length=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, length=length)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm_attn(x)))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,    norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        self.dim1 = dim1 = int(dim/5)
        self.dim2 = dim2 = dim - dim1
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.local1 = Local(dim1,dim1*2, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.global1 = Global(dim1, num_heads,  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                 drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)
        
        self.norm_fusion = norm_layer(dim)
        self.fusion = Mlp(in_features=dim, hidden_features=dim2, act_layer=act_layer, drop=drop)
        
        self.local2 = Local(dim2,dim2*2, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.global2 = Global(dim2, num_heads,  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                 drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*4, act_layer=act_layer, drop=drop)


    def forward(self, x):        
        x1,x2 = torch.split(x, [self.dim1,self.dim2], -1)
        
        x1 = self.local1(x1)
        x2 = self.global2(x2)
        
        x_fusion = torch.cat([x1,x2], -1)
        x_fusion_temp = x_fusion + self.fusion(self.norm_fusion(x_fusion))
        x_fusion_1, x_fusion_2 = torch.split(x_fusion_temp, [self.dim1,self.dim2], -1)
        
        
        x1 = self.global1(x1 + x_fusion_1)
        x2 = self.local2(x2 + x_fusion_2)
        
        x = torch.cat([x1,x2], -1) + x_fusion
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x


class DC_GCT(nn.Module):
    def __init__(self, args, depth=3, embed_dim=160, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=9):
        super().__init__()
        
        depth, embed_dim, mlp_hidden_dim, length  = args.layers, args.channel, args.d_hid, args.frames
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints
        
        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        
        self.patch_embed = nn.Linear(2, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, embed_dim))
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
            for i in range(depth)])
        self.Temporal_norm = norm_layer(embed_dim)
        
        self.fcn = nn.Linear(embed_dim, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        x = self.fcn(x)
        x = x.view(x.shape[0], -1, self.num_joints_out, x.shape[2])
        return x

