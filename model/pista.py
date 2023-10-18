import torch
import torch.nn as nn
from segformer import SegFormer
from segformer_encoder import Attention

class CrossAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        

    def forward(self, cam_x, lidar_x, H, W):
        B, N, C = cam_x.shape
        q = self.q(cam_x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = lidar_x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(lidar_x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    
class Pista(nn.Module):

    def __init__(self,
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=512,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=None,
                align_corners=False,
        ):

        super().__init__()
        self.camera_stream = SegFormer(
            in_channels=in_channels,
            in_index=in_index,
            feature_strides=feature_strides,
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=align_corners)
        
        self.lidar_stream = SegFormer(
            in_channels=in_channels,
            in_index=in_index,
            feature_strides=feature_strides,
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=align_corners,
            in_chans=5)
        
        self.cross_attn_1 = CrossAttention(6_4)
        self.cross_attn_2 = CrossAttention(128)
        self.cross_attn_3 = CrossAttention(320)
        self.cross_attn_4 = CrossAttention(512)
        self.cross_attns = [
            self.cross_attn_1,
            self.cross_attn_2,
            self.cross_attn_3,
            self.cross_attn_4
        ]
        
    def forward_features(self, x, cam_feats):
        B = x.shape[0]
        outs = []

        x, H, W = self.lidar_stream.backbone.patch_embed1(x)
        for i, blk in enumerate(self.lidar_stream.backbone.block1):
            x = blk(x, H, W)
        x = self.lidar_stream.backbone.norm1(x)
        print("Shape cam_x; ", cam_feats[0].shape)
        print("H: ", H)
        print("W: ", W)
        cam_x = cam_feats[0].reshape(B, H*W, -1)
        print("After reshape -> Shape cam_x: ", cam_x.shape)
        print("Lidar x shape: ", x.shape)
        cam_x = self.cross_attn_1(cam_x, x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x += cam_feats[0]
        outs.append(x)

        # stage 2
        x, H, W = self.lidar_stream.backbone.patch_embed2(x)
        for i, blk in enumerate(self.lidar_stream.backbone.block2):
            x = blk(x, H, W)
        x = self.lidar_stream.backbone.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x += cam_feats[1]
        outs.append(x)

        # stage 3
        x, H, W = self.lidar_stream.backbone.patch_embed3(x)
        for i, blk in enumerate(self.lidar_stream.backbone.block3):
            x = blk(x, H, W)
        x = self.lidar_stream.backbone.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x += cam_feats[2]
        outs.append(x)

        # stage 4
        x, H, W = self.lidar_stream.backbone.patch_embed4(x)
        for i, blk in enumerate(self.lidar_stream.backbone.block4):
            x = blk(x, H, W)
        x = self.lidar_stream.backbone.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x += cam_feats[3]
        outs.append(x)

        return outs
    
    def forward(self, x):
        cam_x = x[0]
        lidar_x = x[1]
        cam_feats = self.camera_stream.backbone(cam_x)
        #lidar_feats = self.forward_features(lidar_x, cam_feats)
        lidar_feats = self.lidar_stream.backbone(lidar_x)
        B = 1 # TODO: Replace with the batch size

        #lidar_cam_feats = [cam_x + lidar_x for cam_x, lidar_x in zip(cam_feats, lidar_feats)]

        lidar_cam_feats = []
        for i, (cam_x, lidar_x) in enumerate(zip(cam_feats, lidar_feats)):
            B, C, H, W = cam_x.shape
            cross_attn_out = self.cross_attns[i](cam_x.reshape(B, H*W, -1), lidar_x.reshape(B, H*W, -1), H, W)
            cross_attn_out = cross_attn_out.reshape(B, C, H, W)
            lidar_cam_feats.append(cross_attn_out + lidar_x)

        cam_out = self.camera_stream.decode_head(cam_feats)
        lidar_out = self.lidar_stream.decode_head(lidar_cam_feats)

        return [cam_out, lidar_out]

if __name__=="__main__":
    import time
    N, C, H, W = 1, 3, 224, 224
    x1 = torch.rand([N, C, H, W]).cuda()
    x2 = torch.rand([N, 5, H, W]).cuda()
    model = Pista(num_classes=19, channels=768)
    model.cuda()
    
    start = time.time()
    y = model([x1, x2])
    #y = y.detach().cpu()
    end = time.time()
    print("Len out: ", len(y))
    print("Time (s): ", end - start)
    

