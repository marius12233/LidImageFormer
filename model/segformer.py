import torch
import torch.nn as nn
import torch.nn.functional as F
from segformer_encoder import mit_b5
from segformer_decoder import SegFormerHead
import cv2
from transformers import SegformerFeatureExtractor

class SegFormer(nn.Module):

    def __init__(
        self,
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=None,
        align_corners=False,
        in_chans=3
        ):
        super().__init__()
        self.backbone = mit_b5(in_chans=in_chans)
        self.decode_head = SegFormerHead(
            in_channels=in_channels,
            in_index=in_index,
            feature_strides=feature_strides,
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=align_corners
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x


if __name__ == "__main__":
    import numpy as np
    def random_pixel():
        return np.array([np.random.randint(255), np.random.randint(255), np.random.randint(255)])
    
    LABEL_RGB_MAPPING = {i: random_pixel() for i in range(19)}
    N, C, H, W = 1, 3, 1024, 1024
    x = torch.rand([N, C, H, W])
    model = SegFormer(num_classes=19, channels=768)
    #efficient_attention = Block(768, 8, sr_ratio=2)
    
    PATH = "segformer.b5.1024x1024.city.160k.pth"
    checkpoint = torch.load(PATH, map_location="cpu")
    state_dict = checkpoint['state_dict']
    key1 = [k for k in state_dict.keys()]
    #print(key1)
    #print(state_dict[key1].shape)
    img = cv2.imread("distortion.png")
    h, w = img.shape[:2]
    print("w and h: ", w, h)
    #img = np.array(cv2.resize(img, (W, H)))

    #cv2.imshow("img", img)
    #cv2.waitKey(5)
    #cv2.destroyAllWindows()
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    inputs = feature_extractor(images=img, return_tensors="pt")

    #x = torch.from_numpy(img).float()#.cuda()
    #x = x.unsqueeze(0)
    #x = x.permute(0, 3, 2, 1)
    #print("Input shape: ", x.shape)


    model.load_state_dict(checkpoint['state_dict'])
    #model#.cuda()
    out = model(inputs['pixel_values'])
    out: np.ndarray = torch.argmax(out, dim=1).squeeze(0).numpy()
    print("Output shape: ", out.shape)
    #y = efficient_attention(x, h, w)
    out_image = np.zeros((H, W, 3))

    #out = np.reshape(out, (H, W, 1))

    for i in range(19):
        out_image[out == i] = LABEL_RGB_MAPPING[i]

    # for x in range(W):
    #     for y in range(H):
    #         key = out[y, x, 0]
    #         #print(key)
    #         rgb = LABEL_RGB_MAPPING[key]
    #         out_image[x, y, 0] = rgb[0]
    #         out_image[x, y, 1] = rgb[1]
    #         out_image[x, y, 2] = rgb[2]

    #out = np.array([out, out, out], dtype=float) / 255.
    print("Out image shape: ", out_image.shape)
    print("labels: ", np.unique(out))

    out_image = cv2.resize(out_image, (w, h))
    cv2.imshow("out", out_image / 255.)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #cv2.waitKey(0)
    #print(state_dict["decode_head.linear_fuse.conv.weight"])
    #print(model)

    #for m in model.named_modules()
    #print([m for m in model.modules()][0])
    #print(checkpoint["state_dict"].keys())
