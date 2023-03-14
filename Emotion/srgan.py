# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class SRCNNModel(nn.Module):
    """
    SRCNN Model use for increase size of image 48 -> 384
    """
    def __init__(self):
        super(SRCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, 5, padding=2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def pred_SRCNN(model, image_path, device, scale_factor=8):
    """A function to load pretrained SRCNN to convert image (48px -> 384px).

    Args:
        model: SRCNN model.
        image_path: path of low resolution image PILLOW image.
        scale_factor: scale factor for resolution.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Couples of output image, and output image after interpolation
    """

    model.to(device)
    model.eval()
    image = np.array(Image.open(image_path).convert('RGB'))    # open image as nparray
    image = Image.fromarray(image)

    # split channels
    y, cb, cr = image.convert("YCbCr").split()

    # size will be used in image transform
    original_size = y.size

    # bicubic interpolate it to the original size
    y_bicubic = transforms.Resize(
        (original_size[1] * scale_factor, original_size[0] * scale_factor),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )(y)
    cb_bicubic = transforms.Resize(
        (original_size[1] * scale_factor, original_size[0] * scale_factor),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )(cb)
    cr_bicubic = transforms.Resize(
        (original_size[1] * scale_factor, original_size[0] * scale_factor),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )(cr)
    # turn it into tensor and add batch dimension
    y_bicubic = transforms.ToTensor()(y_bicubic).to(device).unsqueeze(0)

    # get the y channel SRCNN prediction
    y_pred = model(y_bicubic)

    # convert it to numpy image
    y_pred = y_pred[0].cpu().detach().numpy()

    # convert it into regular image pixel values
    y_pred = y_pred * 255
    y_pred.clip(0, 255)

    # conver y channel from array to PIL image format for merging
    y_pred_PIL = Image.fromarray(np.uint8(y_pred[0]), mode="L")

    # merge the SRCNN y channel with cb cr channels
    out_final = Image.merge("YCbCr", [y_pred_PIL, cb_bicubic, cr_bicubic]).convert(
        "RGB"
    )

    image_bicubic = transforms.Resize(
        (original_size[1] * scale_factor, original_size[0] * scale_factor),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )(image)

    return out_final, image_bicubic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRCNNModel().to(device)
model.load_state_dict(
    torch.load("SRCNNmodel_trained.pt", map_location=torch.device(device))
)
model.eval()

def super_reso(input_path, save_name):
    """Take input path on increase size it"""
    with torch.no_grad():
        out_final, image_bicubic = pred_SRCNN(
            model=model, image_path=input_path, device=device
        )
    image_bicubic.save(save_name)



