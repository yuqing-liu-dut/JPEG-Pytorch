# Pytorch
import torch
import torch.nn as nn
# Local
from modules import compress_jpeg, decompress_jpeg
from utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered

    def set_quality(self, quality):
        factor = quality_to_factor(quality)
        self.compress.factor = factor
        self.decompress.factor = factor


if __name__ == '__main__':
    with torch.no_grad():
        import cv2
        import numpy as np

        img = cv2.imread("Lena.png")
        B, G, R = img[..., 0], img[..., 1], img[..., 2]
        img[..., 0], img[..., 1], img[..., 2] = R, G, B

        inputs = np.transpose(img, (2, 0, 1))
        inputs = inputs[np.newaxis, ...]

        tensor = torch.FloatTensor(inputs).cuda()
        jpeg = DiffJPEG(512, 512, differentiable=True).cuda()

        quality = 80
        jpeg.set_quality(quality)

        outputs = jpeg(tensor)
        outputs = outputs.detach().cpu().numpy()
        outputs = np.transpose(outputs[0], (1, 2, 0))

        cv2.imshow("QF:"+str(quality), outputs / 255.)
        cv2.waitKey()

        from skimage.metrics import peak_signal_noise_ratio as PSNR
        print(PSNR(np.uint8(outputs), np.uint8(img)))
