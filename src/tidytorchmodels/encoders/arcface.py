import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..detectors.coord_reg import Landmarker_CoordReg
from ..detectors.mtcnn import Landmarker_MTCNN
from .operations.align import face_align
from ..utils.weights import load_weights

# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py
# https://arxiv.org/abs/2004.04989


class IRNBlock(nn.Module):

    def __init__(self, cin, cout, stride=1):
        super(IRNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            ConvUnit(cin, cout, k=3, s=1, p=1, activ='prelu'),
            ConvUnit(cout, cout, k=3, s=stride, p=1, activ=None)
        )
        if stride > 1:
            self.downsample = ConvUnit(cin, cout, k=1, s=stride, p=0, activ=None)

    def forward(self, x):
        y = x if not hasattr(self, 'downsample') else self.downsample(x)
        x = self.block(x)
        return x + y


class IResNet(nn.Module):

    def __init__(self, device, layers):
        super(IResNet, self).__init__()
        self.main = nn.Sequential(
            ConvUnit(3, 64, k=3, s=1, p=1, activ='prelu'),
            IRNBlock(64, 64, 2),
            nn.Sequential(*[IRNBlock(64, 64) for i in range(1, layers[0])]),
            IRNBlock(64, 128, 2),
            nn.Sequential(*[IRNBlock(128, 128) for i in range(1, layers[1])]),
            IRNBlock(128, 256, 2),
            nn.Sequential(*[IRNBlock(256, 256) for i in range(1, layers[2])]),
            IRNBlock(256, 512, 2),
            nn.Sequential(*[IRNBlock(512, 512) for i in range(1, layers[3])]),
            nn.BatchNorm2d(512),
            nn.Flatten(1),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512)
        )
        self.to(device)

    def forward(self, x):
        x = self.main(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ArcFaceEncoder():

    architecture = {
        'IResNet18': [2, 2, 2, 2],
        'IResNet34': [3, 4, 6, 3],
        'IResNet50': [3, 4, 14, 3],
        'IResNet100': [3, 13, 30, 3],
        'IResNet200': [6, 26, 60, 6]
    }

    links = {
        # https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo
        #'r18': '1-fmCdluzOGEMUFBQeIv_HJiVwAuosrZF',  # glint360k_cosface_r18_fp16_0.1_backbone.pth
        #'r34': '16sPIRJGgof6WoERFqMVg9llID6mSMoym',  # glint360k_cosface_r34_fp16_0.1_backbone.pth
        'r50': '1UYIZkHTklpFMGLhFGzzCvUZS-rMLY4Xv',  # glint360k_cosface_r50_fp16_0.1_backbone.pth
        #'r100': '1nwZhK33-5KwE8nyKWlBr8zDm0Tx3PYQ9', # glint360k_cosface_r100_fp16_0.1_backbone.pth
        # https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#5-pretrain-models
        #'r50_pfc': '1eBTfk0Ozsx0hF0l1z06mlzC6mOLxTscK', # partial_fc_glint360k_r50_insightface.pth
        #'r100_pfc': '1XNMRpB0MydK1stiljHoe4vKz4WfNCAIG' # partial_fc_glint360k_r100_insightface.pth
    }

    def __init__(self, src, device=None, align=True, landmarker='mobilenet', tform='similarity'):
        print('Initializing ArcFace %s model for face feature extraction' % src.upper())
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        if align:
            self.tform = tform
            if landmarker == 'mobilenet':
                self.landmarker = Landmarker_CoordReg()
            else:
                self.landmarker = Landmarker_MTCNN()
        resnet_number = src.split('_')[0][1:]
        self.model = IResNet(dv, self.architecture['IResNet' + resnet_number])
        load_weights(self.model, self.links[src], 'arcface_' + src)
        self.model.eval()

    def __call__(self, images):
        if hasattr(self, 'landmarker'):
            images = [cv2.resize(img, (192, 192)) for img in images]
            lm = self.landmarker(images)
            images = face_align(images, lm, self.tform)
        inp = cv2.dnn.blobFromImages(images, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp).to(next(self.model.parameters()).device)
        with torch.no_grad():
            out = self.model(inp)
        return out.cpu().numpy()


# ==================== EXTRA ====================

class ONNXIResNetEncoder():
    # !pip install onnxruntime
    # !pip install onnxruntime-gpu

    # https://github.com/deepinsight/insightface/tree/master/model_zoo#1-face-recognition-models
    
    def __init__(self, device):
        import onnxruntime
        from ..utils.weights import prep_file
        #modelfile = prep_file('1AhyD9Zjwy5MZgJIjj2Pb-YfIdeFS3T5E', 'w600k_r50.onnx', gdrive=True)
        modelfile = prep_file('1MpRhM76OQ6cTzpr2ZSpHp2_CP19Er4PI', 'glint360k_r50.onnx', gdrive=True)
        self.landmarker = Landmarker_CoordReg(device)
        #self.session = onnxruntime.InferenceSession(modelfile, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.session = onnxruntime.InferenceSession(modelfile, providers=['CPUExecutionProvider'])
        self.inpname = self.session.get_inputs()[0].name

    def __call__(self, images):
        import numpy as np
        images = [cv2.resize(img, (192, 192)) for img in images]
        lm = self.landmarker(images)
        images = face_align(images, lm, 'similarity')
        images = cv2.dnn.blobFromImages(images, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        x = self.session.run(None, {self.inpname: images})[0]
        x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
        return x