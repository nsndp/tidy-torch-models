import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..detectors.coord_reg import Landmarker_CoordReg
from ..detectors.mtcnn import Landmarker_MTCNN
from .operations.align import face_align
from ..utils.weights import load_weights

# combined from:
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/mobilefacenet.py
# https://github.com/foamliu/MobileFaceNet/blob/master/mobilefacenet.py
# https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch/blob/master/face_model.py

# MobileFaceNet paper: https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf
# MobileNetV2 paper:   https://arxiv.org/pdf/1801.04381.pdf


class MBlock(nn.Module):

    # aka bottleneck aka inverted residual block from MobileNetV2 paper
    def __init__(self, cin, cout, bn, relu, grp, s=1, residual=False):
        super(MBlock, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvUnit(cin, grp, 1, 1, 0, relu, bn),
            ConvUnit(grp, grp, 3, s, 1, relu, bn, grp=grp),
            ConvUnit(grp, cout, 1, 1, 0, None, bn)
        )

    def forward(self, x):
        y = self.layers(x)
        if not self.residual: return y
        return x + y


class MBlockRepeat(nn.Module):

    def __init__(self, cin, cout, bn, relu, grp1, grp2, rep):
        super(MBlockRepeat, self).__init__()
        self.first = MBlock(cin, cout, bn, relu, grp1, s=2)
        self.repeats = nn.Sequential(
            *[MBlock(cout, cout, bn, relu, grp2, residual=True) for _ in range(rep)]
        )

    def forward(self, x):
        return self.repeats(self.first(x))


class MobileFaceNet(nn.Module):
    
    def __init__(self, device, channels=[64, 64, 128], emb_size=512,
                 bn=1e-5, relu_type='prelu', diff_relu_f2=None,
                 intro_pw=False, outro_gd=True, outro_pw=128, outro_pw_relu=False,
                 outro_lin=None, lin_bias=True):
        super(MobileFaceNet, self).__init__()
        c0, c1, c2 = channels
        opwr = None if not outro_pw_relu else relu_type
        self.intro = nn.Sequential(
            ConvUnit(3, c0, 3, 2, 1, diff_relu_f2 or relu_type, bn),
            ConvUnit(c0, c0, 3, 1, 1, diff_relu_f2 or relu_type, bn, grp=64),
            ConvUnit(c0, c0, 1, 1, 0, relu_type, bn) if intro_pw else nn.Identity()
        )
        self.main = nn.Sequential(
            # effectively the same as bottlenecks from Table 1 from MobileFaceNet paper
            MBlockRepeat(c0, c1, bn, relu_type, 128, 128, rep=4),
            MBlockRepeat(c1, c2, bn, relu_type, 256, 256, rep=6),
            MBlockRepeat(c2, c2, bn, relu_type, 512, 256, rep=2))
        self.outro = nn.Sequential(
            ConvUnit(c2, 512, 1, 1, 0, relu_type, bn),
            ConvUnit(512, 512, 7, 1, 0, None, bn, grp=512) if outro_gd else nn.Identity(),
            ConvUnit(512, outro_pw, 1, 1, 0, opwr, bn, cbias_explicit=True) if outro_pw else nn.Identity(),
            nn.Flatten(),
            nn.Linear(outro_lin, emb_size, bias=lin_bias) if outro_lin else nn.Identity(),
            nn.BatchNorm1d(emb_size) if outro_lin else nn.Identity())
        self.to(device)

    def forward(self, x):   # [n, 3, 112, 112]
        x = self.intro(x)   # [n, c0, 56, 56]
        x = self.main(x)    # [n, c2, 7, 7]
        x = self.outro(x)   # [n, emb_size]
        x = F.normalize(x, p=2, dim=1) #equivalent: x = x / np.linalg.norm(X, axis=1).reshape(-1, 1)
        return x


class MobileFaceNetEncoder():

    links = {
        'w600k_insightface': '1fi8jBWJlZYjGNarvcQVPD9q449zA_sLo',
        'ms1m_foamliu': 'https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet.pt',
        'ms1m_xue24': 'https://raw.githubusercontent.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch/master/Weights/MobileFace_Net'
    }

    def fl_conversion(self, wd):
        wl = list(wd.items())
        wl.insert(12, wl.pop(7))
        for _ in range(19):
            wl.append(wl.pop(18))
        return dict(wl)

    def __init__(self, src='insightface', device=None, align=True, landmarker='mobilenet', tform='similarity'):
        print('Initializing MobileFaceNet model for face feature extraction')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        if align:
            self.landmarker = Landmarker_CoordReg(dv) if landmarker == 'mobilenet' else Landmarker_MTCNN(dv)
        self.tform = tform
        
        if src == 'w600k_insightface':
            self.model = MobileFaceNet(dv, channels=[128, 128, 256], bn=None,
                                       outro_gd=False, outro_pw=64, outro_pw_relu=True, outro_lin=3136)
        elif src == 'ms1m_foamliu':
            self.model = MobileFaceNet(dv, emb_size=128, relu_type='relu6', diff_relu_f2='relu', intro_pw=True)
        elif src == 'ms1m_xue24':
            self.model = MobileFaceNet(dv, outro_pw=None, outro_lin=512, lin_bias=False)

        mconv = None if src != 'ms1m_foamliu' else self.fl_conversion
        load_weights(self.model, self.links[src], 'mbfnet_' + src, mconv)
        self.model.eval()
    
    def __call__(self, images):
        if hasattr(self, 'landmarker'):
            images = [cv2.resize(img, (192, 192)) for img in images]
            lm = self.landmarker(images)
            images = face_align(images, lm, self.tform)
        inp = cv2.dnn.blobFromImages(images, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp).to(next(self.model.parameters()).device)
        with torch.no_grad():
            return self.model(inp).cpu().numpy()
            

def convert_weights_mbfnet_insightface():
    """Conversion code for Insightface's ONNX model's weights to pytorch. Needs "pip install onnx".
    Should create the same 'mbfnet_w600k_insightface.pt' file that's used above.
    For browsing onnx model: https://netron.app (mentioned https://github.com/onnx/onnx/issues/1425)
    """
    import shutil, os
    import os.path as osp
    from ..utils.weights import prep_file
    home = os.getcwd()
    dst = prep_file('19I-MZdctYKmVf3nu5Da3HS6KH5LBfdzG', 'buffalo_sc.zip')
    os.chdir(osp.dirname(dst))
    print('working at: ' + os.getcwd())
    shutil.unpack_archive('buffalo_sc.zip')
    os.rename('buffalo_sc/w600k_mbf.onnx', 'w600k_mbf.onnx')
    os.remove('buffalo_sc/det_500m.onnx')
    os.remove('buffalo_sc.zip')
    os.rmdir('buffalo_sc')
    print('prepared w600k_mbf.onnx')
        
    import onnx, onnx.numpy_helper
    onnx_model = onnx.load('w600k_mbf.onnx')
    source = dict([(raw.name, onnx.numpy_helper.to_array(raw)) for raw in onnx_model.graph.initializer])
    #for s in SO: print(s, '\t', SO[s].shape)
    match = [
        518, 519, 664, 521, 522, 665,
    
        524, 525, 666, 527, 528, 667, 530, 531,
        533, 534, 668, 536, 537, 669, 539, 540,
        542, 543, 670, 545, 546, 671, 548, 549,
        551, 552, 672, 554, 555, 673, 557, 558,
        560, 561, 674, 563, 564, 675, 566, 567,

        569, 570, 676, 572, 573, 677, 575, 576,
        578, 579, 678, 581, 582, 679, 584, 585,
        587, 588, 680, 590, 591, 681, 593, 594,
        596, 597, 682, 599, 600, 683, 602, 603,
        605, 606, 684, 608, 609, 685, 611, 612,
        614, 615, 686, 617, 618, 687, 620, 621,
        623, 624, 688, 626, 627, 689, 629, 630,

        632, 633, 690, 635, 636, 691, 638, 639,
        641, 642, 692, 644, 645, 693, 647, 648,
        650, 651, 694, 653, 654, 695, 656, 657,

        659, 660, 696, 662, 663, 697,
        'fc.weight', 'fc.bias', 'features.weight', 'features.bias', 'features.running_mean', 'features.running_var'
    ]
    model = MobileFaceNet('cpu', [128, 128, 256], bn=None,
                          outro_gd=False, outro_pw=64, outro_pw_relu=True, outro_lin=3136)
    dst = model.state_dict()
    nbt_name = list(dst)[-1]
    nbt = dst.pop(nbt_name)
    for i, s in enumerate(dst):
        val = source[str(match[i])]
        if dst[s].dim() == 1:
            val = val.squeeze()
        dst[s] = torch.Tensor(val.copy())
    dst[nbt_name] = torch.Tensor(nbt)
    model.load_state_dict(dst)
    model.eval()
    torch.save(model.state_dict(), 'mobilefacenet_w600k_insightface.pt')
    print('saved mobilefacenet_w600k_insightface.pt')
    os.remove('w600k_mbf.onnx')
    print('removed w600k_mbf.onnx')
    os.chdir(home)
    print('returned to: ' + home)