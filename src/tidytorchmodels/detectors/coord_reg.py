import cv2
import torch
import torch.nn as nn

from ..backbones.basic import ConvUnit
from ..backbones.mobilenet import MobileNetV1

from ..utils.weights import load_weights


# https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg        
# https://github.com/deepinsight/insightface/tree/master/model_zoo#21-2d-face-alignment
# https://github.com/nttstar/insightface-resources/blob/master/alignment/images/2d106markup.jpg


class MobileNetV1_Head212(nn.Module):

    def __init__(self, device):
        super(MobileNetV1_Head212, self).__init__()
        self.body = MobileNetV1(0.5, activ='prelu', bn=1e-03)
        self.head = nn.Sequential(
            ConvUnit(512, 64, 3, 2, 1, 'prelu', 1e-03),
            nn.Flatten(),
            nn.Linear(64*3*3, 212)
        )
        self.to(device)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


class Landmarker_CoordReg():

    def __init__(self, device=None):
        print('Initializing coordinate regression model for landmark detection')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = MobileNetV1_Head212(dv)
        load_weights(self.model, '1H1-KkskDrQvQ_J_bFLxZeie6YvTfB7KE', 'coordreg_2d106det_insightface')
        self.model.eval()
    
    def __call__(self, images):
        inp = cv2.dnn.blobFromImages(images, 1 / 128, (192, 192), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp).to(next(self.model.parameters()).device)
        with torch.no_grad():
            lm = self.model(inp).cpu().numpy()
        lm = lm.reshape(-1, 106, 2)
        lm = lm[:, [38, 88, 86, 52, 61], :]
        lm = (lm + 1) * (192 // 2)
        lm = lm.round().astype(int)
        return lm


def convert_weights_2d106det_insightface():
    """Conversion code for Insightface's ONNX model's weights to pytorch. Needs "pip install onnx".
    For browsing onnx model: https://netron.app (mentioned https://github.com/onnx/onnx/issues/1425)
    """
    import shutil, os
    import os.path as osp
    from ..utils.weights import prep_file
    home = os.getcwd()
    dst = prep_file('1M5685m-bKnMCt0u2myJoEK5gUY3TDt_1', '2d106det.onnx')
    os.chdir(osp.dirname(dst))
    print('working at: ' + os.getcwd())

    import onnx, onnx.numpy_helper
    onnx_model = onnx.load('2d106det.onnx')
    src = dict([(raw.name, onnx.numpy_helper.to_array(raw)) for raw in onnx_model.graph.initializer])
    
    match = []
    layers = ['conv_1', 'conv_2_dw', 'conv_2', 'conv_3_dw', 'conv_3', 'conv_4_dw', 'conv_4',
        'conv_5_dw', 'conv_5', 'conv_6_dw', 'conv_6', 'conv_7_dw', 'conv_7', 'conv_8_dw', 'conv_8',
        'conv_9_dw', 'conv_9', 'conv_10_dw', 'conv_10', 'conv_11_dw', 'conv_11', 'conv_12_dw',
        'conv_12','conv_13_dw', 'conv_13', 'conv_14_dw', 'conv_14', 'conv_15']
    for l in layers:
        match.append(l + '_conv2d_weight')
        match.append(l + '_batchnorm_gamma')
        match.append(l + '_batchnorm_beta')
        match.append(l + '_batchnorm_moving_mean')
        match.append(l + '_batchnorm_moving_var')
        match.append(l + '_empty') # placeholder
        match.append(l + '_relu_gamma')
    match.append('fc1_weight')
    match.append('fc1_bias')

    model = MobileNetV1(0.5, activ='prelu', bn=1e-03)
    dst = model.state_dict()
    for i, w in enumerate(dst):
        if w.endswith('.num_batches_tracked'):
            continue
        val = src[str(match[i])]
        if dst[w].dim() == 1:
            val = val.squeeze()
        dst[w] = torch.Tensor(val)
    model.load_state_dict(dst)
    model.eval()
    torch.save(model.state_dict(), 'mbnet_2d106det_insightface.pt')
    print('saved mbnet_2d106det_insightface.pt')
    os.remove('2d106det.onnx')
    print('removed 2d106det.onnx')
    os.chdir(home)
    print('returned to: ' + home)