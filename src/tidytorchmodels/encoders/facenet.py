import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.basic import ConvUnit
from ..utils.weights import load_weights

# adapted from: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py
# explanation:  https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
# paper:        https://arxiv.org/pdf/1602.07261.pdf


def conv_unit(cin, cout, k, s=1, p=0):
    return ConvUnit(cin, cout, k, s, p, 'relu', bn=1e-3)


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = conv_unit(256, 32, 1)
        self.branch1 = nn.Sequential(conv_unit(256, 32, 1), conv_unit(32, 32, 3, p=1))
        self.branch2 = nn.Sequential(conv_unit(256, 32, 1), conv_unit(32, 32, 3, p=1),
                                     conv_unit(32, 32, 3, p=1))
        self.conv2d = nn.Conv2d(96, 256, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = F.relu(out)
        return out
    

class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = conv_unit(896, 128, 1)
        self.branch1 = nn.Sequential(
            conv_unit(896, 128, 1),
            conv_unit(128, 128, (1, 7), p=(0, 3)),
            conv_unit(128, 128, (7, 1), p=(3, 0))
        )
        self.conv2d = nn.Conv2d(256, 896, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = F.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, relu=True):
        super().__init__()
        self.scale = scale
        self.relu = relu
        self.branch0 = conv_unit(1792, 192, 1)
        self.branch1 = nn.Sequential(
            conv_unit(1792, 192, 1),
            conv_unit(192, 192, (1, 3), p=(0, 1)),
            conv_unit(192, 192, (3, 1), p=(1, 0))
        )
        self.conv2d = nn.Conv2d(384, 1792, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.relu:
            out = F.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()
        self.branch0 = conv_unit(256, 384, 3, s=2)
        self.branch1 = nn.Sequential(
            conv_unit(256, 192, 1),
            conv_unit(192, 192, 3, p=1),
            conv_unit(192, 256, 3, s=2)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(conv_unit(896, 256, 1), conv_unit(256, 384, 3, s=2))
        self.branch1 = nn.Sequential(conv_unit(896, 256, 1), conv_unit(256, 256, 3, s=2))
        self.branch2 = nn.Sequential(conv_unit(896, 256, 1), conv_unit(256, 256, 3, p=1),
                                     conv_unit(256, 256, 3, s=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
        

class InceptionResnetV1(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.stem = nn.Sequential(
            conv_unit(3, 32, 3, s=2),     # conv2d_1a
            conv_unit(32, 32, 3),         # conv2d_2a
            conv_unit(32, 64, 3, p=1),    # conv2d_2b
            nn.MaxPool2d(3, stride=2),    # maxpool_3a
            conv_unit(64, 80, 1),         # conv2d_3b
            conv_unit(80, 192, 3),        # conv2d_4a
            conv_unit(192, 256, 3, s=2)   # conv2d_4b
        )
        self.main = nn.Sequential(
            nn.Sequential(*[Block35(scale=0.17) for i in range(5)]), # Inception block A x 5
            Mixed_6a(),                                              # Reduction block A
            nn.Sequential(*[Block17(scale=0.1) for i in range(10)]), # Inception block B x 10
            Mixed_7a(),                                              # Reduction block B
            nn.Sequential(*[Block8(scale=0.2) for i in range(5)]),   # Inception block C x 5
            Block8(relu=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(1792, 512, bias=False),
            nn.BatchNorm1d(512, 0.001)
        )
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class FaceNetEncoder():

    stor = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/'
    links = {
        'vggface2': stor + '20180402-114759-vggface2.pt',
        'casia_webface': stor + '20180408-102900-casia-webface.pt'
    }

    def no_classify(self, wd):
        wd.pop('logits.weight')
        wd.pop('logits.bias')
        return wd

    def __init__(self, src='vggface2', device=None):
        assert src in self.links
        print('Initializing Inception-Resnet V1 model for face feature extraction')
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(dv)
        load_weights(self.model, self.links[src], 'facenet_' + src, extra_conversion=self.no_classify)
        self.model.eval()
    
    def __call__(self, images):
        inp = cv2.dnn.blobFromImages(images, 1 / 128, (160, 160), (127.5, 127.5, 127.5), swapRB=True)
        inp = torch.from_numpy(inp).to(next(self.model.parameters()).device)
        with torch.inference_mode():
            out = self.model(inp)
        return out.cpu().numpy()