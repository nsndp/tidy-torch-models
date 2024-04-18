from functools import partial

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations.classify import predict
from .operations.unfold import unfold_input
from .wconvert.w_vit import wconv_openai, wconv_hface, wconv_facetr, wconv_animesion, wconv_tv
from ..utils.weights import load_weights

# adapted from
# https://github.com/arkel23/animesion/tree/main/classification_tagging/models/vit_animesion
# https://github.com/zhongyy/Face-Transformer/tree/main/copy-to-vit_pytorch-path
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L235
# https://github.com/openai/CLIP/blob/main/clip/model.py
# all dropouts are removed since using only for inference


class MultiHeadedSelfAttention(nn.Module):
    
    def __init__(self, dim, heads, att_scale):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = heads
        self.scale = dim if att_scale != 'per_head' else dim // heads
    
    def split(self, x):
        return x.view(*x.shape[:2], self.n_heads, -1).transpose(1, 2)

    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x) # [bs, n*n+1, dim]
        q, k, v = [self.split(el) for el in [q, k, v]]           # [bs, heads, n*n+1, dim/heads]
        scores = q @ k.transpose(2, 3) / (self.scale ** .5)      # [bs, heads, n*n+1, n*n+1]
        scores = F.softmax(scores, dim=-1)
        h = (scores @ v).transpose(1, 2)   # [bs, n*n+1, heads, dim/heads]
        h = h.reshape(*x.shape[:2], -1)    # [bs, n*n+1, dim]
        return h


class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, dim, ff_dim, gelu_type):
        super().__init__()
        assert gelu_type in ['exact', 'quick']
        self.act = F.gelu if gelu_type == 'exact' else self.quick_gelu
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        
    def quick_gelu(self, x):
        # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
        return x * torch.sigmoid(1.702 * x)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    
    def __init__(self, dim, heads, ff_dim, eps, att_scale, gelu_type):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = MultiHeadedSelfAttention(dim, heads, att_scale)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.pwff = PositionWiseFeedForward(dim, ff_dim, gelu_type)
    
    def forward(self, x):
        h = self.attn(self.norm1(x))
        h = self.proj(h)
        x = x + h
        h = self.pwff(self.norm2(x))
        x = x + h
        return x


class Transformer(nn.Module):
    
    def __init__(self, dim, depth, heads, ff_dim, eps, att_scale, gelu_type):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, ff_dim, eps, att_scale, gelu_type) for _ in range(depth)])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    
    def __init__(self, device, img_size, patch_size, dim, depth, eps=1e-05, stem='conv',
                 pre_norm=False, att_scale='per_head', gelu_type='exact', projection=None,
                 classes=None):
        super().__init__()
        p = patch_size
        self.classes = classes
        self.stem = stem
        self.unfold = partial(unfold_input, stem=stem, patch_size=p)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // p) ** 2 + 1, dim))
        if stem == 'conv': self.patch_embedding = nn.Conv2d(3, dim, (p, p), (p, p), bias = not pre_norm)
        if stem == 'linr': self.patch_embedding = nn.Linear(3 * p ** 2, dim)
        if stem == 'linr_ov': self.patch_embedding = nn.Linear(3 * 12 ** 2, dim)
        if pre_norm:
            self.norm_pre = nn.LayerNorm(dim, eps=eps)
        self.transformer = Transformer(dim, depth, dim // 64, dim * 4, eps, att_scale, gelu_type)
        self.norm = nn.LayerNorm(dim, eps=eps)
        if projection is not None:
            self.projection = nn.Linear(dim, projection, bias=False)
        if classes:
            self.fc = nn.Linear(dim, classes)
        self.to(device)
    
    def forward(self, x):
        if self.stem == 'conv':                     # (n = img_size // patch_size = number of patches)
            x = self.patch_embedding(x)             # [bs, dim, n, n]
            x = x.flatten(2).transpose(1, 2)        # [bs, n*n, dim]
        else:
            x = self.unfold(x)                      # [bs, n*n, 3*p*p]
            x = self.patch_embedding(x)             # [bs, n*n, dim]
        t = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((t, x), dim=1)                # [bs, n*n+1, dim]
        x = x + self.pos_embedding                  # [bs, n*n+1, dim]
        if hasattr(self, 'norm_pre'):
            x = self.norm_pre(x)
        x = self.transformer(x)                     # [bs, n*n+1, dim]
        x = x[:, 0]                                 # [bs, dim]
        x = self.norm(x)                            # [bs, dim]
        if hasattr(self, 'projection'):
            x = self.projection(x)
        if not self.classes:
            return x
        y = self.fc(x)
        y = torch.softmax(y, dim=-1)
        return y, x


class VitEncoder():

    tvhub = 'https://download.pytorch.org/models/'
    oaihub = 'https://openaipublic.azureedge.net/clip/models/'
    links = {
        'B16_ImageNet1K_TorchVision': tvhub + 'vit_b_16_swag-9ac1b537.pth',
        'L16_ImageNet1K_TorchVision': tvhub + 'vit_l_16_swag-4f3808c9.pth',
        'P8S8_MSCeleb1M': '1OZRU430CjABSJtXU0oHZHlxgzXn6Gaqu',
        'P12S8_MSCeleb1M': '1U7c_ojiuRPBfolvziB_VthksABHaFKud',
        
        # two different download locations for clip, openai or huggingface (the weights are the same)
        # openai files are FP16 hence smaller, huggingface is more consistent with download speeds
        'B32_CLIP_OpenAI': oaihub + '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
        'B16_CLIP_OpenAI': oaihub + '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
        'L14_CLIP_OpenAI': oaihub + 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',
        'B32_CLIP_HuggingFace': 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin',
        'B16_CLIP_HuggingFace': 'https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/pytorch_model.bin',
        'L14_CLIP_HuggingFace': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin',
        
        # https://drive.google.com/drive/folders/1fzLLGvu7IzmjFy8LZolwkT-NTzZPYonb
        # 1hEtmrzlh7RrXuUoxi5eqMQd5yIirQ-XC - verify_danbooruFaces_b16_ptTrue_bat....ckpt  336.6MB Aug 20, 2021 : ViT-B-16
        # 1eZai1_gjos6TNeQZg6IY-cIWxtg0Pxah - verify_danbooruFaces_l16_ptTrue_bat....ckpt  1.14GB  Aug 20, 2021 : ViT-L-16
        # 1kJx8eLmY0kv4m8QV-N8MwoLLbxLfN87g - danbooruFaces_B_16_image128_batch16....ckpt  336.6MB Jan 15, 2022 : ViT-B-16-IFA
        # 1V0kF67t9bEsO3sHtcHtPAePGmjfYdvHc - danbooruFaces_L_16_image128_batch16....ckpt  1.14GB  Jan 17, 2022 : ViT-L-16-IFA
        # 1pFADAEGz8woim_MRhDhtBN4hW6BrQByH - danbooruFull_B_16_image128_batch16_....ckpt  378.5MB Aug 20, 2021 : ViLT-B-16
        'B16_Danbooru_Faces': '1hEtmrzlh7RrXuUoxi5eqMQd5yIirQ-XC',
        'L16_Danbooru_Faces': '1eZai1_gjos6TNeQZg6IY-cIWxtg0Pxah',
        'B16_Danbooru_Full': '1pFADAEGz8woim_MRhDhtBN4hW6BrQByH'
    }

    class_count = {
        'ImageNet1K': 1000,
        'Danbooru': 3263,
        'MSCeleb1M': 93431
    }
        
    def __init__(self, src, device=None, classify=False):
        assert src in self.links
        print('Initializing ViT_%s model' % src)
        dv = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataset = src.split('_')[1]
        self.dataset = dataset
        num_cls = None if not classify else self.class_count[dataset]
        w_jit = src.endswith('OpenAI')
        dim =  768 if src[0] != 'L' else 1024
        depth = 12 if src[0] != 'L' else 24

        if dataset == 'ImageNet1K':
            self.img_size = 384 if src[0] != 'L' else 512
            # patch size is contained in the name (e.g. 'L14' => size=14)
            patch_size = int(src[1:3])
            extra_params = { 'eps': 1e-6 }
            wconv = partial(wconv_tv, classify=classify)
            self.prep = { 'lib': 'PIL', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225) }
        
        elif dataset == 'Danbooru':
            self.img_size = 128
            patch_size = 16
            extra_params = { 'eps': 1e-12 }
            wconv = partial(wconv_animesion, classify=classify)
            self.prep = { 'lib': 'cv2', 'mean': (127.5, 127.5, 127.5), 'std': 127.5 }

        elif dataset == 'CLIP':
            self.img_size = 224
            patch_size = int(src[1:3])
            proj = 512 if src[0] != 'L' else 768
            extra_params = { 'pre_norm': True, 'gelu_type': 'quick', 'projection': proj }
            wconv = wconv_openai if src.endswith('OpenAI') else wconv_hface
            self.prep = { 'lib': 'PIL', 'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711) }
        
        else:
            self.img_size = 112
            patch_size, dim, depth = 8, 512, 20
            stem = 'linr_ov' if src.startswith('P12') else 'linr'
            extra_params = { 'stem': stem, 'att_scale': 'total' }
            wconv = partial(wconv_facetr, classify=classify)
            self.prep = { 'lib': 'cv2', 'mean': (0, 0, 0), 'std': 1 }

        self.model = ViT(dv, self.img_size, patch_size, dim, depth, classes=num_cls, **extra_params)
        load_weights(self.model, self.links[src], 'vit_' + src.lower(), wconv, jit=w_jit)
        self.model.eval()
        print()
    
    def __call__(self, images):
        if self.prep['lib'] == 'cv2':
            inp = cv2.dnn.blobFromImages(images, 1 / self.prep['std'], (self.img_size, self.img_size),
                                         self.prep['mean'], swapRB=True)
            dv = next(self.model.parameters()).device
            inp = torch.from_numpy(inp).to(dv)
            with torch.inference_mode():
                out = self.model(inp)
            if isinstance(out, tuple):
                return (out[0].cpu().numpy(), out[1].cpu().numpy())
            return out.cpu().numpy()

        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
        prep = Compose([Resize(self.img_size, interpolation=InterpolationMode.BICUBIC),
                        CenterCrop(self.img_size), ToTensor(),
                        Normalize(self.prep['mean'], self.prep['std']),])
        with torch.inference_mode():
            dv = next(self.model.parameters()).device
            inp = torch.stack([prep(im) for im in images]).to(dv)
            out = self.model(inp)
        if isinstance(out, tuple):
            return (out[0].cpu().numpy(), out[1].cpu().numpy())
        return out.cpu().numpy()

    def get_predictions(self, probs, topk=5):
        return predict(self.dataset, probs, topk)