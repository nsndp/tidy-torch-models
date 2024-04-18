import torch.nn.functional as F

def unfold_input(x, stem, patch_size): # [bs, 3, 112, 112]
    """If ViT stem (1st layer) isn't conv but linear, need to unwrap every [3, p, p] patch from input images into a flattened column.
    FaceTransformer's P12S8 corresponds to 12x12 patches with stride 8, i.e. having overlapping ('linr_ov')
    FaceTransformer's P8S8 is 8x8 patches with stride 8, i.e. the usual division of images into 8x8 regions ('linr')
    There is also an implementation difference in how the flattened columns assembled: a 3x2x2 patch from RGB image can go like
    [0 1] [16 17] [32 33] => a) [0 16 32 1 17 33 4 20 36 5 21 37] (1st pixel from all channels, then 2nd pixel from all channels, etc)
    [4 5] [20 21] [36 37] => b) [0 1 4 5 16 17 20 21 32 33 36 37] (all pixels from R channel, then all pixel from G channel, etc)
    tensorflow.space_to_depth or einops.rearrange does a), but torch.nn.functional's unfold or pixel_unshuffle does b)
    Here it's a) for P8S8 and b) for P12S8, since github.com/zhongyy/Face-Transformer's models were trained like this
    a) is from https://stackoverflow.com/a/44359328
    b) have hardcoded values
    """
    if stem == 'linr_ov':
        return F.unfold(x, 12, stride=8, padding=4).transpose(1, 2) # [bs, 196, 432]
    n, c, h, w = x.shape
    p = patch_size
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(n, h // p, p, w // p, p, c)
    x = x.transpose(2, 3)
    x = x.reshape(n, h // p * w // p, -1) # [bs, 196, 192]
    #x = F.pixel_unshuffle(x, p).reshape(n, c * p ** 2, -1).transpose(1, 2)
    #x = F.unfold(x, p, stride=p).transpose(1, 2)
    return x