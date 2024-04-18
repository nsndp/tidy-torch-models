import struct

import cv2
import numpy as np


def resize_keep_ratio(img, to_area, upscale=True):
    """Resizes an image to fit into the specified area while keeping the aspect ratio.
    If ``to_area`` is not tuple, the area is inferred to be square (to_area, to_area).
    If ``upscale`` is False, only bigger images are resized, while smaller ones return unaltered.
    """
    h, w = img.shape[:2]
    aw, ah = to_area if isinstance(to_area, tuple) else (to_area, to_area)
    scale = min(aw / w, ah / h)
    if scale != 1 and (upscale or scale < 1):
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img, scale


def pad_to_area(img, area, halign='left', valign='top'):
    """"""
    assert halign in ['left', 'right', 'center']
    assert valign in ['top', 'middle', 'bottom']
    h, w = img.shape[:2]
    aw, ah = area if isinstance(area, tuple) else (area, area)
    assert aw >= w, 'area smaller than image at x'
    assert ah >= h, 'area smaller than image at y'
    px = max(0, aw - w)
    py = max(0, ah - h)
    px = (0, px) if halign == 'left' else ((px, 0) if halign == 'right' else (px // 2, px - px //2))
    py = (0, py) if valign == 'top' else ((py, 0) if valign == 'bottom' else (py // 2, py - py //2))
    img = np.pad(img, (py, px, (0, 0)))
    return img


#def crop_to_area(img, area):
#    """"""
#    h, w = img.shape[:2]
#    px1, py1, px2, py2 = area
#    x1, x2 = int(px1 * w), int(px2 * w + 1)
#    y1, y2 = int(py1 * h), int(py2 * h + 1)
#    return img[y1:y2, x1:x2, :]


def read_imsize_binary(path):
    """Extracts the width and height of an image located at ``path``
    without reading all data by analyzing the beginning as raw bytes.
    
    Sources:
    https://github.com/scardine/image_size/blob/master/get_image_size.py
    JPG: https://stackoverflow.com/a/63479164
    JPG: https://stackoverflow.com/a/35443269
    PNG: https://stackoverflow.com/a/5354562
    """
    w, h = None, None
    with open(path, 'rb') as f:
        start = f.read(2)
        if start == b'\xFF\xD8': # JPEG
            b = f.read(1)
            while (b and ord(b) != 0xDA):
                while (ord(b) != 0xFF): b = f.read(1)
                while (ord(b) == 0xFF): b = f.read(1)
                if (ord(b) == 0x01 or ord(b) >= 0xD0 and ord(b) <= 0xD9):
                    b = f.read(1)
                elif (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                    f.read(3)
                    h, w = struct.unpack('>HH', f.read(4))
                    break
                else:
                    seg_len = int(struct.unpack(">H", f.read(2))[0])
                    f.read(seg_len - 2)
                    b = f.read(1)
        elif start == b'\x89\x50': # PNG
            f.read(14)
            w, h = struct.unpack(">LL", f.read(8))
    return w, h