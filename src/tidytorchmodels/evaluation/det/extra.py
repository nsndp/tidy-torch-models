import os
import os.path as osp
import shutil

import cv2
import numpy as np

from ...utils.image import resize_keep_ratio
from ...utils.pbar import tqdm
from ...utils.download import url_download


def convert_dataset(fn, gt, imdir, gtdir, set_name, new_name, jpeg_quality=60):
    imdirDST = gtdir.replace(set_name, new_name)
    gtdirDST = imdir.replace(set_name, new_name)
    res = gt.copy()
    with open(osp.join(gtdirDST, 'boxes.txt'), 'w') as f:
        for k in tqdm(len(fn)):
            im = cv2.imread(osp.join(imdir, fn[k]))
            im, scale = resize_keep_ratio(im, 1024, upscale=False)
            cv2.imwrite(osp.join(imdirDST, fn[k]), im, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            res[k] = np.rint(gt[k] * scale).astype(int)
            f.write(fn[k] + '\n')
            f.write(str(len(res[k])) + '\n')
            for b in res[k]:
                f.write(' '.join([str(c) for c in b]) + '\n')


def download_pixiv2018_ORIG():
    """https://github.com/qhgz2013/anime-face-detector
    3.09 GB, 6641 images (0-5999 - test, 6000-6640 - val)
    """
    #pip install py7zr
    from py7zr import unpack_7zarchive
    link = 'https://drive.google.com/uc?id=1nDPimhiwbAWc2diok-6davhubNVe82pr'
    arcn = 'VOC2007-anime-face-detector-dataset.7z'
    url_download(link, arcn, gdrive=True)
    print('Unpacking archive...')
    shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
    shutil.unpack_archive(arcn)
    os.rename(osp.join('VOCdevkit2007', 'VOC2007', 'JPEGImages'), 'images')
    os.rename(osp.join('VOCdevkit2007', 'VOC2007', 'Annotations'), 'ground_truth')
    shutil.rmtree('VOCdevkit2007')
    os.remove(arcn)


def get_pixiv2018_data_ORIG(gtdir):
    import xml.etree.ElementTree as ET
    fn = ['%06d.jpg' % k for k in range(0, 6641)]
    gt = []
    for k in range(0, 6641):
        tree = ET.parse(osp.join(gtdir, '%06d.xml' % k))
        root = tree.getroot()
        gtk = []
        for face in root.findall('object'):
            x1 = int(face.find('bndbox/xmin').text)
            y1 = int(face.find('bndbox/ymin').text)
            x2 = int(face.find('bndbox/xmax').text)
            y2 = int(face.find('bndbox/ymax').text)
            gtk.append([x1, y1, x2, y2])
        gt.append(gtk)
    gt = [np.array(e) for e in gt]
    return fn, gt