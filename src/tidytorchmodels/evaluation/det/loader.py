import os
import os.path as osp
import shutil

from ...utils.download import url_download
from . import extra


def prepare_dataset(set_name):
    """Automatically downloads and unpacks the specified validation dataset's files to
    ``<project_root>/datasets/val/<set_name>`` (unless such folder already exists,
    then assumes it's all already been correctly downloaded).
    """
    if '__file__' in globals():
        # expected way (the project's root is 3 levels up from this file)
        home = osp.dirname(osp.dirname(osp.dirname(__file__)))
    else:
        # in case we decide to use zip_safe=True when packaging later
        home = os.getcwd()
    
    updir = osp.join(home, 'datasets', 'val', set_name)
    imdir = osp.join(updir, 'images')
    gtdir = osp.join(updir, 'ground_truth')
    if osp.isdir(updir):
        print('Using %s validation files at: %s' % (set_name, updir))
        return updir, imdir, gtdir
    
    print('Downloading %s validation files to: %s' % (set_name, updir))
    cwd = os.getcwd()
    os.makedirs(updir)
    os.chdir(updir)
    try:
        if set_name == 'WIDER_FACE':
            download_wider_images()
            download_wider_annotations()
        elif set_name == 'FDDB':
            download_fddb_images()
            download_fddb_annotations()
        elif set_name == 'ICARTOON':
            download_icartoon_images()
            download_icartoon_annotations()
        elif set_name == 'PIXIV2018':
            download_pixiv2018()
        elif set_name == 'PIXIV2018_ORIG':
            extra.download_pixiv2018_ORIG()
    finally:
        os.chdir(cwd)
    return updir, imdir, gtdir


def download_wider_images():
    # the link is from the dataset main page: http://shuoyang1213.me/WIDERFACE/
    link = 'https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q'
    arcn = 'WIDER_val.zip'
    url_download(link, arcn, gdrive=True)
    shutil.unpack_archive(arcn)
    os.remove(arcn)
    shutil.move(osp.join('WIDER_val', 'images'), 'images')
    shutil.rmtree('WIDER_val')


def download_wider_annotations():
    link = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip'
    arcn = 'eval_tools.zip'
    url_download(link, arcn)
    shutil.unpack_archive(arcn)
    os.remove(arcn)
    shutil.rmtree('__MACOSX')
    shutil.move(osp.join('eval_tools', 'ground_truth'), 'ground_truth')
    shutil.rmtree('eval_tools')


def download_fddb_images():
    # there's a direct link on the dataset page: http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
    # but that's a larger dataset with 28204 images, while only 2845 of them are used for validation
    # so I reuploaded those 2845 images alone to Google Drive
    link = 'https://drive.google.com/uc?id=1GLYnqrKbsHdkptQr1d2pZzDUyZ3NrCGX'
    arcn = 'FDDB_val.zip'
    url_download(link, arcn, gdrive=True)
    shutil.unpack_archive(arcn, 'images')
    os.remove(arcn)


def download_fddb_annotations():
    link = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'
    arcn = 'FDBB-folds.tgz'
    url_download(link, arcn)
    shutil.unpack_archive(arcn)
    os.remove(arcn)
    os.rename('FDDB-folds', 'ground_truth')


def download_icartoon_images():
    link = 'https://drive.google.com/uc?id=111cgWh3Z1QBviMMahAGwPKpR3IlNCrsd'
    arcn = 'personai_icartoonface_detval.zip'
    url_download(link, arcn, gdrive=True)
    print('Unpacking archive...')
    shutil.unpack_archive(arcn)
    os.remove(arcn)
    os.rename('personai_icartoonface_detval', 'images')


def download_icartoon_annotations():
    link = 'https://drive.google.com/uc?id=1qiHHCP1RvMl6kH017pAV8-QDdcMyy8PR'
    os.mkdir('ground_truth')
    flnm = osp.join('ground_truth', 'personai_icartoonface_detval.csv')
    url_download(link, flnm, gdrive=True)


def download_pixiv2018():
    link = 'https://drive.google.com/uc?id=1TV8MCoWlaX9gztysihNgW53rnYdX_R4_'
    arcn = 'PIXIV2018_1024px.zip'
    url_download(link, arcn, gdrive=True)
    print('Unpacking archive...')
    shutil.unpack_archive(arcn)
    os.remove(arcn)