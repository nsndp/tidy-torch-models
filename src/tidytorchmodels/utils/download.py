import os
import os.path as osp
import re
import requests
import shutil

from .pbar import tqdm


def url_download(url, dst=None, gdrive=False):
    # adapted from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    # and https://github.com/wkentaro/gdown/blob/main/gdown/download.py
    CHUNK_SIZE = 1024 * 1024 # 1MB
    session = requests.session()
    headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36' }
    params = { 'confirm': 1 }
    response = session.get(url, headers=headers, params=params, stream=True, verify=True)
    
    # for Google Drive in case it returns a "file too big for virus scan, confirm to download anyway" page
    # (adding 'confirm 1' to params like above seems to be enough but let's keep this too for failproofing)
    if gdrive and 'Content-Disposition' not in response.headers:
        m = re.search('id="downloadForm" action="(.+?)"', response.text)
        if m:
            url = m.groups()[0].replace('&amp;', '&')
            response = session.get(url, headers=headers, stream=True, verify=True)
        else:
            print('Unable to download from Google Drive')
            return
  
    try:
        total = response.headers.get('Content-Length')
        total = int(total) if total else None
        dst = dst if dst else osp.basename(url)
        with open(dst, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
    finally:
        session.close()