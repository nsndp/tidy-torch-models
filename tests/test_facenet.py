import unittest

import numpy as np

from common import get_images
from tidytorchmodels import FaceNetEncoder


class TestFaceNet(unittest.TestCase):

    def test_regression(self):
        model = FaceNetEncoder()
        emb = model(get_images('irl_face_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual(emb.shape, (4, 512))
        np.testing.assert_almost_equal(emb[0][100:108], np.array([ 0.0068, -0.0066, -0.0551, -0.0322, -0.0331, -0.0548,  0.0612, -0.0518]), decimal=4)
        np.testing.assert_almost_equal(emb[1][:8],      np.array([-0.0300,  0.0069, -0.0658, -0.0612,  0.0508, -0.0651,  0.0128,  0.0467]), decimal=4)
        np.testing.assert_almost_equal(emb[2][-8:],     np.array([-0.0204,  0.0470,  0.0248,  0.0154, -0.0144, -0.0156,  0.0506, -0.0088]), decimal=4)
        np.testing.assert_almost_equal(emb[3][400:408], np.array([ 0.0297, -0.0122, -0.0281,  0.0492, -0.0473,  0.0425, -0.0185, -0.0171]), decimal=4)

    def _test_comparison(self):
        #!pip install facenet-pytorch
        for ds in ['vggface2', 'casia-webface']:
            imgs = get_faces_irl()
            model = FaceNetEncoder(ds)
            emb = model(imgs)

            from facenet_pytorch import InceptionResnetV1
            import cv2, torch, numpy as np
            with torch.no_grad():
                morig = InceptionResnetV1(pretrained=ds).eval()
            inp = cv2.dnn.blobFromImages(imgs, 1 / 128, (160, 160), (127.5, 127.5, 127.5), swapRB=True)
            inp = torch.from_numpy(inp).to('cpu')
            org = morig(inp).detach().cpu().numpy()

            #print(np.max(np.abs(emb - org)))
            np.testing.assert_almost_equal(emb, org)
            

if __name__ == '__main__':
    unittest.main()