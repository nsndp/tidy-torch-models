import unittest

import numpy as np

from common import get_images
from tidytorchmodels import Detector_SSD


class TestSSD(unittest.TestCase):

    def test_ssd_300_torchvision(self):
        model = Detector_SSD('vgg16_300_tv')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((200, 4), (200,)))
        self.assertEqual((b[1].shape, l[1].shape), ((200, 4), (200,)))
        np.testing.assert_almost_equal(b[0][100], np.array([63.50, 300.53, 417.14, 390.15]), decimal=2)
        np.testing.assert_almost_equal(b[0][10], np.array([466.41, 354.58, 640.00, 426.00]), decimal=2)
        np.testing.assert_almost_equal(b[1][20], np.array([388.02, 291.70, 578.63, 466.87]), decimal=2)
        np.testing.assert_almost_equal(b[1][-40], np.array([387.71, 253.64, 640.00, 358.64]), decimal=2)
        np.testing.assert_almost_equal(s[0][24:29], np.array([0.1136, 0.1127, 0.1123, 0.1087, 0.1075]), decimal=4)
        np.testing.assert_almost_equal(s[1][95:100], np.array([0.0534, 0.0530, 0.0530, 0.0529, 0.0527]), decimal=4)
        np.testing.assert_equal(l[0][115:130], np.array([64, 62, 1, 1, 84, 64, 64, 67, 86, 67, 84, 62, 67, 64, 67]))
        np.testing.assert_equal(l[1][45:60], np.array([15, 33, 15, 1, 67, 1, 1, 1, 1, 1, 31, 15, 67, 1, 15]))

    def test_ssd_300_mmdetection(self):
        model = Detector_SSD('vgg16_300_mm')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((200, 4), (200,)))
        self.assertEqual((b[1].shape, l[1].shape), ((200, 4), (200,)))
        np.testing.assert_almost_equal(b[0][100], np.array([425.72, 264.66, 439.83, 304.39]), decimal=2)
        np.testing.assert_almost_equal(b[0][25], np.array([279.50, 219.78, 431.77, 256.41]), decimal=2)
        np.testing.assert_almost_equal(b[1][1], np.array([161.43, 164.21, 338.99, 477.41]), decimal=2)
        np.testing.assert_almost_equal(b[1][-40], np.array([124.60, 116.47, 175.72, 200.03]), decimal=2)
        np.testing.assert_almost_equal(s[0][26:31], np.array([0.1258, 0.1240, 0.1222, 0.1205, 0.1163]), decimal=4)
        np.testing.assert_almost_equal(s[1][80:85], np.array([0.0558, 0.0556, 0.0550, 0.0546, 0.0545]), decimal=4)
        np.testing.assert_equal(l[0][71:86], np.array([56, 56, 0, 58, 60, 56, 58, 56, 56, 73, 73, 73, 58, 72, 0]))
        np.testing.assert_equal(l[1][50:65], np.array([25, 25, 25, 73, 25, 60, 0, 56, 24, 56, 60, 0, 73, 56, 25]))

    def test_ssd_512_mmdetection(self):
        model = Detector_SSD('vgg16_512_mm')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((200, 4), (200,)))
        self.assertEqual((b[1].shape, l[1].shape), ((200, 4), (200,)))
        np.testing.assert_almost_equal(b[0][75], np.array([275.15, 211.36, 372.67, 231.54]), decimal=2)
        np.testing.assert_almost_equal(b[0][18], np.array([327.60, 220.27, 371.37, 309.70]), decimal=2)
        np.testing.assert_almost_equal(b[1][5], np.array([130.33, 198.97, 251.73, 498.45]), decimal=2)
        np.testing.assert_almost_equal(b[1][-66], np.array([536.68, 460.75, 616.41, 505.82]), decimal=2)
        np.testing.assert_almost_equal(s[0][12:17], np.array([0.1847, 0.1794, 0.1740, 0.1729, 0.1543]), decimal=4)
        np.testing.assert_almost_equal(s[1][44:49], np.array([0.0771, 0.0764, 0.0760, 0.0756, 0.0756]), decimal=4)
        np.testing.assert_equal(l[0][116:131], np.array([0, 73, 58, 60, 56, 56, 0, 56, 60, 56, 60, 56, 73, 60, 56]))
        np.testing.assert_equal(l[1][32:47], np.array([0, 13, 73, 0, 13, 60, 73, 13, 13, 13, 73, 13, 0, 13, 13]))

    def test_ssdlite_mmdetection(self):
        model = Detector_SSD('mobile2_320_mm')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((200, 4), (200,)))
        self.assertEqual((b[1].shape, l[1].shape), ((200, 4), (200,)))
        np.testing.assert_almost_equal(b[0][44], np.array([455.63, 350.44, 537.32, 426.00]), decimal=2)
        np.testing.assert_almost_equal(b[0][155], np.array([341.20, 195.73, 352.07, 212.85]), decimal=2)
        np.testing.assert_almost_equal(b[1][6], np.array([204.46, 126.71, 369.76, 344.50]), decimal=2)
        np.testing.assert_almost_equal(b[1][25], np.array([93.65, 327.72, 231.41, 508.98]), decimal=2)
        np.testing.assert_almost_equal(s[0][35:40], np.array([0.0828, 0.0811, 0.0811, 0.0770, 0.0760]), decimal=4)
        np.testing.assert_almost_equal(s[1][1:6], np.array([0.8175, 0.3833, 0.2570, 0.2292, 0.2250]), decimal=4)
        np.testing.assert_equal(l[0][5:20], np.array([0, 60, 62, 60, 62, 72, 56, 56, 60, 56, 72, 60, 62, 72, 58]))
        np.testing.assert_equal(l[1][105:120], np.array([0, 0, 0, 60, 24, 26, 56, 56, 39, 73, 73, 13, 56, 13, 0]))

    def test_ssdlite_torchvision(self):
        model = Detector_SSD('mobile3_320_tv')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((300, 4), (300,)))
        self.assertEqual((b[1].shape, l[1].shape), ((300, 4), (300,)))
        np.testing.assert_almost_equal(b[0][8], np.array([218.58, 227.20, 307.70, 324.38]), decimal=2)
        np.testing.assert_almost_equal(b[0][-8], np.array([379.70, 340.14, 618.97, 426.00]), decimal=2)
        np.testing.assert_almost_equal(b[1][16], np.array([333.78, 270.33, 640.00, 430.68]), decimal=2)
        np.testing.assert_almost_equal(b[1][-16], np.array([485.82, 109.70, 514.96, 151.76]), decimal=2)
        np.testing.assert_almost_equal(s[0][10:15], np.array([0.1867, 0.1594, 0.1376, 0.1212, 0.1211]), decimal=4)
        np.testing.assert_almost_equal(s[1][:5], np.array([0.9402, 0.8387, 0.6000, 0.5337, 0.4320]), decimal=4)
        np.testing.assert_equal(l[0][2:17], np.array([72, 67, 82, 82, 67, 67, 62, 62, 67, 67, 82, 67, 67, 67, 72]))
        np.testing.assert_equal(l[1][202:217], np.array([28, 1, 1, 84, 1, 1, 73, 64, 31, 27, 64, 84, 84, 1, 1]))

    
if __name__ == '__main__':
    unittest.main()