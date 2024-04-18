import unittest

import numpy as np

from common import get_images
from tidytorchmodels import Detector_YOLOv3


class TestYOLOv3(unittest.TestCase):

    def test_anime(self):
        model = Detector_YOLOv3('anime')
        b, s, _ = model(get_images('anime_det_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual((len(b), len(s)), (4, 4))
        res = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (12, 5))
        self.assertEqual(res[1].shape, (69, 5))
        self.assertEqual(res[2].shape, (8, 5))
        self.assertEqual(res[3].shape, (4, 5))
        np.testing.assert_almost_equal(res[0][4], np.array([614.6, 298.6086, 656.0338, 337.2882, 0.5093]), decimal=4)
        np.testing.assert_almost_equal(res[1][3], np.array([461.1025, 230.8962, 500.1655, 269.5829, 0.9740]), decimal=4)
        np.testing.assert_almost_equal(res[2][2], np.array([397.1730, 155.0277, 520.8693, 325.2291, 0.9882]), decimal=4)
        np.testing.assert_almost_equal(res[3][1], np.array([439.94296, 22.34629, 538.27466, 119.58426, 0.99201]), decimal=4)

    def test_wider(self):
        model = Detector_YOLOv3('wider')
        b, s, _ = model(get_images('irl_det_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual((len(b), len(s)), (4, 4))
        res = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (20, 5))
        self.assertEqual(res[1].shape, (10, 5))
        self.assertEqual(res[2].shape, (100, 5))
        self.assertEqual(res[3].shape, (93, 5))
        np.testing.assert_almost_equal(res[0][10], np.array([286.4944, 335.9040, 354.3441, 426.0989, 0.9969]), decimal=4)
        np.testing.assert_almost_equal(res[3][25], np.array([460.0020, 143.5856, 493.6367, 193.8361, 0.8309]), decimal=4)

    def test_coco(self):
        model = Detector_YOLOv3('coco_darknet')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((100, 4), (100,)))
        self.assertEqual((b[1].shape, s[1].shape), ((34, 4), (34,)))
        np.testing.assert_almost_equal(b[0][15], np.array([449.3380, 121.4010, 461.3889, 141.6084]), decimal=4)
        np.testing.assert_almost_equal(b[1][2], np.array([162.6238, 157.5956, 351.1005, 415.0244]), decimal=4)
        np.testing.assert_almost_equal(s[0][40:45], np.array([0.0249, 0.0228, 0.0222, 0.0216, 0.0187]), decimal=4)
        np.testing.assert_equal(l[1][14:29], np.array([13, 73, 26, 13, 26, 0, 13, 13, 13, 13, 13, 13, 13, 73, 63]))
        return

    def test_coco_mobile_416(self):
        model = Detector_YOLOv3('coco_mobile2_416')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((100, 4), (100,)))
        self.assertEqual((b[1].shape, s[1].shape), ((100, 4), (100,)))
        np.testing.assert_almost_equal(b[0][50], np.array([354.92, 218.82, 370.41, 230.51]), decimal=2)
        np.testing.assert_almost_equal(b[1][20], np.array([539.94, 260.32, 632.27, 315.14]), decimal=2)
        np.testing.assert_almost_equal(s[0][5:10], np.array([0.4073, 0.3790, 0.3634, 0.3282, 0.2709]), decimal=4)
        np.testing.assert_equal(l[1][35:50], np.array([60, 8, 39, 67, 24, 60, 26, 7, 39, 41, 0, 13, 26, 26, 0]))
        return

    def test_coco_mobile_320(self):
        model = Detector_YOLOv3('coco_mobile2_320')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((100, 4), (100,)))
        self.assertEqual((b[1].shape, s[1].shape), ((100, 4), (100,)))
        np.testing.assert_almost_equal(b[0][50], np.array([427.88, 164.43, 459.94, 300.62]), decimal=2)
        np.testing.assert_almost_equal(b[1][20], np.array([261.48, 334.35, 328.53, 381.78]), decimal=2)
        np.testing.assert_almost_equal(s[0][5:10], np.array([0.3886, 0.3190, 0.3121, 0.2784, 0.2763]), decimal=4)
        np.testing.assert_equal(l[1][35:50], np.array([13, 13, 26, 67, 25, 62, 24, 13, 0, 67, 67, 56, 67, 0, 0]))
        return


if __name__ == '__main__':
    unittest.main()