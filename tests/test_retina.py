import unittest

import numpy as np

from common import get_images, run_training_coco
from tidytorchmodels import Detector_Retina


class TestRetina(unittest.TestCase):

    def test_retinanet_tv_T(self):
        model = Detector_Retina('coco_resnet50_tv', train=True)
        losses = run_training_coco(model)
        self.assertEqual(len(losses), 2)
        self.assertAlmostEqual(losses[0], 0.34683, places=5)
        self.assertAlmostEqual(losses[1], 0.32876, places=5)

    def test_retinanet_tv(self):
        model = Detector_Retina('coco_resnet50_tv')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((241, 4), (241,)))
        self.assertEqual((b[1].shape, l[1].shape), ((289, 4), (289,)))
        np.testing.assert_almost_equal(b[0][100], np.array([355.996, 214.633, 381.615, 289.376]), decimal=3)
        np.testing.assert_almost_equal(b[0][200], np.array([316.892, 223.678, 359.765, 310.691]), decimal=3)
        np.testing.assert_almost_equal(b[1][4], np.array([524.262, 271.283, 640.000, 486.419]), decimal=3)
        np.testing.assert_almost_equal(b[1][-10], np.array([273.629, 282.719, 343.064, 312.126]), decimal=3)
        np.testing.assert_almost_equal(s[0][50:55], np.array([0.1915, 0.1914, 0.1911, 0.1909, 0.1908]), decimal=4)
        np.testing.assert_almost_equal(s[1][5:10], np.array([0.3458, 0.3283, 0.3276, 0.3109, 0.3082]), decimal=4)
        np.testing.assert_equal(l[0][130:140], np.array([86, 63, 77, 67, 62, 1, 62, 67, 62, 76]))
        np.testing.assert_equal(l[1][10:25], np.array([67, 67, 15, 1, 15, 15, 67, 67, 67, 1, 28, 27, 84, 67, 67]))

    def test_mobilenet(self):
        model = Detector_Retina('wider_mobile_b6')
        b, s = model(get_images('irl_det_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual((len(b), len(s)), (4, 4))
        rb = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(rb[0].shape, (24, 5))
        self.assertEqual(rb[1].shape, (20, 5))
        self.assertEqual(rb[2].shape, (358, 5))
        self.assertEqual(rb[3].shape, (76, 5))
        np.testing.assert_almost_equal(rb[0][10], np.array([541.10864, 105.35722, 591.18164, 169.09984, 0.99779]), decimal=4)
        np.testing.assert_almost_equal(rb[0][22], np.array([585.0051, 129.43068, 612.2147, 164.56346, 0.02089]), decimal=4)
        np.testing.assert_almost_equal(rb[1][4], np.array([931.3569, 213.46419, 981.17365, 282.21033, 0.98094]), decimal=4)
        np.testing.assert_almost_equal(rb[1][8], np.array([298.3359, 194.7117, 355.0907, 270.89612, 0.04269]), decimal=4)
        np.testing.assert_almost_equal(rb[2][1], np.array([451.18222, 312.84677, 506.44626, 387.15103, 0.99937]), decimal=5)
        np.testing.assert_almost_equal(rb[2][92], np.array([454.6434, 258.5825, 471.7198, 280.2969, 0.1227]), decimal=4)
        np.testing.assert_almost_equal(rb[3][12], np.array([196.038, 235.07074, 256.24564, 313.40533, 0.99574]), decimal=5)
        np.testing.assert_almost_equal(rb[3][-5], np.array([621.6558, 4.7005, 637.6158, 27.7780, 0.0208]), decimal=4)

    def test_resnet50_A(self):
        model = Detector_Retina('wider_resnet50_b6')
        b, s = model(get_images('irl_det_%u.jpg', [1, 2]))
        self.assertEqual((len(b), len(s)), (2, 2))
        rb = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(rb[0].shape, (14, 5))
        self.assertEqual(rb[1].shape, (9, 5))
        np.testing.assert_almost_equal(rb[0][5], np.array([290.9456, 339.1132, 352.7952, 420.90085, 0.99942017]), decimal=4)
        np.testing.assert_almost_equal(rb[1][1], np.array([81.9536, 199.9863, 172.0615, 331.8756, 0.99976426]), decimal=4)

    def test_resnet50_B(self):
        model = Detector_Retina('mixed_resnet50_bbt')
        b, s = model(get_images('irl_det_%u.jpg', [3, 4]))
        self.assertEqual((len(b), len(s)), (2, 2))
        rb = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(rb[0].shape, (43, 5))
        self.assertEqual(rb[1].shape, (27, 5)) 
        np.testing.assert_almost_equal(rb[0][2], np.array([162.3646, 301.0576, 217.6351, 372.3147, 0.9209]), decimal=4)
        np.testing.assert_almost_equal(rb[1][6], np.array([398.3162, 309.1897, 479.2617, 406.1430, 0.8273]), decimal=4)

    def test_resnet152(self):
        model = Detector_Retina('mixed_resnet152_bbt')
        b, s = model(get_images('irl_det_%u.jpg', [3, 4]))
        self.assertEqual((len(b), len(s)), (2, 2))
        rb = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(rb[0].shape, (43, 5))
        self.assertEqual(rb[1].shape, (26, 5))
        np.testing.assert_almost_equal(rb[0][1], np.array([46.0401, 243.8737, 103.3121, 305.6734, 0.9241]), decimal=4)
        np.testing.assert_almost_equal(rb[1][4], np.array([24.19569, 221.45947, 83.07320, 294.79553, 0.85426]), decimal=5)


if __name__ == '__main__':
    unittest.main()