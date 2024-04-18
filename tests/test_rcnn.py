import unittest

import numpy as np

from common import get_images, run_training_coco
from tidytorchmodels import Detector_FasterRCNN


class TestRCNN(unittest.TestCase):

    def test_rcnn_torchvision_v1_T(self):
        model = Detector_FasterRCNN('tv_resnet50_v1', train=True)
        losses = run_training_coco(model, seed=0)
        self.assertEqual(len(losses), 4)
        self.assertAlmostEqual(losses[0], 0.07744, places=5)
        self.assertAlmostEqual(losses[1], 0.06049, places=5)
        self.assertAlmostEqual(losses[2], 0.39630, places=5)
        self.assertAlmostEqual(losses[3], 0.46581, places=5)

    def test_rcnn_torchvision_v2_T(self):
        model = Detector_FasterRCNN('tv_resnet50_v2', train=True)
        losses = run_training_coco(model, seed=0)
        self.assertEqual(len(losses), 4)
        self.assertAlmostEqual(losses[0], 0.07088, places=5)
        self.assertAlmostEqual(losses[1], 0.04533, places=5)
        self.assertAlmostEqual(losses[2], 0.6054, places=4)
        self.assertAlmostEqual(losses[3], 0.3580, places=4)

    def test_rcnn_torchvision_mobile_lores_T(self):
        model = Detector_FasterRCNN('tv_mobilenetv3l_lores', train=True)
        losses = run_training_coco(model, seed=0)
        self.assertEqual(len(losses), 4)
        self.assertAlmostEqual(losses[0], 0.09477, places=5)
        self.assertAlmostEqual(losses[1], 0.09340, places=5)
        self.assertAlmostEqual(losses[2], 0.37972, places=5)
        self.assertAlmostEqual(losses[3], 0.35782, places=5)

    def test_rcnn_torchvision_mobile_hires_T(self):
        model = Detector_FasterRCNN('tv_mobilenetv3l_hires', train=True)
        losses = run_training_coco(model, seed=0)
        self.assertEqual(len(losses), 4)
        self.assertAlmostEqual(losses[0], 0.09, places=2)
        self.assertAlmostEqual(losses[1], 0.0549, places=4)
        self.assertAlmostEqual(losses[2], 0.349, places=3)
        self.assertAlmostEqual(losses[3], 0.4599, places=4)

    def test_rcnn_torchvision_v1(self):
        model = Detector_FasterRCNN('tv_resnet50_v1')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((88, 4), (88,)))
        self.assertEqual((b[1].shape, l[1].shape), ((42, 4), (42,)))
        np.testing.assert_almost_equal(b[0][10], np.array([334.253, 178.055, 368.729, 225.988]), decimal=3)
        np.testing.assert_almost_equal(b[0][50], np.array([300.852, 213.452, 353.634, 220.782]), decimal=3)
        np.testing.assert_almost_equal(b[1][4], np.array([93.350, 309.727, 475.038, 543.760]), decimal=3)
        np.testing.assert_almost_equal(b[1][-10], np.array([208.974, 258.579, 226.194, 311.582]), decimal=3)
        np.testing.assert_almost_equal(s[0][20:25], np.array([0.6796, 0.6519, 0.6061, 0.5157, 0.5043]), decimal=4)
        np.testing.assert_almost_equal(s[1][5:10], np.array([0.7237, 0.6540, 0.5068, 0.3902, 0.3619]), decimal=4)
        np.testing.assert_equal(l[0][70:80], np.array([67, 67, 51, 67, 62, 64, 86, 62, 86, 62]))
        np.testing.assert_equal(l[1][10:25], np.array([67, 1, 1, 67, 15, 27, 77, 15, 32, 15, 15, 67, 67, 67, 15]))

    def test_rcnn_torchvision_v2(self):
        model = Detector_FasterRCNN('tv_resnet50_v2')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((75, 4), (75,)))
        self.assertEqual((b[1].shape, l[1].shape), ((46, 4), (46,)))
        np.testing.assert_almost_equal(b[0][20], np.array([352.610, 205.653, 361.419, 219.859]), decimal=3)
        np.testing.assert_almost_equal(b[0][60], np.array([143.423, 276.501, 181.135, 286.915]), decimal=3)
        np.testing.assert_almost_equal(b[1][1], np.array([183.487, 367.362, 465.613, 558.607]), decimal=3)
        np.testing.assert_almost_equal(b[1][-5], np.array([158.761, 81.945, 237.026, 153.771]), decimal=3)
        np.testing.assert_almost_equal(s[0][40:45], np.array([0.2162, 0.1905, 0.1892, 0.1671, 0.1590]), decimal=4)
        np.testing.assert_almost_equal(s[1][10:15], np.array([0.6772, 0.6085, 0.4874, 0.3936, 0.3767]), decimal=4)
        np.testing.assert_equal(l[0][25:40], np.array([86, 86, 62, 44, 62, 86, 67, 67, 67, 64, 67, 64, 67, 86, 86]))
        np.testing.assert_equal(l[1][-15:], np.array([15, 73, 18, 67, 84, 31, 15, 15, 15, 15, 2, 15, 1, 15, 84]))

    def test_rcnn_torchvision_mobile_hires(self):
        model = Detector_FasterRCNN('tv_mobilenetv3l_hires')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((80, 4), (80,)))
        self.assertEqual((b[1].shape, l[1].shape), ((57, 4), (57,)))
        np.testing.assert_almost_equal(b[0][24], np.array([489.39, 170.88, 518.76, 282.54]), decimal=2)
        np.testing.assert_almost_equal(b[0][70], np.array([578.38, 250.50, 604.59, 266.73]), decimal=2)
        np.testing.assert_almost_equal(b[1][5], np.array([298.91, 275.88, 592.31, 512.61]), decimal=2)
        np.testing.assert_almost_equal(b[1][-8], np.array([157.61, 173.17, 230.45, 378.77]), decimal=2)
        np.testing.assert_almost_equal(s[0][5:10], np.array([0.8966, 0.8761, 0.8308, 0.7757, 0.7747]), decimal=4)
        np.testing.assert_almost_equal(s[1][25:30], np.array([0.1595, 0.1541, 0.1459, 0.1360, 0.1359]), decimal=4)
        np.testing.assert_equal(l[0][60:75], np.array([64, 86, 79, 44, 84, 64, 84, 82, 64, 62, 53, 67, 82, 64, 64]))
        np.testing.assert_equal(l[1][:15], np.array([28, 1, 15, 1, 15, 67, 1, 15, 67, 62, 1, 15, 28, 27, 15]))

    def test_rcnn_torchvision_mobile_lores(self):
        model = Detector_FasterRCNN('tv_mobilenetv3l_lores')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((32, 4), (32,)))
        self.assertEqual((b[1].shape, l[1].shape), ((15, 4), (15,)))
        np.testing.assert_almost_equal(b[0][5], np.array([544.0790, 299.3280, 594.0629, 397.1502]), decimal=4)
        np.testing.assert_almost_equal(b[1][0], np.array([213.9087, 131.7052, 448.1617, 298.2360]), decimal=4)
        np.testing.assert_almost_equal(s[0][17:22], np.array([0.1037, 0.1011, 0.0793, 0.0790, 0.0786]), decimal=4)
        np.testing.assert_equal(l[1], np.array([28, 1, 15, 1, 27, 15, 27, 67, 31, 33, 15, 31, 62, 31, 1]))

    def test_rcnn_mmdet_resnet50(self):
        model = Detector_FasterRCNN('mm_resnet50')
        b, s, l = model(get_images('coco_val2017_%s.jpg', ['000139', '455157']))
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        self.assertEqual((b[0].shape, s[0].shape), ((82, 4), (82,)))
        self.assertEqual((b[1].shape, l[1].shape), ((40, 4), (40,)))
        np.testing.assert_almost_equal(b[0][25], np.array([382.1297, 215.6923, 421.2562, 222.4981]), decimal=4)
        np.testing.assert_almost_equal(b[0][72], np.array([411.7101, 218.6185, 425.0581, 232.2054]), decimal=4)
        np.testing.assert_almost_equal(b[1][9], np.array([532.8469, 262.6869, 639.9117, 329.0356]), decimal=4)
        np.testing.assert_almost_equal(b[1][-9], np.array([155.6823, 142.8020, 364.9139, 330.7866]), decimal=4)
        np.testing.assert_almost_equal(s[0][32:37], np.array([0.2276, 0.2159, 0.2128, 0.1902, 0.1612]), decimal=4)
        np.testing.assert_almost_equal(s[1][2:7], np.array([0.9045, 0.8858, 0.8521, 0.6879, 0.5815]), decimal=4)
        np.testing.assert_equal(l[0][64:79], np.array([56, 60, 58, 56, 60, 56, 46, 56, 56, 60, 68, 56, 60, 56, 56]))
        np.testing.assert_equal(l[1][14:29], np.array([13, 13, 67, 24, 13, 60, 60, 60, 0, 26, 13, 0, 26, 60, 13]))

    def test_rcnn_mmdet_resnet50_animefaces(self):
        model = Detector_FasterRCNN('mm_resnet50_animefaces')
        b, s, _ = model(get_images('anime_det_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual((len(b), len(s)), (4, 4))
        self.assertEqual((b[0].shape, s[0].shape), ((14, 4), (14,)))
        self.assertEqual((b[1].shape, s[1].shape), ((64, 4), (64,)))
        self.assertEqual((b[2].shape, s[2].shape), ((6, 4), (6,)))
        self.assertEqual((b[3].shape, s[3].shape), ((4, 4), (4,)))
        np.testing.assert_almost_equal(b[0][10], np.array([751.9342, 276.2107, 783.7333, 311.8178]), decimal=4)
        np.testing.assert_almost_equal(b[1][50], np.array([329.8422, 381.0872, 367.5275, 419.2162]), decimal=4)
        np.testing.assert_almost_equal(b[2][3], np.array([404.4612, 164.2291, 520.1513, 310.8856]), decimal=4)
        np.testing.assert_almost_equal(b[3][1], np.array([752.1040, 98.5442, 1095.4589, 422.9254]), decimal=4)
        np.testing.assert_almost_equal(s[0][5:10], np.array([0.9873, 0.9793, 0.9594, 0.9509, 0.8711]), decimal=4)
        np.testing.assert_almost_equal(s[1][-5:], np.array([0.6398, 0.5793, 0.5513, 0.4126, 0.2921]), decimal=4)
        np.testing.assert_almost_equal(s[2], np.array([0.9989, 0.9956, 0.7671, 0.7199, 0.6205, 0.0755]), decimal=4)
        np.testing.assert_almost_equal(s[3], np.array([0.9991, 0.9988, 0.9988, 0.9686]), decimal=4)
        

if __name__ == '__main__':
    unittest.main()