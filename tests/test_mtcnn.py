import unittest

import numpy as np

from common import get_images
from tidytorchmodels import Detector_MTCNN, Landmarker_MTCNN


class TestMTCNN(unittest.TestCase):

    def test_main(self):
        model = Detector_MTCNN()
        res = model(get_images('irl_det_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (15, 5))
        self.assertEqual(res[1].shape, (5, 5))
        self.assertEqual(res[2].shape, (51, 5))
        self.assertEqual(res[3].shape, (28, 5))
        np.testing.assert_almost_equal(res[0][7], np.array([682.8788, 122.9998, 739.7405, 192.9459, 0.9997]), decimal=4)
        np.testing.assert_almost_equal(res[1][-1], np.array([927.6433, 221.3357, 974.1216, 276.0959, 0.9989]), decimal=4)
        np.testing.assert_almost_equal(res[2][44], np.array([162.0115, 53.9863, 173.8801, 67.2544, 0.8978]), decimal=4)
        np.testing.assert_almost_equal(res[3][22], np.array([150.9578, 234.9925, 199.8160, 301.9932, 0.9934]), decimal=4)
        return

    def test_landmarks(self):
        model = Landmarker_MTCNN()
        res = model(get_images('irl_face_%u.jpg', [1, 2]))
        self.assertEqual([len(res), res[0].shape, res[1].shape], [2, (5, 2), (5,  2)])
        np.testing.assert_equal(res[0], np.array([[102, 128], [158, 120], [133, 160], [116, 196], [162, 189]]))
        np.testing.assert_equal(res[1], np.array([[ 75, 120], [132,  98], [ 88, 130], [ 84, 179], [134, 164]]))


if __name__ == '__main__':
    unittest.main()