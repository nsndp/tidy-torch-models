import unittest

import numpy as np

from common import get_images
from tidytorchmodels import Landmarker_CoordReg


class TestLandmarker(unittest.TestCase):

    def test1(self):
        model = Landmarker_CoordReg()
        res = model(get_images('irl_face_%u.jpg', [1, 2]))
        self.assertEqual([len(res), res[0].shape, res[1].shape], [2, (5, 2), (5,  2)])
        np.testing.assert_equal(res[0], np.array([[77, 94], [116, 89], [97, 118], [86, 148], [121, 142]]))
        np.testing.assert_equal(res[1], np.array([[58, 90], [98, 75], [67, 96], [65, 133], [100, 120]]))


if __name__ == '__main__':
    unittest.main()