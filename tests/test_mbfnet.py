import unittest

import numpy as np

from common import get_images, pairwise_cosine_distances
from tidytorchmodels import MobileFaceNetEncoder


class TestMBFNet(unittest.TestCase):

    def test_regression_1(self):
        model = MobileFaceNetEncoder('w600k_insightface')
        emb = model(get_images('irl_face_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual(emb.shape, (4, 512))
        np.testing.assert_almost_equal(emb[1][400:404], np.array([-0.0228, -0.0056,  0.0399, -0.0358]), decimal=4)
        np.testing.assert_almost_equal(emb[3][84:88],   np.array([ 0.0299,  0.0021, -0.0269,  0.0303]), decimal=4)
        d = pairwise_cosine_distances(emb)
        np.testing.assert_almost_equal(d[-1][:3], np.array([1.10933, 0.98076, 0.41409]), decimal=5)

    def test_regression_2(self):
        model = MobileFaceNetEncoder('ms1m_foamliu')
        emb = model(get_images('irl_face_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual(emb.shape, (4, 128))
        np.testing.assert_almost_equal(emb[1][25:29], np.array([0.0343,  0.0459,  0.1150, 0.0646]), decimal=4)
        np.testing.assert_almost_equal(emb[2][88:92], np.array([0.0345, -0.0788, -0.0353, 0.0785]), decimal=4)
        d = pairwise_cosine_distances(emb)
        np.testing.assert_almost_equal(d[-1][:3], np.array([0.97962, 0.88919, 0.38921]), decimal=5)

    def test_regression_3(self):
        model = MobileFaceNetEncoder('ms1m_xue24')
        emb = model(get_images('irl_face_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual(emb.shape, (4, 512))
        np.testing.assert_almost_equal(emb[0][142:146], np.array([-0.0717, -0.0153, -0.0437,  0.0823]), decimal=4)
        np.testing.assert_almost_equal(emb[2][265:269], np.array([ 0.0087,  0.0144,  0.0587, -0.0542]), decimal=4)
        d = pairwise_cosine_distances(emb)
        np.testing.assert_almost_equal(d[-1][:3], np.array([0.97642, 0.97108, 0.37185]), decimal=5)


if __name__ == '__main__':
    unittest.main()