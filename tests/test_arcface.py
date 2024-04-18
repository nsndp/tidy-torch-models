import unittest

import numpy as np

from common import get_images, pairwise_cosine_distances
from tidytorchmodels import ArcFaceEncoder


class TestArcFace(unittest.TestCase):

    def test_regression(self):
        model = ArcFaceEncoder('r50')
        emb = model(get_images('irl_face_%u.jpg', [1, 2, 3, 4]))
        self.assertEqual(emb.shape, (4, 512))
        np.testing.assert_almost_equal(emb[0][220:224], np.array([0.0365, 0.0669, 0.0125, -0.0024]), decimal=4)
        np.testing.assert_almost_equal(emb[2][-80:-76], np.array([0.0305, 0.0325, 0.0341, -0.0226]), decimal=4)
        d = pairwise_cosine_distances(emb)
        np.testing.assert_almost_equal(d[-1][:3], np.array([0.98302, 1.04085, 0.30475]), decimal=5)


if __name__ == '__main__':
    unittest.main()