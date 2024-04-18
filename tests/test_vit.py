import unittest

import numpy as np

from common import get_images
from tidytorchmodels import VitEncoder

EXTENDED = False


class TestVIT(unittest.TestCase):

    def test_vit_anime(self):
        imgs = get_images('aniface%u.jpg', [1, 2])
        model = VitEncoder('B16_Danbooru_Faces', classify=True)
        prob, emb = model(imgs)
        self.assertEqual(emb.shape, (2, 768))
        np.testing.assert_almost_equal(emb[0][100:105], np.array([-0.4530, -2.1694, 0.0624, -0.7991, -0.3798]), decimal=4)
        np.testing.assert_almost_equal(emb[1][640:645], np.array([0.3255, -0.6816, -0.1108,  0.2946,  1.7022]), decimal=4)
        self.assertEqual(prob.shape, (2, 3263))
        pred = model.get_predictions(prob)
        self.assertEqual([len(pred), len(pred[0]), len(pred[1])], [2, 5, 5])
        txt = [[(nm, '%.6f' % pb) for nm, pb in group[:2]] for group in pred]
        self.assertEqual(txt[0], [('shirai_kuroko', '99.998939'), ('fukuzawa_yumi', '0.000314')])
        self.assertEqual(txt[1], [('misaka_mikoto', '99.951982'), ('misaka_imouto', '0.040299')])
        if EXTENDED:
            model = VitEncoder('L16_Danbooru_Faces', classify=True)
            prob, emb = model(imgs)
            self.assertEqual(emb.shape, (2, 1024))
            np.testing.assert_almost_equal(emb[0][900:905], np.array([-1.5694, 0.1522, -1.8948, -1.2867, -1.8749]), decimal=4)
            np.testing.assert_almost_equal(emb[1][175:180], np.array([0.3184, -1.2670, -0.0992, -0.0231,  0.5195]), decimal=4)
            self.assertEqual(prob.shape, (2, 3263))
            pred = model.get_predictions(prob)
            self.assertEqual([len(pred), len(pred[0]), len(pred[1])], [2, 5, 5])
            txt = [[(nm, '%.5f' % pb) for nm, pb in group[:2]] for group in pred]
            self.assertEqual(txt[0], [('shirai_kuroko', '99.97436'), ('misaka_imouto', '0.00944')])
            self.assertEqual(txt[1], [('misaka_mikoto', '99.99293'), ('misaka_imouto', '0.00635')])
            model = VitEncoder('B16_Danbooru_Full', classify=True)
            prob, _ = model(imgs)
            pred = model.get_predictions(prob)
            txt = [[(nm, '%.5f' % pb) for nm, pb in group[:2]] for group in pred]
            self.assertEqual(txt[0], [('shirai_kuroko', '96.67073'), ('sakura_kyouko', '1.81580')])
            self.assertEqual(txt[1], [('misaka_mikoto', '99.94698'), ('misaka_imouto', '0.04537')])

    def test_vit_torchvision(self):
        imgs = get_images('sml_%s.jpg', ['bear1', 'cat1', 'city1'], lib='PIL')
        model = VitEncoder('B16_ImageNet1K_TorchVision', classify=True)
        prob, _ = model(imgs)
        pred = model.get_predictions(prob)
        txt = [[(nm, '%.3f' % pb) for nm, pb in group[:3]] for group in pred]
        self.assertEqual(prob.shape, (3, 1000))
        self.assertEqual([len(pred), len(pred[0]), len(pred[1])], [3, 5, 5])
        self.assertEqual(txt[0], [('American black bear', '74.586'), ('brown bear', '18.587'), ('unicycle', '1.782')])
        self.assertEqual(txt[1], [('Egyptian cat', '50.508'), ('tabby', '31.426'), ('Persian cat', '5.010')])
        self.assertEqual(txt[2], [('steel arch bridge', '40.054'), ('planetarium', '10.477'), ('balloon', '6.173')])

    def test_vit_face(self):
        imgs = get_images('irl_face_%u.jpg', [1, 2])
        model = VitEncoder('P8S8_MSCeleb1M', classify=False)
        res = model(imgs)
        self.assertEqual(res.shape, (2, 512))
        np.testing.assert_almost_equal(res[0][485:490], np.array([1.1649, -0.4840, 1.0156, -0.7108, -0.0953]), decimal=4)
        np.testing.assert_almost_equal(res[1][250:255], np.array([0.9379, -0.4167, -0.8178, 0.4852,  0.5944]), decimal=4)

    def test_vit_clip_similarity(self):
        # images are taken from this question on similarity search: https://stackoverflow.com/a/71567609
        names = ['bear1', 'bear2', 'cat1', 'cat1copy', 'cat2', 'city1', 'city2']
        imgs = get_images('sml_%s.jpg', names, lib='PIL')
        model = VitEncoder('B32_CLIP_OpenAI')
        emb = model(imgs)
        self.assertEqual(emb.shape, (7, 512))
        np.testing.assert_almost_equal(emb[0][:8],      np.array([ 0.3134, -0.3096,  0.1799,  0.4285, -0.2289, -0.1865,  0.0012,  0.4130]), decimal=4)
        np.testing.assert_almost_equal(emb[1][100:108], np.array([-0.1534, -0.2122,  0.1828,  0.0394, -0.2925,  0.3063, -1.1866, -0.0776]), decimal=4)
        np.testing.assert_almost_equal(emb[4][260:268], np.array([-0.0787,  0.4456, -0.3024,  0.2425,  0.1895, -0.4241, -0.0121, -0.1657]), decimal=4)
        np.testing.assert_almost_equal(emb[-1][-8:],    np.array([-0.0502, -0.1492, -0.3381,  0.1884,  0.0382,  0.0586, -0.1712,  0.1970]), decimal=4)
        if EXTENDED:
            model = VitEncoder('B16_CLIP_OpenAI')
            emb = model(imgs)
            self.assertEqual(emb.shape, (7, 512))
            np.testing.assert_almost_equal(emb[0][:8],      np.array([-0.6832, -0.8956, -0.1834,  0.7935,  0.1787, -0.2087,  0.0365,  0.2310]), decimal=4)
            np.testing.assert_almost_equal(emb[1][100:108], np.array([-0.2757, -0.1250,  0.1197, -1.1228,  0.1665,  0.0114,  0.2684, -0.0327]), decimal=4)
            np.testing.assert_almost_equal(emb[4][260:268], np.array([ 0.1359, -0.7762,  0.0599, -0.3834,  0.0420,  0.0850, -0.8429,  0.3719]), decimal=4)
            np.testing.assert_almost_equal(emb[-1][-8:],    np.array([ 0.3126,  0.7126,  0.0109,  0.2893, -0.0855,  0.2969,  0.3100,  0.4065]), decimal=4)
            model = VitEncoder('L14_CLIP_OpenAI')
            emb = model(imgs)
            self.assertEqual(emb.shape, (7, 768))
            np.testing.assert_almost_equal(emb[0][:8],      np.array([ 0.1204, -0.5597,  0.1497, -0.7864,  0.2567,  1.0152,  0.1416,  0.1061]), decimal=4)
            np.testing.assert_almost_equal(emb[1][100:108], np.array([ 0.4827,  0.3201, -0.4523, -0.0289, -0.1447, -0.1372, -1.3364, -0.0777]), decimal=4)
            np.testing.assert_almost_equal(emb[4][260:268], np.array([ 0.3235, -0.4756, -0.7215, -0.5527, -0.3293, -0.2288, -0.1420, -0.6778]), decimal=4)
            np.testing.assert_almost_equal(emb[-1][-8:],    np.array([-0.3065,  0.3800,  0.1017, -0.0862,  0.4355, -0.0234,  0.4609,  0.5594]), decimal=4)

    def _test_vit_torchvision_comparison(self):
        imgs = get_images('sml_%s.jpg', ['bear1', 'cat1', 'city1'], lib='PIL')
        model = VitEncoder('B16_ImageNet1K_TorchVision', classify=True)
        prob, _ = model(imgs)
        pred = model.get_predictions(prob)
        
        from tidytorchmodels.encoders.operations.classify import predict
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        import torch
        morig = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)#.cuda()
        prep = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        inp = torch.stack([prep(im) for im in imgs])#.cuda()
        with torch.no_grad():
            logits = morig(inp)
            prob_orig = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_orig = predict('ImageNet1K', prob_orig)
        
        txt1 = [[(nm, '%.3f' % pb) for nm, pb in group[:3]] for group in pred]
        txt2 = [[(nm, '%.3f' % pb) for nm, pb in group[:3]] for group in pred_orig]
        self.assertEqual(txt1, txt2)

if __name__ == '__main__':
    if EXTENDED:
        print('EXTENDED')
    unittest.main()