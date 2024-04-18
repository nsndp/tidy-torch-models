import os.path as osp
import cv2
try:
    from google.colab.patches import cv2_imshow
    NO_COLAB = False
except:
    NO_COLAB = True


def _draw_boxes(im, color, b, s, l, classes, fs, cutoff, lower=False):
    for i in range(len(b)):
        if cutoff and s is not None and s[i] < cutoff:
            continue
        x1, y1, x2, y2 = [int(c) for c in b[i]]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
        txt = []
        if s is not None:
            txt.append(str(round(s[i], 2)))
        if l is not None:
            txt.append(classes[l[i]])
        if txt:
            txt = ' '.join(txt)
            pos = (x1 if x1 > 0 else x2 - 10, y1 - 2 if y1 > 10 else y2 - 12)
            if lower:
                pos = (pos[0], pos[1] + round(30 * fs))
            cv2.putText(im, txt, pos, 0, fs, color, 1, lineType=cv2.LINE_AA)


def draw_on_image(imgpath,
                  boxes=None, scores=None, labels=None, points=None,
                  boxes2=None, scores2=None, labels2=None, points2=None,
                  colab=True, dataset=None,
                  font_scale=0.5, score_cutoff=None,
                  color1=(0, 255, 0), color2=(0, 0, 255)):
    """Shows an image using cv2 with stuff drawn on it (bounding boxes, landmarks).
    ``imgpath`` can be just a name for images from test folder.

    Usage examples:
    import cv2; from tidytorchmodels import Detector_FasterRCNN, draw_on_image

    # ANIME FACES
    imgsA = [cv2.imread('tidy-torch-models/tests/images/anime_det_%u.jpg' % u) for u in [1, 2, 3, 4]]
    modelA = Detector_FasterRCNN('mm_resnet50_animefaces')
    ba, sa, _ = modelA(imgsA)
    draw_on_image('anime_det_2.jpg', ba[1], sa[1])

    # COCO
    from tidytorchmodels.detectors.data.coco import idx_91_to_80
    imgs = [cv2.imread('tidy-torch-models/tests/images/coco_val2017_%s.jpg' % s) for s in ['000139', '455157']]
    model1 = Detector_FasterRCNN('tv_resnet50_v1')
    model2 = Detector_FasterRCNN('mm_resnet50')
    b1, s1, l1 = model1(imgs)
    b2, s2, l2 = model2(imgs)
    draw_on_image('coco_val2017_000139.jpg', b1[0], s1[0], idx_91_to_80(l1[0]) - 1, None,
                  b2[0], s2[0], l2[0], dataset='coco', score_cutoff=0.75, font_scale=0.5)

    # LANDMARKS
    from tidytorchmodels import Landmarker_MTCNN, Landmarker_CoordReg
    imgsL = [cv2.imread('tidy-torch-models/tests/images/irl_face_%u.jpg' % u) for u in [1, 2, 3, 4]]
    ps1 = Landmarker_CoordReg()(imgsL)
    ps2 = Landmarker_MTCNN()(imgsL)
    draw_on_image('irl_face_1.jpg', points=ps1[0], points2=ps2[0])
    """
    if not osp.isabs(imgpath):
        pt = osp.realpath(__file__)
        root = osp.dirname(osp.dirname(osp.dirname(osp.dirname(pt)))) # 4 levels up
        tdir = osp.join(root, 'tests', 'images')
        imgpath =  osp.join(tdir, imgpath)
    im = cv2.imread(imgpath)

    classes = None
    if dataset:
        assert dataset == 'coco'
        from ..detectors.data.coco import class_names as coco
        classes = coco

    if boxes is not None:
        _draw_boxes(im, color1, boxes, scores, labels, classes, font_scale, score_cutoff)
    if boxes2 is not None:
        _draw_boxes(im, color2, boxes2, scores2, labels2, classes, font_scale, score_cutoff, lower=True)
    if points is not None:
        for p in points:
            cv2.circle(im, p, 2, color1, 1)
    if points2 is not None:
        for p in points2:
            cv2.circle(im, p, 2, color2, 1)
    
    if not colab:
        cv2.imshow(imgpath, im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif NO_COLAB:
        raise ModuleNotFoundError('no Colab extension for cv2 display found.\
                                   If not inside Colab, call with colab=False')
    else:
        cv2_imshow(im)