import sys
from .utils.BaseDetector import baseDet
from v5_object_detect import non_max_suppression, letterbox, scale_coords
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


class Detector(baseDet):

    def __init__(self, model, object_list, shape, conf_threshold, iou_threshold, names):
        super(Detector, self).__init__()
        self.names = txt2list(names)
        self.class_num = len(self.names)
        self.object_list = object_list
        self.model = model
        self.build_config()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.shape = shape  # (IMG_HEIGHT, IMG_WIDTH)
        anchors = [[10, 13, 16, 30, 33, 23],
                   [30, 61, 62, 45, 59, 119],
                   [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.array(anchors, dtype=np.float32).reshape(3, -1, 2).reshape(3, 1, -1, 1, 1, 2)
        self.stride = [8, 16, 32]
        ny_nx = []
        grids = []
        for i in range(len(self.stride)):
            ny_nx.append([shape[0] // self.stride[i], shape[1] // self.stride[i]])
            xv, yv = np.meshgrid(np.arange(ny_nx[i][1]), np.arange(ny_nx[i][0]))
            grid = np.stack((xv, yv), 2).reshape((1, 1, ny_nx[i][0], ny_nx[i][1], 2)).astype(np.float16)
            grids.append(grid)
        self.grids = grids

    @ staticmethod
    def preprocess(bgr_img, shape):
        img, ratio = letterbox(bgr_img, new_shape=shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # bgr2rgb, HWC2CHW
        img = np.ascontiguousarray(img, dtype=np.float16) / 255.0
        return [img], ratio

    def postprocess(self, pred):
        z = []
        for index, p in enumerate(pred):
            n, c, h, w, k = p.shape
            x = p
            y = 1.0 / (1.0 + np.exp(-x))
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grids[index]) * self.stride[index]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[index]
            z.append(y.reshape(n, -1, k))
        return np.concatenate(z, 1)

    def detect(self, im):
        shape = im.shape
        img, ratio = self.preprocess(im, self.shape)
        output = self.model.execute(img)
        pred = self.postprocess(output)
        bbox = []
        result_return = non_max_suppression(pred, class_num=self.class_num,
                                            conf_thres=self.conf_threshold,
                                            iou_thres=self.iou_threshold)
        if len(result_return['detection_classes']):
            det = np.array(result_return['detection_boxes'])[:, :4]
            bbox = scale_coords(self.shape, det, shape, ratio)
        pred_boxes = []
        # result_return keys: 'detection_classes', 'detection_boxes', 'detection_scores'
        for det_index in range(len(result_return.get('detection_classes'))):
            lbl = self.names[int(result_return.get('detection_classes')[det_index])]
            if not lbl in self.object_list:
                continue
            x1 = int(bbox[det_index][0])
            y1 = int(bbox[det_index][1])
            x2 = int(bbox[det_index][2])
            y2 = int(bbox[det_index][3])
            pred_boxes.append((x1, y1, x2, y2, lbl, result_return.get('detection_scores')[det_index]))

        return im, pred_boxes


def txt2list(txt_path, ):
    names_list = []
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        names_list.append(line)
    return names_list

