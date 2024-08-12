import cv2
import numpy as np

from deepsort.utils.BaseDetector import BaseDet


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class Detector(BaseDet):
    def __init__(self, model, deepsort_path, neth, netw, object_list, shape, conf_threshold, iou_threshold, names_file_path):
        super(Detector, self).__init__(deepsort_path)
        self.names = self.txt2list(names_file_path)
        self.class_num = len(self.names)
        self.object_list = object_list  # 待跟踪的物体名称列表，字符串列表
        self.model = model
        self.neth = neth
        self.netw = netw
        self.build_config()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.shape = shape  # (IMG_HEIGHT, IMG_WIDTH)

    def txt2list(self, txt_path):
        names_list = []
        with open(txt_path) as f:
            data = f.readlines()
        for line in data:
            line = line.strip("\n")  # 去除末尾的换行符
            names_list.append(line)
        return names_list

    def detect(self, img_bgr):
        imgh, imgw = img_bgr.shape[0], img_bgr.shape[1]
        imginfo = np.array([self.neth, self.netw, imgh, imgw], dtype=np.float16)
        img_padding = letterbox(img_bgr, new_shape=(self.neth, self.netw))[0]  # padding resize bgr

        img0 = []
        img = []
        img_info = []

        img0.append(img_bgr)
        img.append(img_padding)
        img_info.append(imginfo)
        img = np.stack(img, axis=0)
        img_info = np.stack(img_info, axis=0)
        img = img[..., ::-1].transpose(0, 3, 1, 2)  # BGR tp RGB
        image_np = np.array(img, dtype=np.float32)
        image_np_expanded = image_np / 255.0
        img = np.ascontiguousarray(image_np_expanded).astype(np.float16)

        result = self.model.execute([img, imginfo])  # net out, infer time
        batch_boxout, boxnum = result

        pred_boxes = []
        idx = 0
        num_det = int(boxnum[idx][0])
        bbox = batch_boxout[idx][:num_det * 6].reshape(6, -1).transpose().astype(np.float32)  # 6xN -> Nx6

        for idx, class_id in enumerate(bbox[:, 5]):
            obj_name = self.names[int(bbox[idx][5])]
            if not obj_name in self.object_list:
                continue
            confidence = bbox[idx][4]
            if float(confidence) < self.conf_threshold:
                continue
            x1 = int(bbox[idx][0])
            y1 = int(bbox[idx][1])
            x2 = int(bbox[idx][2])
            y2 = int(bbox[idx][3])
            
            pred_boxes.append([x1, y1, x2, y2, obj_name, confidence])
                              
        return img_bgr, pred_boxes
