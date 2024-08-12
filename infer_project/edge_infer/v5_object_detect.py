import os
import numpy as np
import cv2
import argparse
from acl_resource import AclResource
from acl_model import Model
import time

INPUT_DIR = './data/'
OUTPUT_TXT_DIR = './mAP/predicted'
FPS = []
INFER = []
IMG_WIDTH = 640
IMG_HEIGHT = 640

anchors = [[10, 13, 16, 30, 33, 23],
           [30, 61, 62, 45, 59, 119],
           [116, 90, 156, 198, 373, 326]]
anchor_grid = np.array(anchors, dtype=np.float32).reshape(3, -1, 2).reshape(3, 1, -1, 1, 1, 2)
stride = [8, 16, 32]
ny_nx = []
grids = []
for i in range(len(stride)):
    ny_nx.append([IMG_HEIGHT // stride[i], IMG_WIDTH // stride[i]])
    xv, yv = np.meshgrid(np.arange(ny_nx[i][1]), np.arange(ny_nx[i][0]))
    grid = np.stack((xv, yv), 2).reshape((1, 1, ny_nx[i][0], ny_nx[i][1], 2)).astype(np.float16)
    grids.append(grid)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
    ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])  # width, height ratios

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return img, ratio


def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left


def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area


def apply_nms(all_boxes, thres, class_num):
    res = []
    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]
        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue
            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1
        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res


def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [int(index) for index in label_list]
    return label_names


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, img_shape):
    boxes[:, 0] = boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio):
    coords[:, [0, 2]] /= ratio[0]  # divide ratio
    coords[:, [1, 3]] /= ratio[1]
    clip_coords(coords, img0_shape)
    return coords


def yolov5_post(pred):
    z = []
    for index, p in enumerate(pred):
        n, c, h, w, k = p.shape
        x = p
        y = 1.0 / (1.0 + np.exp(-x))
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grids[index]) * stride[index]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[index]
        z.append(y.reshape(n, -1, k))
    return np.concatenate(z, 1)


def non_max_suppression(prediction, class_num, conf_thres=0.01, iou_thres=0.5):
    '''
        prediction shape: [batch_size, 128520, 5+class_num]    (cx, cy, w, h, conf, cls...)
    '''
    xc = prediction[..., 4] > conf_thres
    x = prediction[0][xc[0]]
    # Compute conf      conf = obj_conf * cls_conf
    x[:, 5:] *= x[:, 4:5]
    box = xywh2xyxy(x[:, :4])

    conf_ = x[:, 5:].max(1)     # 最大值
    conf = conf_[:, np.newaxis]
    j = x[:, 5:].argmax(-1)     # 最大值的位置
    j = j[:, np.newaxis]

    pred = np.concatenate((box, conf, j), 1)[conf_ > conf_thres]
    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):     # 所有选出来的box
        bbox = [int(pred[ix, iy])for iy in range(4)]
        bbox.append(int(pred[ix, 5]))
        bbox.append(pred[ix, 4])
        all_boxes[bbox[4]-1].append(bbox)   # 这个类对应的bbox

    res = apply_nms(all_boxes, iou_thres, class_num)
    result_return = dict()
    if not res:
        result_return['detection_classes'] = []
        result_return['detection_boxes'] = []
        result_return['detection_scores'] = []
        return result_return
    else:
        new_res = np.array(res)
        picked_boxes = new_res[:, 0:4]
        picked_classes = convert_labels(new_res[:, 4])
        picked_score = new_res[:, 5]
        result_return['detection_classes'] = picked_classes
        result_return['detection_boxes'] = picked_boxes.tolist()
        result_return['detection_scores'] = picked_score.tolist()
        return result_return   


def parse_args():
    parser = argparse.ArgumentParser("Convert '.pt' model into '.onnx'.")
    parser.add_argument("--model", required=True, type=str, help="Input om model path.",
                        default=r"yolov5s_bs1.om")
    parser.add_argument("--conf_threshold", required=False, type=float, help="conf_threshold.",
                        default=0.25)
    parser.add_argument("--iou_threshold", required=False, type=float, help="iou_threshold.",
                        default=0.6)
    parser.add_argument("--class_num", required=True, type=int, help="class_num.",
                        default=80)
    parser.add_argument("--output_path", required=True, type=str, help="output_path.",
                        default="./out_res")

    parser.set_defaults()
    return parser.parse_args()


def main(conf_threshold, model_path, iou_threshold, class_num, output_path):
    acl_resource = AclResource()
    acl_resource.init()
    model = Model(acl_resource, model_path)

    for pic in os.listdir(INPUT_DIR):
        pic_path = os.path.join(INPUT_DIR, pic)
        pic_name = pic.split('.')[0]

        t1 = time.time()
        bgr_img = cv2.imread(pic_path)
        img, ratio = letterbox(bgr_img, new_shape=(IMG_HEIGHT, IMG_WIDTH))
        img = img[:, :, ::-1].transpose(2, 0, 1)    # bgr2rgb, HWC2CHW
        img = np.ascontiguousarray(img, dtype=np.float16) / 255.0
        t2 = time.time()
        output = model.execute([img])

        t3 = time.time()
        pred = yolov5_post(output)
        result_return = non_max_suppression(pred, class_num, conf_thres=conf_threshold, iou_thres=iou_threshold)
        if len(result_return['detection_classes']):
            det = np.array(result_return['detection_boxes'])[:, :4]
            bbox = scale_coords((IMG_HEIGHT, IMG_WIDTH), det, bgr_img.shape, ratio)
        t4 = time.time()
        print("result: ", result_return)
        print("pre cost:{:.1f}ms".format((t2 - t1) * 1000))
        print("forward cost:{:.1f}ms".format((t3 - t2) * 1000))
        INFER.append((t3 - t2) * 1000)
        print("post cost:{:.1f}ms".format((t4 - t3) * 1000))
        print("total cost:{:.1f}ms".format((t4 - t1) * 1000))
        print("FPS:{:.1f}".format(1 / (t4 - t1)))
        FPS.append(1 / (t4 - t1))

        for i in range(len(result_return['detection_classes'])):
            box = bbox[i]
            class_name = result_return['detection_classes'][i]
            confidence = result_return['detection_scores'][i]
            cv2.rectangle(bgr_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
            p3 = (max(int(box[0]), 15), max(int(box[1]), 15))
            cv2.putText(bgr_img, "{}_{:.1f}%".format(class_name, confidence * 100), p3, cv2.FONT_ITALIC, 0.6, (255, 0, 0), 1)
        output_file = os.path.join(output_path, "out_" + pic)
        print("output: %s" % output_file)
        cv2.imwrite(output_file, bgr_img)

        predict_result_path = os.path.join(OUTPUT_TXT_DIR, str(pic_name) + '.txt')
        with open(predict_result_path, 'w') as f:
            for i in range(len(result_return['detection_classes'])):
                box = bbox[i]
                class_name = result_return['detection_classes'][i]
                confidence = result_return['detection_scores'][i]
                box = list(map(int, box))
                box = list(map(str, box))
                bbox_mess = "{} {:.4f} {} {} {} {}\n".format(class_name, confidence, box[0], box[1], box[2], box[3])
                f.write(bbox_mess)
    print(f"avg infer time {sum(INFER)/len(INFER):.3f} ms, e2e fps: {sum(FPS)/len(FPS):.3f}")
    print("Execute end")


if __name__ == '__main__':
    args = parse_args()
    conf_threshold = args.conf_threshold
    model_path = args.model
    iou_threshold = args.iou_threshold
    class_num = args.class_num
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(OUTPUT_TXT_DIR):
        os.mkdir(OUTPUT_TXT_DIR)
    main(conf_threshold, model_path, iou_threshold, class_num, output_path)
