"""
Copyright 2022 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import time

import cv2
import glob
import json
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from ais_bench.infer.interface import InferSession

from edge_infer.det_utils import letterbox, scale_coords, nms, xyxy2xywh


class BatchDataLoader:
    def __init__(self, data_path_list, batch_size, input_shape):
        self.data_path_list = data_path_list
        self.sample_num = len(data_path_list)
        self.batch_size = batch_size
        self.input_shape = input_shape

    def __len__(self):
        return self.sample_num // self.batch_size + int(self.sample_num % self.batch_size > 0)

    def read_data(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_padded, scale_ratio, pad_size = letterbox(img_bgr, new_shape=self.input_shape)
        return img_bgr, img_padded, (scale_ratio, pad_size)

    def __getitem__(self, item):
        # form a batch
        if (item + 1) * self.batch_size <= self.sample_num:
            slice_end = (item + 1) * self.batch_size
            pad_num = 0
        else:
            slice_end = self.sample_num
            pad_num = (item + 1) * self.batch_size - self.sample_num

        original_img_list = []
        padded_img_list = []
        padding_args_list = []
        img_name_list = []
        for path in self.data_path_list[item * self.batch_size:slice_end]:
            img_name = os.path.basename(path)
            img_bgr, img_padded, padding_args = self.read_data(path)
            original_img_list.append(img_bgr)
            padded_img_list.append(img_padded)
            padding_args_list.append(padding_args)
            img_name_list.append(img_name)
        valid_num = len(padded_img_list)
        for _ in range(pad_num):
            padded_img_list.append(padded_img_list[0])
        return valid_num, img_name_list, original_img_list, np.stack(padded_img_list, axis=0), padding_args_list


def read_class_names(ground_truth_json):
    with open(ground_truth_json, 'r') as file:
        content = file.read()
    content = json.loads(content)
    categories = content.get('categories')
    names = {}
    for id, category in enumerate(categories):
        category_name = category.get('name')
        if len(category_name.split()) == 2:
            temp = category_name.split()
            category_name = temp[0] + '_' + temp[1]
        names[id] = category_name.strip('\n')
    return names


def draw_bbox(bbox, img0, color, wt, names):
    det_result_str = ''
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4] < float(0.05)):
            continue
        img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                             color, wt)
        img0 = cv2.putText(img0, str(idx) + ' ' + names[int(class_id)], (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        img0 = cv2.putText(img0, '{:.4f}'.format(bbox[idx][4]), (int(bbox[idx][0]), int(bbox[idx][1] + 32)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        det_result_str += '{} {} {} {} {} {}\n'.format(
            names[bbox[idx][5]], str(bbox[idx][4]), bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3])
    return img0


def eval(ground_truth_json, detection_results_json):
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here
    print('Start evaluate *%s* results...' % (annType))
    cocoGt_file = ground_truth_json
    cocoDt_file = detection_results_json
    cocoGt = COCO(cocoGt_file)
    try:
        cocoDt = cocoGt.loadRes(cocoDt_file)
    except IndexError:
        print('The prediction result is empty. Please check if the dataset or the training process is correct.')
        return
    imgIds = cocoGt.getImgIds()
    print('get %d images' % len(imgIds))
    imgIds = sorted(imgIds)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # copy-paste style
    eval_results = OrderedDict()
    metric = annType
    metric_items = [
        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    ]
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }

    for metric_item in metric_items:
        key = f'{metric}_{metric_item}'
        val = float(
            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
        )
        eval_results[key] = val
    ap = cocoEval.stats[:6]
    eval_results[f'{metric}_mAP_copypaste'] = (
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f}'
    )
    print(dict(eval_results))


def preprocess_img_batch(img_batch):
    # BGR to RGB, HWC to CHW
    img_batch = img_batch[..., ::-1].transpose(0, 3, 1, 2)
    img_batch = img_batch / 255.0
    img_batch = np.ascontiguousarray(img_batch).astype(np.float16)
    return img_batch


def main():
    args = parse_args()
    cfg = {
        'conf_thres': 0.4,
        'iou_thres': 0.5,
        'input_shape': [640, 640],
    }
    yaml_file = 'data.yaml'
    with open(yaml_file, errors='ignore') as f:
        yaml_data = yaml.safe_load(f)

    if args.visible:
        class_names = read_class_names(args.ground_truth_json)
        save_pred_dir = os.path.join(args.output_dir, 'img')
        os.makedirs(save_pred_dir, exist_ok=True)
    else:
        class_names = yaml_data['names']

    model = InferSession(args.device_id, args.model)

    img_path_list = glob.glob(os.path.join(args.img_path, '*.jpg'))
    dataloader = BatchDataLoader(img_path_list, args.batch_size, cfg['input_shape'])
    det_result_list = []
    total_time = 0
    category_ids = [i + 1 for i in range(len(class_names))]
    for inputs in tqdm(dataloader, total=len(dataloader)):
        valid_num, img_name_list, original_img_list, img_batch, padding_args_list = inputs
        img_batch = preprocess_img_batch(img_batch)

        infer_start = time.time()
        output = model.infer([img_batch])
        output = torch.tensor(output[0])
        boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
        total_time += time.time() - infer_start

        for idx in range(valid_num):
            pred_all = boxout[idx].numpy()
            scale_coords(cfg['input_shape'], pred_all[:, :4], original_img_list[idx].shape,
                         ratio_pad=padding_args_list[idx])

            basename = img_name_list[idx]
            # Convert to coco style.
            image_id = int(basename.split('.')[0])
            box = xyxy2xywh(pred_all[:, :4])
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for pred, bbox in zip(pred_all.tolist(), box.tolist()):
                det_result_list.append({'image_id': image_id,
                                        'category_id': category_ids[int(pred[5])],
                                        'bbox': [round(x, 3) for x in bbox],
                                        'score': round(pred[4], 5)})

            if args.visible:
                img_dw = draw_bbox(pred_all, original_img_list[idx], (0, 255, 0), 2, class_names)
                cv2.imwrite(os.path.join(save_pred_dir, basename), img_dw)

    print(f'Average infer time: {total_time * 1000 / len(dataloader):.3f} ms / {args.batch_size} image')
    print('saveing predictions.json to output/')
    pred_json_path = os.path.join(args.output_dir, 'predictions.json')
    with open(pred_json_path, 'w') as f:
        json.dump(det_result_list, f)

    if args.eval:
        eval(args.ground_truth_json, pred_json_path)


def parse_args():
    parser = argparse.ArgumentParser(description='YoloV5 offline model inference.')
    parser.add_argument('--ground_truth_json', type=str, default="test/test.json",
                        help='annotation file path')
    parser.add_argument('--img-path', type=str, default="test/images", help='input images dir')
    parser.add_argument('--model', type=str, default="output/yolov5s.om", help='om model path')
    parser.add_argument('--batch-size', type=int, default=1, help='om batch size')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    parser.add_argument('--output-dir', type=str, default='output', help='output path')
    parser.add_argument('--eval', action='store_true', help='compute mAP')
    parser.add_argument('--visible', action='store_true',
                        help='draw detect result at image and save to output/img')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
