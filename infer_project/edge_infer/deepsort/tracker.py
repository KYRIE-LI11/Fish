import torch
import cv2


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def update_tracker(deepsort, bboxes, image):
    new_faces = []

    bbox_xywh = []
    confs = []
    bboxes2draw = []
    face_bboxes = []
    if len(bboxes):

        # Adapt detections to deep sort input format
        for x1, y1, x2, y2, _, conf in bboxes:

            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        # Pass detections to deepsort
        outputs = deepsort.update(xywhs, confss, image)

        for value in list(outputs):
            x1,y1,x2,y2,track_id = value
            bboxes2draw.append(
                (x1, y1, x2, y2, '', track_id)
            )

    image = plot_bboxes(image, bboxes2draw)
    return image, new_faces, face_bboxes
