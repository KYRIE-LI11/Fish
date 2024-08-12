import os
import imutils
import cv2
import argparse
from acl_resource import AclResource

acl_resource = AclResource()
acl_resource.init()

from deepsort.AIDetector_pytorch import Detector
from acl_model import Model


def main(args):
    func_status = {}
    func_status['headpose'] = None

    name = 'video_track'
    yolov5_weights = args.yolov5_weights
    if not os.path.isfile(yolov5_weights):
        return 'ERROR: Input yolov5_weights path not exist.'
    if not os.path.isfile(args.input_video_path):
        return 'ERROR: Input video path not exist.'
    if os.path.exists(args.save_output_path):
        return 'ERROR: output_path already exist.'
    os.mkdir(args.save_output_path)
    shape = (640, 640)
    object_list = args.track_objects.split(',')
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    names = args.names

    model_det = Model(acl_resource, yolov5_weights)
    det = Detector(model_det, object_list, shape, conf_threshold, iou_threshold, names)
    cap = cv2.VideoCapture(args.input_video_path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    size = None
    videoWriter = None
    while True:
        _, im = cap.read()
        if im is None:
            break

        result = det.feedCap(im, func_status)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                f'{args.save_output_path}/result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        # cv2.imwrite('./1.jpg', result)

    cap.release()
    videoWriter.release()
    return "INFO:Infer success!"


def parse_args():
    parser = argparse.ArgumentParser("Infer video with yolov5+deepsort.")
    parser.add_argument("--yolov5_weights", required=True, type=str, help="Input labelme dataset abs path.",
                        default=r"'weights/yolov5s.pt'")
    parser.add_argument("--track_objects", required=True, type=str, help="Intput object names, such like: cat,dog.",
                        default="cat,dog")
    parser.add_argument("--input_video_path", required=True, type=str, help="Intput video path.", default="")
    parser.add_argument("--conf_threshold", required=False, type=float, help="conf_threshold.", default=0.4)
    parser.add_argument("--iou_threshold", required=False, type=float, help="iou_threshold.", default=0.5)
    parser.add_argument("--names", required=False, type=str, help="iou_threshold.", default='./names.txt')
    parser.add_argument("--save_output_path", required=False, type=str, help="Output video path.", default="./output")

    parser.set_defaults()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        ret = main(args)
        print(ret)
    except Exception as e:
        print(f"ERROR:{e}")

        
'''

--yolov5_weights=/home/HwHiAiUser/sample/yolov5/output/yolov5s_v6.1_sim_nms_bs1.om \
--track_objects=people,cat,dog
--input_video_path=/home/HwHiAiUser/sample/dataset/video/video-result.mp4

sample/yolov5/common/video-result.mp4


'''