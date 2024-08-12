import os

import imutils
import cv2
import argparse

from AIDetector_pytorch import Detector



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

    object_list = args.objects.split(',')
    det = Detector(yolov5_weights, object_list)
    cap = cv2.VideoCapture(args.input_video_path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    size = None
    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break

        result = det.feedCap(im, func_status)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                f'{args.save_output_path}/result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    return "INFO:Infer success!"


def parse_args():
    parser = argparse.ArgumentParser("Infer video with yolov5+deepsort.")
    parser.add_argument("--yolov5_weights", required=True, type=str, help="Input labelme dataset abs path.",
                        default=r"'weights/yolov5s.pt'")
    parser.add_argument("--objects", required=True, type=str, help="Intput object names, such like: cat,dog.",
                        default="cat,dog")
    parser.add_argument("--input_video_path", required=True, type=str, help="Intput video path.", default="")
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
