import cv2

from ..tracker import update_tracker
from ..deep_sort.utils.parser import get_config
from ..deep_sort.deep_sort import DeepSort


class BaseDet(object):

    def __init__(self, deepsort_path):
        self.img_size = 640
        self.threshold = 0.5
        self.stride = 1

        cfg = get_config()
        cfg.merge_from_file("deepsort/deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(deepsort_path, max_dist=cfg.DEEPSORT.MAX_DIST,
                                 min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

    def build_config(self):
        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, image):
        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1

        _, bboxes = self.detect(image)
        im, faces, face_bboxes = update_tracker(self.deepsort, bboxes, image)

        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes
        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
