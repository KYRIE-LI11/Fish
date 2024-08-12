import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from acl_net_dynamic import NetDynamic


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = NetDynamic(device_id=0, model_path=model_path)

    def _preprocess(self, im_crops):
        """
        1. to float with scale from 0 to 1
        2. resize to (64, 128) as Market1501 dataset did
        3. concatenate to a numpy array
        3. to torch Tensor
        4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            dynamic_dim = im_batch.shape[0]
            dims = {'dimCount': 4, 'name': '', 'dims': [dynamic_dim, 3, 128, 64]}
            im_batch = im_batch.cpu().numpy()
            features = self.model([im_batch], dims)
            # features = self.net(im_batch)
            return features[0]

    def __del__(self):
        del self.model


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

