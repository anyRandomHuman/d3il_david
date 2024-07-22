import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch

cup_pred_class = 41
stacked_3cups_class = 39
BOX_FEATURE = "box"
MASK_FEATURE = "mask"


class Object_Detector:
    def __init__(
        self,
        is_joint,
        feature_type,
        obj_classes=[cup_pred_class, stacked_3cups_class],
        cfg="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    ):
        self.cfg = get_cfg()
        self.is_joint = is_joint
        self.feature_type = feature_type
        self.obj_classes = obj_classes
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg)
        self.predictor = DefaultPredictor(self.cfg)

    def extract_feature(self, img):
        outputs = self.predictor(img)
        v = Visualizer(
            img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1
        )
        instances = outputs["instances"]

        is_obj = torch.zeros(len(instances)).cuda()

        print(f'all classes identified:{outputs["instances"].pred_classes}')

        for c in self.obj_classes:
            is_obj_c = torch.where(outputs["instances"].pred_classes == c, 1, 0)
            is_obj = torch.logical_or(is_obj, is_obj_c)

        obj_indices = torch.nonzero(is_obj).squeeze()

        if self.feature_type == BOX_FEATURE:
            box_bounds = instances.pred_boxes.tensor
            features = torch.zeros((len(instances),) + img.shape[:2]).cuda()
            for i in range(len(instances)):
                t = box_bounds[i].to(dtype=torch.long)
                features[i, t[1] : t[3], t[0] : t[2]] = 1
        elif self.feature_type == MASK_FEATURE:
            features = instances.pred_masks

        obj_features = (
            (torch.index_select(features, 0, obj_indices)).permute((1, 2, 0)).cpu()
        )

        return obj_features

    def joint_feature(self, features):
        joint_mask = np.zeros(features.shape[:-1])
        for i in range(features.shape[-1]):
            joint_mask = np.logical_or(joint_mask, features[:, :, i])
        return joint_mask

    def get_feature(self, img):
        features = self.extract_feature(img)
        if self.is_joint:
            features = self.joint_feature(features)
        return features

    def get_features(self, imgs):
        fs = []
        for img in imgs:
            fs.append(self.get_feature(img))
        return fs


if __name__ == "__main__":
    img = cv2.imread("moving_cup.jpg")

    obj_det = Object_Detector(True, MASK_FEATURE, obj_classes=[41, 39])

    mask = obj_det.get_feature(img)
    mask = np.expand_dims(mask, -1)
    mask = np.repeat(mask, 3, axis=-1)
    cv2.imshow("1", np.where(mask, img, np.zeros(img.shape)))
    cv2.waitKey(0)
