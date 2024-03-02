from torch import nn

from models.experimental import attempt_load
from utils.general import non_max_suppression


class DirectMHPInfer(nn.Module):
    """
    Predictions: (Array[N, 9]), x1, y1, x2, y2, conf, class, pitch, yaw, roll
    """

    def __init__(self, weights, device, conf_threshold=0.7, iou_threshold=0.45):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = attempt_load(weights, map_location=device)

    def forward(self, x):
        return non_max_suppression(self.model(x, augment=True)[0], conf_thres=self.conf_threshold,
                                   iou_thres=self.iou_threshold)
