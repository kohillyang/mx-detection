import mobula
import numpy as np

'''
This folder contains an unofficial implementation of probabilistic anchor assignment (PAA).
Please see <https://arxiv.org/pdf/2007.08103.pdf> for more information.

Instead of smooth L1 Loss, according to the PAA paper, we adopt DIoU loss for bounding box regression, 
please see <https://arxiv.org/abs/1911.08287> for more information. 
'''
@mobula.op.register
class PAALoss(object):
    def __init__(self, number_of_gt_boxes,
                 base_sizes=((32, 32), (64, 64), (128, 128), (256, 256), (512, 512)),
                 ratios=(.5, 1, 2),
                 scales=(2**0, 2**(1.0/3), 2**(2.0/3)),
                 strides=(8, 16, 32, 64, 128)):
        self.number_of_gt_boxes = number_of_gt_boxes
        self.base_sizes = base_sizes
        self.ratios = ratios
        self.scales = scales
        self.strides = strides
        assert len(base_sizes) == len(strides)

    def forward(self, data, pred_cls_0, pred_cls_1, pred_cls_2, pred_cls_3, pred_cls_4,
                pred_reg_0, pred_reg_1, pred_reg_2, pred_reg_3, pred_reg_4):
        # pred_reg_x are the outputs of the regression heads.
        # data is used to determine the shape of each prediction
        anchors_base_whs = []
        for base_size in self.base_sizes:
            anchors = [[(s * np.sqrt(r), s * np.sqrt(1/r)) for s in self.scales] for r in self.ratios]
            anchors_base_wh = np.array(anchors) * np.array(base_size)[np.newaxis, np.newaxis, :]
            anchors_base_wh = anchors_base_wh.reshape(-1, 2)
            anchors_base_whs.append(anchors_base_wh)
        anchors_base_whs = np.array(anchors_base_whs)
        mobula.op.paa_loss_forward(pred_cls_0, pred_cls_1, pred_cls_2, pred_cls_3, pred_cls_4,
                                   pred_reg_0, pred_reg_1, pred_reg_2, pred_reg_3, pred_reg_4,
                                   anchors_base_whs,
                                   anchors_base_whs.shape[0],  # same as number of fpn_layers.
                                   anchors_base_whs[1],        # same as number of anchors.
                                   self.strides,
                                   )

    def backward(self, y0, y1):
        pass

    def infer_shape(self, in_shape):
        return in_shape, [(5, 1), (10, 4)]