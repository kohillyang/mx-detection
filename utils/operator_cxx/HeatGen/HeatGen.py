import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class HeatGen:
    def __init__(self, stride, sigma):
        self.stride = stride
        self.sigma = sigma

    def forward(self, data, bboxes, keypoints):
        if self.req[0] == req.null:
            return
        out = self.y
        sigma = self.sigma
        h, w, c = data.shape
        nperson = len(bboxes)
        nparts = keypoints.shape[1]
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            mobula.func.heat_gen(keypoints=keypoints, h=h, w=w, stride=self.stride, nperson=nperson, nparts=nparts,
                                 sigma=sigma, output=out_temp)
            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.heat_gen(keypoints=keypoints, h=h, w=w, stride=self.stride, nperson=nperson, nparts=nparts,
                                 sigma=sigma, output=self.y)

    def backward(self):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 3  # Image Height * Image Width * 3
        assert len(in_shape[1]) == 2  # number of person * 5
        assert len(in_shape[2]) == 3  # number of person * number of parts * 3
        h, w, c = in_shape[0]
        number_of_person = in_shape[2][0]
        number_of_parts = in_shape[2][1]
        stride = self.stride
        assert h % stride == 0
        assert w % stride == 0
        # extra channel for background
        return in_shape, [(number_of_parts + 1, h // self.stride, w // self.stride)]