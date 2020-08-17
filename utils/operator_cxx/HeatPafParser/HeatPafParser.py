import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class HeatPafParser:
    def __init__(self, limb_sequence, max_number_person=100):
        self.max_number_person = max_number_person
        self.limb_sequence = limb_sequence

    def forward(self, heatmap, pafmap):
        if self.req[0] == req.null:
            return
        number_of_parts_plus1, h0, w0 = heatmap.shape
        number_of_parts = number_of_parts_plus1 - 1  # one channel for background
        if self.req[0] == req.add:
            assert False
        else:
            assert self.req[0] == req.write
            self.Y[0][:] = 0
            self.Y[1][:] = 0
            mobula.func.heat_paf_parser(heatmap, pafmap, self.limb_sequence, number_of_parts,
                                        len(self.limb_sequence), w0, h0, self.max_number_person,
                                        self.Y[0], self.Y[1])

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 3  # number of parts * Image Height * Image Width
        assert len(in_shape[1]) == 3  # number of limbs * Image Height * Image Width
        number_of_parts_plus1, h0, w0 = in_shape[0]
        number_of_parts = number_of_parts_plus1 - 1
        number_of_limbs, h1, w1 = in_shape[1]
        assert h0 == h1
        assert w0 == w1

        # There are two outputs, one for keypoints(x, y, visible), and the other one is for scores.
        return in_shape, [(self.max_number_person, number_of_parts, 3), (self.max_number_person, 1)]

    def backward(self):
        pass    # nothing need to do
