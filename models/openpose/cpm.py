import mxnet as mx
import numpy as np
import logging
from .resnet import resnet50_v1b


def get_vgg_cpm_symbol(data, number_of_parts=19, number_of_pafs=19):
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data, num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                    no_bias=False)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1, act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1, num_filter=64, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2, act_type='relu')
    pool1_stage1 = mx.symbol.Pooling(name='pool1_stage1', data=relu1_2, pooling_convention='full', pad=(0, 0),
                                     kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1_stage1, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1, act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2, act_type='relu')
    pool2_stage1 = mx.symbol.Pooling(name='pool2_stage1', data=relu2_2, pooling_convention='full', pad=(0, 0),
                                     kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2_stage1, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1, act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2, act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3, act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4, act_type='relu')
    pool3_stage1 = mx.symbol.Pooling(name='pool3_stage1', data=relu3_4, pooling_convention='full', pad=(0, 0),
                                     kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3_stage1, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1, act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2, act_type='relu')
    conv4_3_CPM = mx.symbol.Convolution(name='conv4_3_CPM', data=relu4_2, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                        stride=(1, 1), no_bias=False)
    relu4_3_CPM = mx.symbol.Activation(name='relu4_3_CPM', data=conv4_3_CPM, act_type='relu')
    conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=relu4_3_CPM, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                        stride=(1, 1), no_bias=False)
    relu4_4_CPM = mx.symbol.Activation(name='relu4_4_CPM', data=conv4_4_CPM, act_type='relu')
    conv5_1_CPM_L1 = mx.symbol.Convolution(name='conv5_1_CPM_L1', data=relu4_4_CPM, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_1_CPM_L1 = mx.symbol.Activation(name='relu5_1_CPM_L1', data=conv5_1_CPM_L1, act_type='relu')
    conv5_1_CPM_L2 = mx.symbol.Convolution(name='conv5_1_CPM_L2', data=relu4_4_CPM, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_1_CPM_L2 = mx.symbol.Activation(name='relu5_1_CPM_L2', data=conv5_1_CPM_L2, act_type='relu')
    conv5_2_CPM_L1 = mx.symbol.Convolution(name='conv5_2_CPM_L1', data=relu5_1_CPM_L1, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_2_CPM_L1 = mx.symbol.Activation(name='relu5_2_CPM_L1', data=conv5_2_CPM_L1, act_type='relu')
    conv5_2_CPM_L2 = mx.symbol.Convolution(name='conv5_2_CPM_L2', data=relu5_1_CPM_L2, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_2_CPM_L2 = mx.symbol.Activation(name='relu5_2_CPM_L2', data=conv5_2_CPM_L2, act_type='relu')
    conv5_3_CPM_L1 = mx.symbol.Convolution(name='conv5_3_CPM_L1', data=relu5_2_CPM_L1, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_3_CPM_L1 = mx.symbol.Activation(name='relu5_3_CPM_L1', data=conv5_3_CPM_L1, act_type='relu')
    conv5_3_CPM_L2 = mx.symbol.Convolution(name='conv5_3_CPM_L2', data=relu5_2_CPM_L2, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_3_CPM_L2 = mx.symbol.Activation(name='relu5_3_CPM_L2', data=conv5_3_CPM_L2, act_type='relu')
    conv5_4_CPM_L1 = mx.symbol.Convolution(name='conv5_4_CPM_L1', data=relu5_3_CPM_L1, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    relu5_4_CPM_L1 = mx.symbol.Activation(name='relu5_4_CPM_L1', data=conv5_4_CPM_L1, act_type='relu')
    conv5_4_CPM_L2 = mx.symbol.Convolution(name='conv5_4_CPM_L2', data=relu5_3_CPM_L2, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    relu5_4_CPM_L2 = mx.symbol.Activation(name='relu5_4_CPM_L2', data=conv5_4_CPM_L2, act_type='relu')
    conv5_5_CPM_L1 = mx.symbol.Convolution(name='conv5_5_CPM_L1', data=relu5_4_CPM_L1, num_filter=38, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    conv5_5_CPM_L2 = mx.symbol.Convolution(name='conv5_5_CPM_L2', data=relu5_4_CPM_L2, num_filter=19, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage2 = mx.symbol.Concat(name='concat_stage2', *[conv5_5_CPM_L1, conv5_5_CPM_L2, relu4_4_CPM])
    Mconv1_stage2_L1 = mx.symbol.Convolution(name='Mconv1_stage2_L1', data=concat_stage2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage2_L1 = mx.symbol.Activation(name='Mrelu1_stage2_L1', data=Mconv1_stage2_L1, act_type='relu')
    Mconv1_stage2_L2 = mx.symbol.Convolution(name='Mconv1_stage2_L2', data=concat_stage2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage2_L2 = mx.symbol.Activation(name='Mrelu1_stage2_L2', data=Mconv1_stage2_L2, act_type='relu')
    Mconv2_stage2_L1 = mx.symbol.Convolution(name='Mconv2_stage2_L1', data=Mrelu1_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage2_L1 = mx.symbol.Activation(name='Mrelu2_stage2_L1', data=Mconv2_stage2_L1, act_type='relu')
    Mconv2_stage2_L2 = mx.symbol.Convolution(name='Mconv2_stage2_L2', data=Mrelu1_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage2_L2 = mx.symbol.Activation(name='Mrelu2_stage2_L2', data=Mconv2_stage2_L2, act_type='relu')
    Mconv3_stage2_L1 = mx.symbol.Convolution(name='Mconv3_stage2_L1', data=Mrelu2_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage2_L1 = mx.symbol.Activation(name='Mrelu3_stage2_L1', data=Mconv3_stage2_L1, act_type='relu')
    Mconv3_stage2_L2 = mx.symbol.Convolution(name='Mconv3_stage2_L2', data=Mrelu2_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage2_L2 = mx.symbol.Activation(name='Mrelu3_stage2_L2', data=Mconv3_stage2_L2, act_type='relu')
    Mconv4_stage2_L1 = mx.symbol.Convolution(name='Mconv4_stage2_L1', data=Mrelu3_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage2_L1 = mx.symbol.Activation(name='Mrelu4_stage2_L1', data=Mconv4_stage2_L1, act_type='relu')
    Mconv4_stage2_L2 = mx.symbol.Convolution(name='Mconv4_stage2_L2', data=Mrelu3_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage2_L2 = mx.symbol.Activation(name='Mrelu4_stage2_L2', data=Mconv4_stage2_L2, act_type='relu')
    Mconv5_stage2_L1 = mx.symbol.Convolution(name='Mconv5_stage2_L1', data=Mrelu4_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage2_L1 = mx.symbol.Activation(name='Mrelu5_stage2_L1', data=Mconv5_stage2_L1, act_type='relu')
    Mconv5_stage2_L2 = mx.symbol.Convolution(name='Mconv5_stage2_L2', data=Mrelu4_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage2_L2 = mx.symbol.Activation(name='Mrelu5_stage2_L2', data=Mconv5_stage2_L2, act_type='relu')
    Mconv6_stage2_L1 = mx.symbol.Convolution(name='Mconv6_stage2_L1', data=Mrelu5_stage2_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage2_L1 = mx.symbol.Activation(name='Mrelu6_stage2_L1', data=Mconv6_stage2_L1, act_type='relu')
    Mconv6_stage2_L2 = mx.symbol.Convolution(name='Mconv6_stage2_L2', data=Mrelu5_stage2_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage2_L2 = mx.symbol.Activation(name='Mrelu6_stage2_L2', data=Mconv6_stage2_L2, act_type='relu')
    Mconv7_stage2_L1 = mx.symbol.Convolution(name='Mconv7_stage2_L1', data=Mrelu6_stage2_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage2_L2 = mx.symbol.Convolution(name='Mconv7_stage2_L2', data=Mrelu6_stage2_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage3 = mx.symbol.Concat(name='concat_stage3', *[Mconv7_stage2_L1, Mconv7_stage2_L2, relu4_4_CPM])
    Mconv1_stage3_L1 = mx.symbol.Convolution(name='Mconv1_stage3_L1', data=concat_stage3, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage3_L1 = mx.symbol.Activation(name='Mrelu1_stage3_L1', data=Mconv1_stage3_L1, act_type='relu')
    Mconv1_stage3_L2 = mx.symbol.Convolution(name='Mconv1_stage3_L2', data=concat_stage3, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage3_L2 = mx.symbol.Activation(name='Mrelu1_stage3_L2', data=Mconv1_stage3_L2, act_type='relu')
    Mconv2_stage3_L1 = mx.symbol.Convolution(name='Mconv2_stage3_L1', data=Mrelu1_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage3_L1 = mx.symbol.Activation(name='Mrelu2_stage3_L1', data=Mconv2_stage3_L1, act_type='relu')
    Mconv2_stage3_L2 = mx.symbol.Convolution(name='Mconv2_stage3_L2', data=Mrelu1_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage3_L2 = mx.symbol.Activation(name='Mrelu2_stage3_L2', data=Mconv2_stage3_L2, act_type='relu')
    Mconv3_stage3_L1 = mx.symbol.Convolution(name='Mconv3_stage3_L1', data=Mrelu2_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage3_L1 = mx.symbol.Activation(name='Mrelu3_stage3_L1', data=Mconv3_stage3_L1, act_type='relu')
    Mconv3_stage3_L2 = mx.symbol.Convolution(name='Mconv3_stage3_L2', data=Mrelu2_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage3_L2 = mx.symbol.Activation(name='Mrelu3_stage3_L2', data=Mconv3_stage3_L2, act_type='relu')
    Mconv4_stage3_L1 = mx.symbol.Convolution(name='Mconv4_stage3_L1', data=Mrelu3_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage3_L1 = mx.symbol.Activation(name='Mrelu4_stage3_L1', data=Mconv4_stage3_L1, act_type='relu')
    Mconv4_stage3_L2 = mx.symbol.Convolution(name='Mconv4_stage3_L2', data=Mrelu3_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage3_L2 = mx.symbol.Activation(name='Mrelu4_stage3_L2', data=Mconv4_stage3_L2, act_type='relu')
    Mconv5_stage3_L1 = mx.symbol.Convolution(name='Mconv5_stage3_L1', data=Mrelu4_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage3_L1 = mx.symbol.Activation(name='Mrelu5_stage3_L1', data=Mconv5_stage3_L1, act_type='relu')
    Mconv5_stage3_L2 = mx.symbol.Convolution(name='Mconv5_stage3_L2', data=Mrelu4_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage3_L2 = mx.symbol.Activation(name='Mrelu5_stage3_L2', data=Mconv5_stage3_L2, act_type='relu')
    Mconv6_stage3_L1 = mx.symbol.Convolution(name='Mconv6_stage3_L1', data=Mrelu5_stage3_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage3_L1 = mx.symbol.Activation(name='Mrelu6_stage3_L1', data=Mconv6_stage3_L1, act_type='relu')
    Mconv6_stage3_L2 = mx.symbol.Convolution(name='Mconv6_stage3_L2', data=Mrelu5_stage3_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage3_L2 = mx.symbol.Activation(name='Mrelu6_stage3_L2', data=Mconv6_stage3_L2, act_type='relu')
    Mconv7_stage3_L1 = mx.symbol.Convolution(name='Mconv7_stage3_L1', data=Mrelu6_stage3_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage3_L2 = mx.symbol.Convolution(name='Mconv7_stage3_L2', data=Mrelu6_stage3_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage4 = mx.symbol.Concat(name='concat_stage4', *[Mconv7_stage3_L1, Mconv7_stage3_L2, relu4_4_CPM])
    Mconv1_stage4_L1 = mx.symbol.Convolution(name='Mconv1_stage4_L1', data=concat_stage4, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage4_L1 = mx.symbol.Activation(name='Mrelu1_stage4_L1', data=Mconv1_stage4_L1, act_type='relu')
    Mconv1_stage4_L2 = mx.symbol.Convolution(name='Mconv1_stage4_L2', data=concat_stage4, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage4_L2 = mx.symbol.Activation(name='Mrelu1_stage4_L2', data=Mconv1_stage4_L2, act_type='relu')
    Mconv2_stage4_L1 = mx.symbol.Convolution(name='Mconv2_stage4_L1', data=Mrelu1_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage4_L1 = mx.symbol.Activation(name='Mrelu2_stage4_L1', data=Mconv2_stage4_L1, act_type='relu')
    Mconv2_stage4_L2 = mx.symbol.Convolution(name='Mconv2_stage4_L2', data=Mrelu1_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage4_L2 = mx.symbol.Activation(name='Mrelu2_stage4_L2', data=Mconv2_stage4_L2, act_type='relu')
    Mconv3_stage4_L1 = mx.symbol.Convolution(name='Mconv3_stage4_L1', data=Mrelu2_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage4_L1 = mx.symbol.Activation(name='Mrelu3_stage4_L1', data=Mconv3_stage4_L1, act_type='relu')
    Mconv3_stage4_L2 = mx.symbol.Convolution(name='Mconv3_stage4_L2', data=Mrelu2_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage4_L2 = mx.symbol.Activation(name='Mrelu3_stage4_L2', data=Mconv3_stage4_L2, act_type='relu')
    Mconv4_stage4_L1 = mx.symbol.Convolution(name='Mconv4_stage4_L1', data=Mrelu3_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage4_L1 = mx.symbol.Activation(name='Mrelu4_stage4_L1', data=Mconv4_stage4_L1, act_type='relu')
    Mconv4_stage4_L2 = mx.symbol.Convolution(name='Mconv4_stage4_L2', data=Mrelu3_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage4_L2 = mx.symbol.Activation(name='Mrelu4_stage4_L2', data=Mconv4_stage4_L2, act_type='relu')
    Mconv5_stage4_L1 = mx.symbol.Convolution(name='Mconv5_stage4_L1', data=Mrelu4_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage4_L1 = mx.symbol.Activation(name='Mrelu5_stage4_L1', data=Mconv5_stage4_L1, act_type='relu')
    Mconv5_stage4_L2 = mx.symbol.Convolution(name='Mconv5_stage4_L2', data=Mrelu4_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage4_L2 = mx.symbol.Activation(name='Mrelu5_stage4_L2', data=Mconv5_stage4_L2, act_type='relu')
    Mconv6_stage4_L1 = mx.symbol.Convolution(name='Mconv6_stage4_L1', data=Mrelu5_stage4_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage4_L1 = mx.symbol.Activation(name='Mrelu6_stage4_L1', data=Mconv6_stage4_L1, act_type='relu')
    Mconv6_stage4_L2 = mx.symbol.Convolution(name='Mconv6_stage4_L2', data=Mrelu5_stage4_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage4_L2 = mx.symbol.Activation(name='Mrelu6_stage4_L2', data=Mconv6_stage4_L2, act_type='relu')
    Mconv7_stage4_L1 = mx.symbol.Convolution(name='Mconv7_stage4_L1', data=Mrelu6_stage4_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage4_L2 = mx.symbol.Convolution(name='Mconv7_stage4_L2', data=Mrelu6_stage4_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage5 = mx.symbol.Concat(name='concat_stage5', *[Mconv7_stage4_L1, Mconv7_stage4_L2, relu4_4_CPM])
    Mconv1_stage5_L1 = mx.symbol.Convolution(name='Mconv1_stage5_L1', data=concat_stage5, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage5_L1 = mx.symbol.Activation(name='Mrelu1_stage5_L1', data=Mconv1_stage5_L1, act_type='relu')
    Mconv1_stage5_L2 = mx.symbol.Convolution(name='Mconv1_stage5_L2', data=concat_stage5, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage5_L2 = mx.symbol.Activation(name='Mrelu1_stage5_L2', data=Mconv1_stage5_L2, act_type='relu')
    Mconv2_stage5_L1 = mx.symbol.Convolution(name='Mconv2_stage5_L1', data=Mrelu1_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage5_L1 = mx.symbol.Activation(name='Mrelu2_stage5_L1', data=Mconv2_stage5_L1, act_type='relu')
    Mconv2_stage5_L2 = mx.symbol.Convolution(name='Mconv2_stage5_L2', data=Mrelu1_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage5_L2 = mx.symbol.Activation(name='Mrelu2_stage5_L2', data=Mconv2_stage5_L2, act_type='relu')
    Mconv3_stage5_L1 = mx.symbol.Convolution(name='Mconv3_stage5_L1', data=Mrelu2_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage5_L1 = mx.symbol.Activation(name='Mrelu3_stage5_L1', data=Mconv3_stage5_L1, act_type='relu')
    Mconv3_stage5_L2 = mx.symbol.Convolution(name='Mconv3_stage5_L2', data=Mrelu2_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage5_L2 = mx.symbol.Activation(name='Mrelu3_stage5_L2', data=Mconv3_stage5_L2, act_type='relu')
    Mconv4_stage5_L1 = mx.symbol.Convolution(name='Mconv4_stage5_L1', data=Mrelu3_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage5_L1 = mx.symbol.Activation(name='Mrelu4_stage5_L1', data=Mconv4_stage5_L1, act_type='relu')
    Mconv4_stage5_L2 = mx.symbol.Convolution(name='Mconv4_stage5_L2', data=Mrelu3_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage5_L2 = mx.symbol.Activation(name='Mrelu4_stage5_L2', data=Mconv4_stage5_L2, act_type='relu')
    Mconv5_stage5_L1 = mx.symbol.Convolution(name='Mconv5_stage5_L1', data=Mrelu4_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage5_L1 = mx.symbol.Activation(name='Mrelu5_stage5_L1', data=Mconv5_stage5_L1, act_type='relu')
    Mconv5_stage5_L2 = mx.symbol.Convolution(name='Mconv5_stage5_L2', data=Mrelu4_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage5_L2 = mx.symbol.Activation(name='Mrelu5_stage5_L2', data=Mconv5_stage5_L2, act_type='relu')
    Mconv6_stage5_L1 = mx.symbol.Convolution(name='Mconv6_stage5_L1', data=Mrelu5_stage5_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage5_L1 = mx.symbol.Activation(name='Mrelu6_stage5_L1', data=Mconv6_stage5_L1, act_type='relu')
    Mconv6_stage5_L2 = mx.symbol.Convolution(name='Mconv6_stage5_L2', data=Mrelu5_stage5_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage5_L2 = mx.symbol.Activation(name='Mrelu6_stage5_L2', data=Mconv6_stage5_L2, act_type='relu')
    Mconv7_stage5_L1 = mx.symbol.Convolution(name='Mconv7_stage5_L1', data=Mrelu6_stage5_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage5_L2 = mx.symbol.Convolution(name='Mconv7_stage5_L2', data=Mrelu6_stage5_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage6 = mx.symbol.Concat(name='concat_stage6', *[Mconv7_stage5_L1, Mconv7_stage5_L2, relu4_4_CPM])
    Mconv1_stage6_L1 = mx.symbol.Convolution(name='Mconv1_stage6_L1', data=concat_stage6, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage6_L1 = mx.symbol.Activation(name='Mrelu1_stage6_L1', data=Mconv1_stage6_L1, act_type='relu')
    Mconv1_stage6_L2 = mx.symbol.Convolution(name='Mconv1_stage6_L2', data=concat_stage6, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage6_L2 = mx.symbol.Activation(name='Mrelu1_stage6_L2', data=Mconv1_stage6_L2, act_type='relu')
    Mconv2_stage6_L1 = mx.symbol.Convolution(name='Mconv2_stage6_L1', data=Mrelu1_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage6_L1 = mx.symbol.Activation(name='Mrelu2_stage6_L1', data=Mconv2_stage6_L1, act_type='relu')
    Mconv2_stage6_L2 = mx.symbol.Convolution(name='Mconv2_stage6_L2', data=Mrelu1_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage6_L2 = mx.symbol.Activation(name='Mrelu2_stage6_L2', data=Mconv2_stage6_L2, act_type='relu')
    Mconv3_stage6_L1 = mx.symbol.Convolution(name='Mconv3_stage6_L1', data=Mrelu2_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage6_L1 = mx.symbol.Activation(name='Mrelu3_stage6_L1', data=Mconv3_stage6_L1, act_type='relu')
    Mconv3_stage6_L2 = mx.symbol.Convolution(name='Mconv3_stage6_L2', data=Mrelu2_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage6_L2 = mx.symbol.Activation(name='Mrelu3_stage6_L2', data=Mconv3_stage6_L2, act_type='relu')
    Mconv4_stage6_L1 = mx.symbol.Convolution(name='Mconv4_stage6_L1', data=Mrelu3_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage6_L1 = mx.symbol.Activation(name='Mrelu4_stage6_L1', data=Mconv4_stage6_L1, act_type='relu')
    Mconv4_stage6_L2 = mx.symbol.Convolution(name='Mconv4_stage6_L2', data=Mrelu3_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage6_L2 = mx.symbol.Activation(name='Mrelu4_stage6_L2', data=Mconv4_stage6_L2, act_type='relu')
    Mconv5_stage6_L1 = mx.symbol.Convolution(name='Mconv5_stage6_L1', data=Mrelu4_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage6_L1 = mx.symbol.Activation(name='Mrelu5_stage6_L1', data=Mconv5_stage6_L1, act_type='relu')
    Mconv5_stage6_L2 = mx.symbol.Convolution(name='Mconv5_stage6_L2', data=Mrelu4_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage6_L2 = mx.symbol.Activation(name='Mrelu5_stage6_L2', data=Mconv5_stage6_L2, act_type='relu')
    Mconv6_stage6_L1 = mx.symbol.Convolution(name='Mconv6_stage6_L1', data=Mrelu5_stage6_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage6_L1 = mx.symbol.Activation(name='Mrelu6_stage6_L1', data=Mconv6_stage6_L1, act_type='relu')
    Mconv6_stage6_L2 = mx.symbol.Convolution(name='Mconv6_stage6_L2', data=Mrelu5_stage6_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage6_L2 = mx.symbol.Activation(name='Mrelu6_stage6_L2', data=Mconv6_stage6_L2, act_type='relu')
    Mconv7_stage6_L1 = mx.symbol.Convolution(name='Mconv7_stage6_L1', data=Mrelu6_stage6_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage6_L2 = mx.symbol.Convolution(name='Mconv7_stage6_L2', data=Mrelu6_stage6_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)

    return mx.symbol.Group([Mconv7_stage6_L1, Mconv7_stage6_L2,
                            Mconv7_stage5_L1, Mconv7_stage5_L2,
                            Mconv7_stage4_L1, Mconv7_stage4_L2,
                            Mconv7_stage3_L1, Mconv7_stage3_L2,
                            Mconv7_stage2_L1, Mconv7_stage2_L2,
                            conv5_5_CPM_L1, conv5_5_CPM_L2])


def get_cpm_symbol(data, number_of_parts, number_of_pafs):
    # data = mx.sym.transpose(data, (0, 3, 1, 2)) /256 -0.5
    # conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data, num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
    #                                 no_bias=False)
    # relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1, act_type='relu')
    # conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1, num_filter=64, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    # relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2, act_type='relu')
    # pool1_stage1 = mx.symbol.Pooling(name='pool1_stage1', data=relu1_2, pooling_convention='full', pad=(0, 0),
    #                                  kernel=(2, 2), stride=(2, 2), pool_type='max')
    # conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1_stage1, num_filter=128, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    # relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1, act_type='relu')
    # conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1, num_filter=128, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    # relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2, act_type='relu')
    # pool2_stage1 = mx.symbol.Pooling(name='pool2_stage1', data=relu2_2, pooling_convention='full', pad=(0, 0),
    #                                  kernel=(2, 2), stride=(2, 2), pool_type='max')
    # conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2_stage1, num_filter=256, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    # relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1, act_type='relu')
    # conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1, num_filter=256, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    # relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2, act_type='relu')
    # conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2, num_filter=256, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    # relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3, act_type='relu')
    # conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3, num_filter=256, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    # relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4, act_type='relu')
    # pool3_stage1 = mx.symbol.Pooling(name='pool3_stage1', data=relu3_4, pooling_convention='full', pad=(0, 0),
    #                                  kernel=(2, 2), stride=(2, 2), pool_type='max')
    # conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3_stage1, num_filter=512, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    # relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1, act_type='relu')
    # conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1, num_filter=512, pad=(1, 1), kernel=(3, 3),
    #                                 stride=(1, 1), no_bias=False)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=data, act_type='relu')
    conv4_3_CPM = mx.symbol.Convolution(name='conv4_3_CPM', data=relu4_2, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                        stride=(1, 1), no_bias=False)
    relu4_3_CPM = mx.symbol.Activation(name='relu4_3_CPM', data=conv4_3_CPM, act_type='relu')
    conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=relu4_3_CPM, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                        stride=(1, 1), no_bias=False)
    relu4_4_CPM = mx.symbol.Activation(name='relu4_4_CPM', data=conv4_4_CPM, act_type='relu')
    conv5_1_CPM_L1 = mx.symbol.Convolution(name='conv5_1_CPM_L1', data=relu4_4_CPM, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_1_CPM_L1 = mx.symbol.Activation(name='relu5_1_CPM_L1', data=conv5_1_CPM_L1, act_type='relu')
    conv5_1_CPM_L2 = mx.symbol.Convolution(name='conv5_1_CPM_L2', data=relu4_4_CPM, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_1_CPM_L2 = mx.symbol.Activation(name='relu5_1_CPM_L2', data=conv5_1_CPM_L2, act_type='relu')
    conv5_2_CPM_L1 = mx.symbol.Convolution(name='conv5_2_CPM_L1', data=relu5_1_CPM_L1, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_2_CPM_L1 = mx.symbol.Activation(name='relu5_2_CPM_L1', data=conv5_2_CPM_L1, act_type='relu')
    conv5_2_CPM_L2 = mx.symbol.Convolution(name='conv5_2_CPM_L2', data=relu5_1_CPM_L2, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_2_CPM_L2 = mx.symbol.Activation(name='relu5_2_CPM_L2', data=conv5_2_CPM_L2, act_type='relu')
    conv5_3_CPM_L1 = mx.symbol.Convolution(name='conv5_3_CPM_L1', data=relu5_2_CPM_L1, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_3_CPM_L1 = mx.symbol.Activation(name='relu5_3_CPM_L1', data=conv5_3_CPM_L1, act_type='relu')
    conv5_3_CPM_L2 = mx.symbol.Convolution(name='conv5_3_CPM_L2', data=relu5_2_CPM_L2, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_3_CPM_L2 = mx.symbol.Activation(name='relu5_3_CPM_L2', data=conv5_3_CPM_L2, act_type='relu')
    conv5_4_CPM_L1 = mx.symbol.Convolution(name='conv5_4_CPM_L1', data=relu5_3_CPM_L1, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    relu5_4_CPM_L1 = mx.symbol.Activation(name='relu5_4_CPM_L1', data=conv5_4_CPM_L1, act_type='relu')
    conv5_4_CPM_L2 = mx.symbol.Convolution(name='conv5_4_CPM_L2', data=relu5_3_CPM_L2, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    relu5_4_CPM_L2 = mx.symbol.Activation(name='relu5_4_CPM_L2', data=conv5_4_CPM_L2, act_type='relu')
    conv5_5_CPM_L1 = mx.symbol.Convolution(name='conv5_5_CPM_L1', data=relu5_4_CPM_L1, num_filter=number_of_pafs*2, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    conv5_5_CPM_L2 = mx.symbol.Convolution(name='conv5_5_CPM_L2', data=relu5_4_CPM_L2, num_filter=number_of_parts, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage2 = mx.symbol.Concat(name='concat_stage2', *[conv5_5_CPM_L1, conv5_5_CPM_L2, relu4_4_CPM])
    Mconv1_stage2_L1 = mx.symbol.Convolution(name='Mconv1_stage2_L1', data=concat_stage2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage2_L1 = mx.symbol.Activation(name='Mrelu1_stage2_L1', data=Mconv1_stage2_L1, act_type='relu')
    Mconv1_stage2_L2 = mx.symbol.Convolution(name='Mconv1_stage2_L2', data=concat_stage2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage2_L2 = mx.symbol.Activation(name='Mrelu1_stage2_L2', data=Mconv1_stage2_L2, act_type='relu')
    Mconv2_stage2_L1 = mx.symbol.Convolution(name='Mconv2_stage2_L1', data=Mrelu1_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage2_L1 = mx.symbol.Activation(name='Mrelu2_stage2_L1', data=Mconv2_stage2_L1, act_type='relu')
    Mconv2_stage2_L2 = mx.symbol.Convolution(name='Mconv2_stage2_L2', data=Mrelu1_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage2_L2 = mx.symbol.Activation(name='Mrelu2_stage2_L2', data=Mconv2_stage2_L2, act_type='relu')
    Mconv3_stage2_L1 = mx.symbol.Convolution(name='Mconv3_stage2_L1', data=Mrelu2_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage2_L1 = mx.symbol.Activation(name='Mrelu3_stage2_L1', data=Mconv3_stage2_L1, act_type='relu')
    Mconv3_stage2_L2 = mx.symbol.Convolution(name='Mconv3_stage2_L2', data=Mrelu2_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage2_L2 = mx.symbol.Activation(name='Mrelu3_stage2_L2', data=Mconv3_stage2_L2, act_type='relu')
    Mconv4_stage2_L1 = mx.symbol.Convolution(name='Mconv4_stage2_L1', data=Mrelu3_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage2_L1 = mx.symbol.Activation(name='Mrelu4_stage2_L1', data=Mconv4_stage2_L1, act_type='relu')
    Mconv4_stage2_L2 = mx.symbol.Convolution(name='Mconv4_stage2_L2', data=Mrelu3_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage2_L2 = mx.symbol.Activation(name='Mrelu4_stage2_L2', data=Mconv4_stage2_L2, act_type='relu')
    Mconv5_stage2_L1 = mx.symbol.Convolution(name='Mconv5_stage2_L1', data=Mrelu4_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage2_L1 = mx.symbol.Activation(name='Mrelu5_stage2_L1', data=Mconv5_stage2_L1, act_type='relu')
    Mconv5_stage2_L2 = mx.symbol.Convolution(name='Mconv5_stage2_L2', data=Mrelu4_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage2_L2 = mx.symbol.Activation(name='Mrelu5_stage2_L2', data=Mconv5_stage2_L2, act_type='relu')
    Mconv6_stage2_L1 = mx.symbol.Convolution(name='Mconv6_stage2_L1', data=Mrelu5_stage2_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage2_L1 = mx.symbol.Activation(name='Mrelu6_stage2_L1', data=Mconv6_stage2_L1, act_type='relu')
    Mconv6_stage2_L2 = mx.symbol.Convolution(name='Mconv6_stage2_L2', data=Mrelu5_stage2_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage2_L2 = mx.symbol.Activation(name='Mrelu6_stage2_L2', data=Mconv6_stage2_L2, act_type='relu')
    Mconv7_stage2_L1 = mx.symbol.Convolution(name='Mconv7_stage2_L1', data=Mrelu6_stage2_L1, num_filter=number_of_pafs*2, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage2_L2 = mx.symbol.Convolution(name='Mconv7_stage2_L2', data=Mrelu6_stage2_L2, num_filter=number_of_parts, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage3 = mx.symbol.Concat(name='concat_stage3', *[Mconv7_stage2_L1, Mconv7_stage2_L2, relu4_4_CPM])
    Mconv1_stage3_L1 = mx.symbol.Convolution(name='Mconv1_stage3_L1', data=concat_stage3, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage3_L1 = mx.symbol.Activation(name='Mrelu1_stage3_L1', data=Mconv1_stage3_L1, act_type='relu')
    Mconv1_stage3_L2 = mx.symbol.Convolution(name='Mconv1_stage3_L2', data=concat_stage3, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage3_L2 = mx.symbol.Activation(name='Mrelu1_stage3_L2', data=Mconv1_stage3_L2, act_type='relu')
    Mconv2_stage3_L1 = mx.symbol.Convolution(name='Mconv2_stage3_L1', data=Mrelu1_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage3_L1 = mx.symbol.Activation(name='Mrelu2_stage3_L1', data=Mconv2_stage3_L1, act_type='relu')
    Mconv2_stage3_L2 = mx.symbol.Convolution(name='Mconv2_stage3_L2', data=Mrelu1_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage3_L2 = mx.symbol.Activation(name='Mrelu2_stage3_L2', data=Mconv2_stage3_L2, act_type='relu')
    Mconv3_stage3_L1 = mx.symbol.Convolution(name='Mconv3_stage3_L1', data=Mrelu2_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage3_L1 = mx.symbol.Activation(name='Mrelu3_stage3_L1', data=Mconv3_stage3_L1, act_type='relu')
    Mconv3_stage3_L2 = mx.symbol.Convolution(name='Mconv3_stage3_L2', data=Mrelu2_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage3_L2 = mx.symbol.Activation(name='Mrelu3_stage3_L2', data=Mconv3_stage3_L2, act_type='relu')
    Mconv4_stage3_L1 = mx.symbol.Convolution(name='Mconv4_stage3_L1', data=Mrelu3_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage3_L1 = mx.symbol.Activation(name='Mrelu4_stage3_L1', data=Mconv4_stage3_L1, act_type='relu')
    Mconv4_stage3_L2 = mx.symbol.Convolution(name='Mconv4_stage3_L2', data=Mrelu3_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage3_L2 = mx.symbol.Activation(name='Mrelu4_stage3_L2', data=Mconv4_stage3_L2, act_type='relu')
    Mconv5_stage3_L1 = mx.symbol.Convolution(name='Mconv5_stage3_L1', data=Mrelu4_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage3_L1 = mx.symbol.Activation(name='Mrelu5_stage3_L1', data=Mconv5_stage3_L1, act_type='relu')
    Mconv5_stage3_L2 = mx.symbol.Convolution(name='Mconv5_stage3_L2', data=Mrelu4_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage3_L2 = mx.symbol.Activation(name='Mrelu5_stage3_L2', data=Mconv5_stage3_L2, act_type='relu')
    Mconv6_stage3_L1 = mx.symbol.Convolution(name='Mconv6_stage3_L1', data=Mrelu5_stage3_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage3_L1 = mx.symbol.Activation(name='Mrelu6_stage3_L1', data=Mconv6_stage3_L1, act_type='relu')
    Mconv6_stage3_L2 = mx.symbol.Convolution(name='Mconv6_stage3_L2', data=Mrelu5_stage3_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage3_L2 = mx.symbol.Activation(name='Mrelu6_stage3_L2', data=Mconv6_stage3_L2, act_type='relu')
    Mconv7_stage3_L1 = mx.symbol.Convolution(name='Mconv7_stage3_L1', data=Mrelu6_stage3_L1, num_filter=number_of_pafs*2, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage3_L2 = mx.symbol.Convolution(name='Mconv7_stage3_L2', data=Mrelu6_stage3_L2, num_filter=number_of_parts, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage4 = mx.symbol.Concat(name='concat_stage4', *[Mconv7_stage3_L1, Mconv7_stage3_L2, relu4_4_CPM])
    Mconv1_stage4_L1 = mx.symbol.Convolution(name='Mconv1_stage4_L1', data=concat_stage4, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage4_L1 = mx.symbol.Activation(name='Mrelu1_stage4_L1', data=Mconv1_stage4_L1, act_type='relu')
    Mconv1_stage4_L2 = mx.symbol.Convolution(name='Mconv1_stage4_L2', data=concat_stage4, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage4_L2 = mx.symbol.Activation(name='Mrelu1_stage4_L2', data=Mconv1_stage4_L2, act_type='relu')
    Mconv2_stage4_L1 = mx.symbol.Convolution(name='Mconv2_stage4_L1', data=Mrelu1_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage4_L1 = mx.symbol.Activation(name='Mrelu2_stage4_L1', data=Mconv2_stage4_L1, act_type='relu')
    Mconv2_stage4_L2 = mx.symbol.Convolution(name='Mconv2_stage4_L2', data=Mrelu1_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage4_L2 = mx.symbol.Activation(name='Mrelu2_stage4_L2', data=Mconv2_stage4_L2, act_type='relu')
    Mconv3_stage4_L1 = mx.symbol.Convolution(name='Mconv3_stage4_L1', data=Mrelu2_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage4_L1 = mx.symbol.Activation(name='Mrelu3_stage4_L1', data=Mconv3_stage4_L1, act_type='relu')
    Mconv3_stage4_L2 = mx.symbol.Convolution(name='Mconv3_stage4_L2', data=Mrelu2_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage4_L2 = mx.symbol.Activation(name='Mrelu3_stage4_L2', data=Mconv3_stage4_L2, act_type='relu')
    Mconv4_stage4_L1 = mx.symbol.Convolution(name='Mconv4_stage4_L1', data=Mrelu3_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage4_L1 = mx.symbol.Activation(name='Mrelu4_stage4_L1', data=Mconv4_stage4_L1, act_type='relu')
    Mconv4_stage4_L2 = mx.symbol.Convolution(name='Mconv4_stage4_L2', data=Mrelu3_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage4_L2 = mx.symbol.Activation(name='Mrelu4_stage4_L2', data=Mconv4_stage4_L2, act_type='relu')
    Mconv5_stage4_L1 = mx.symbol.Convolution(name='Mconv5_stage4_L1', data=Mrelu4_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage4_L1 = mx.symbol.Activation(name='Mrelu5_stage4_L1', data=Mconv5_stage4_L1, act_type='relu')
    Mconv5_stage4_L2 = mx.symbol.Convolution(name='Mconv5_stage4_L2', data=Mrelu4_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage4_L2 = mx.symbol.Activation(name='Mrelu5_stage4_L2', data=Mconv5_stage4_L2, act_type='relu')
    Mconv6_stage4_L1 = mx.symbol.Convolution(name='Mconv6_stage4_L1', data=Mrelu5_stage4_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage4_L1 = mx.symbol.Activation(name='Mrelu6_stage4_L1', data=Mconv6_stage4_L1, act_type='relu')
    Mconv6_stage4_L2 = mx.symbol.Convolution(name='Mconv6_stage4_L2', data=Mrelu5_stage4_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage4_L2 = mx.symbol.Activation(name='Mrelu6_stage4_L2', data=Mconv6_stage4_L2, act_type='relu')
    Mconv7_stage4_L1 = mx.symbol.Convolution(name='Mconv7_stage4_L1', data=Mrelu6_stage4_L1, num_filter=number_of_pafs*2, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage4_L2 = mx.symbol.Convolution(name='Mconv7_stage4_L2', data=Mrelu6_stage4_L2, num_filter=number_of_parts, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage5 = mx.symbol.Concat(name='concat_stage5', *[Mconv7_stage4_L1, Mconv7_stage4_L2, relu4_4_CPM])
    Mconv1_stage5_L1 = mx.symbol.Convolution(name='Mconv1_stage5_L1', data=concat_stage5, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage5_L1 = mx.symbol.Activation(name='Mrelu1_stage5_L1', data=Mconv1_stage5_L1, act_type='relu')
    Mconv1_stage5_L2 = mx.symbol.Convolution(name='Mconv1_stage5_L2', data=concat_stage5, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage5_L2 = mx.symbol.Activation(name='Mrelu1_stage5_L2', data=Mconv1_stage5_L2, act_type='relu')
    Mconv2_stage5_L1 = mx.symbol.Convolution(name='Mconv2_stage5_L1', data=Mrelu1_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage5_L1 = mx.symbol.Activation(name='Mrelu2_stage5_L1', data=Mconv2_stage5_L1, act_type='relu')
    Mconv2_stage5_L2 = mx.symbol.Convolution(name='Mconv2_stage5_L2', data=Mrelu1_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage5_L2 = mx.symbol.Activation(name='Mrelu2_stage5_L2', data=Mconv2_stage5_L2, act_type='relu')
    Mconv3_stage5_L1 = mx.symbol.Convolution(name='Mconv3_stage5_L1', data=Mrelu2_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage5_L1 = mx.symbol.Activation(name='Mrelu3_stage5_L1', data=Mconv3_stage5_L1, act_type='relu')
    Mconv3_stage5_L2 = mx.symbol.Convolution(name='Mconv3_stage5_L2', data=Mrelu2_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage5_L2 = mx.symbol.Activation(name='Mrelu3_stage5_L2', data=Mconv3_stage5_L2, act_type='relu')
    Mconv4_stage5_L1 = mx.symbol.Convolution(name='Mconv4_stage5_L1', data=Mrelu3_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage5_L1 = mx.symbol.Activation(name='Mrelu4_stage5_L1', data=Mconv4_stage5_L1, act_type='relu')
    Mconv4_stage5_L2 = mx.symbol.Convolution(name='Mconv4_stage5_L2', data=Mrelu3_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage5_L2 = mx.symbol.Activation(name='Mrelu4_stage5_L2', data=Mconv4_stage5_L2, act_type='relu')
    Mconv5_stage5_L1 = mx.symbol.Convolution(name='Mconv5_stage5_L1', data=Mrelu4_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage5_L1 = mx.symbol.Activation(name='Mrelu5_stage5_L1', data=Mconv5_stage5_L1, act_type='relu')
    Mconv5_stage5_L2 = mx.symbol.Convolution(name='Mconv5_stage5_L2', data=Mrelu4_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage5_L2 = mx.symbol.Activation(name='Mrelu5_stage5_L2', data=Mconv5_stage5_L2, act_type='relu')
    Mconv6_stage5_L1 = mx.symbol.Convolution(name='Mconv6_stage5_L1', data=Mrelu5_stage5_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage5_L1 = mx.symbol.Activation(name='Mrelu6_stage5_L1', data=Mconv6_stage5_L1, act_type='relu')
    Mconv6_stage5_L2 = mx.symbol.Convolution(name='Mconv6_stage5_L2', data=Mrelu5_stage5_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage5_L2 = mx.symbol.Activation(name='Mrelu6_stage5_L2', data=Mconv6_stage5_L2, act_type='relu')
    Mconv7_stage5_L1 = mx.symbol.Convolution(name='Mconv7_stage5_L1', data=Mrelu6_stage5_L1, num_filter=number_of_pafs*2, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage5_L2 = mx.symbol.Convolution(name='Mconv7_stage5_L2', data=Mrelu6_stage5_L2, num_filter=number_of_parts, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage6 = mx.symbol.Concat(name='concat_stage6', *[Mconv7_stage5_L1, Mconv7_stage5_L2, relu4_4_CPM])
    Mconv1_stage6_L1 = mx.symbol.Convolution(name='Mconv1_stage6_L1', data=concat_stage6, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage6_L1 = mx.symbol.Activation(name='Mrelu1_stage6_L1', data=Mconv1_stage6_L1, act_type='relu')
    Mconv1_stage6_L2 = mx.symbol.Convolution(name='Mconv1_stage6_L2', data=concat_stage6, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage6_L2 = mx.symbol.Activation(name='Mrelu1_stage6_L2', data=Mconv1_stage6_L2, act_type='relu')
    Mconv2_stage6_L1 = mx.symbol.Convolution(name='Mconv2_stage6_L1', data=Mrelu1_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage6_L1 = mx.symbol.Activation(name='Mrelu2_stage6_L1', data=Mconv2_stage6_L1, act_type='relu')
    Mconv2_stage6_L2 = mx.symbol.Convolution(name='Mconv2_stage6_L2', data=Mrelu1_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage6_L2 = mx.symbol.Activation(name='Mrelu2_stage6_L2', data=Mconv2_stage6_L2, act_type='relu')
    Mconv3_stage6_L1 = mx.symbol.Convolution(name='Mconv3_stage6_L1', data=Mrelu2_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage6_L1 = mx.symbol.Activation(name='Mrelu3_stage6_L1', data=Mconv3_stage6_L1, act_type='relu')
    Mconv3_stage6_L2 = mx.symbol.Convolution(name='Mconv3_stage6_L2', data=Mrelu2_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage6_L2 = mx.symbol.Activation(name='Mrelu3_stage6_L2', data=Mconv3_stage6_L2, act_type='relu')
    Mconv4_stage6_L1 = mx.symbol.Convolution(name='Mconv4_stage6_L1', data=Mrelu3_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage6_L1 = mx.symbol.Activation(name='Mrelu4_stage6_L1', data=Mconv4_stage6_L1, act_type='relu')
    Mconv4_stage6_L2 = mx.symbol.Convolution(name='Mconv4_stage6_L2', data=Mrelu3_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage6_L2 = mx.symbol.Activation(name='Mrelu4_stage6_L2', data=Mconv4_stage6_L2, act_type='relu')
    Mconv5_stage6_L1 = mx.symbol.Convolution(name='Mconv5_stage6_L1', data=Mrelu4_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage6_L1 = mx.symbol.Activation(name='Mrelu5_stage6_L1', data=Mconv5_stage6_L1, act_type='relu')
    Mconv5_stage6_L2 = mx.symbol.Convolution(name='Mconv5_stage6_L2', data=Mrelu4_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage6_L2 = mx.symbol.Activation(name='Mrelu5_stage6_L2', data=Mconv5_stage6_L2, act_type='relu')
    Mconv6_stage6_L1 = mx.symbol.Convolution(name='Mconv6_stage6_L1', data=Mrelu5_stage6_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage6_L1 = mx.symbol.Activation(name='Mrelu6_stage6_L1', data=Mconv6_stage6_L1, act_type='relu')
    Mconv6_stage6_L2 = mx.symbol.Convolution(name='Mconv6_stage6_L2', data=Mrelu5_stage6_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage6_L2 = mx.symbol.Activation(name='Mrelu6_stage6_L2', data=Mconv6_stage6_L2, act_type='relu')
    Mconv7_stage6_L1 = mx.symbol.Convolution(name='Mconv7_stage6_L1', data=Mrelu6_stage6_L1, num_filter=number_of_pafs*2, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage6_L2 = mx.symbol.Convolution(name='Mconv7_stage6_L2', data=Mrelu6_stage6_L2, num_filter=number_of_parts, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)

    return mx.symbol.Group([Mconv7_stage6_L1, Mconv7_stage6_L2,
                            Mconv7_stage5_L1, Mconv7_stage5_L2,
                            Mconv7_stage4_L1, Mconv7_stage4_L2,
                            Mconv7_stage3_L1, Mconv7_stage3_L2,
                            Mconv7_stage2_L1, Mconv7_stage2_L2,
                            conv5_5_CPM_L1, conv5_5_CPM_L2])


class CPMMobileNet(mx.gluon.nn.HybridBlock):
    def __init__(self, number_of_parts, number_of_pafs, resize=False):
        super(CPMMobileNet, self).__init__()
        inputs = mx.sym.var(name="mobilenet_outputs")
        sym = get_cpm_symbol(inputs, number_of_parts, number_of_pafs)
        from models.backbones.dilate_mobilenet._mobilenet import get_mobilenet
        self.feat = get_mobilenet(multiplier=1.0, pretrained=True)
        self.cpm_head = mx.gluon.SymbolBlock(sym, inputs)

        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())
        self._resize = resize

    def hybrid_forward(self, F, x, mean, std=None):
        input = F.transpose(x, (0, 3, 1, 2))
        x = input / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        res5 = self.feat(x)
        cpmhead = self.cpm_head(res5)
        if self._resize:
            return [F.contrib.BilinearResize2D(x, like=input, mode="like") for x in cpmhead]
        else:
            return cpmhead

class CPMNet(mx.gluon.nn.HybridBlock):
    def __init__(self, number_of_parts, number_of_pafs, resize=False):
        super(CPMNet, self).__init__()
        inputs = mx.sym.var(name="resnet50_outputs")
        sym = get_cpm_symbol(inputs, number_of_parts, number_of_pafs)
        self.feat = resnet50_v1b(dilated=True, pretrained=True)
        self.cpm_head = mx.gluon.SymbolBlock(sym, inputs)

        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())
        self._resize = resize

    def hybrid_forward(self, F, x, mean, std=None):
        input = F.transpose(x, (0, 3, 1, 2))
        x = input / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        x = self.feat.conv1(x)
        x = self.feat.bn1(x)
        x = self.feat.relu(x)
        x = self.feat.maxpool(x)

        res2 = self.feat.layer1(x)
        res3 = self.feat.layer2(res2)
        res4 = self.feat.layer3(res3)
        res5 = self.feat.layer4(res4)
        cpmhead = self.cpm_head(res5)
        if self._resize:
            return [F.contrib.BilinearResize2D(x, like=input, mode="like") for x in cpmhead]
        else:
            return cpmhead


class ResNet(mx.gluon.nn.HybridBlock):
    def __init__(self, number_of_parts, number_of_pafs, resize=False):
        super(ResNet, self).__init__()
        bias_init = mx.init.Constant(-5)
        self.feat = resnet50_v1b(dilated=True, pretrained=True, bias_init=bias_init)
        self.last_conv = mx.gluon.nn.Conv2D(channels=number_of_pafs+number_of_parts, kernel_size=1, padding=0)
        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())
        self._resize = resize

    def hybrid_forward(self, F, x, mean, std=None):
        input = F.transpose(x, (0, 3, 1, 2))
        x = input / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        x = self.feat.conv1(x)
        x = self.feat.bn1(x)
        x = self.feat.relu(x)
        x = self.feat.maxpool(x)

        res2 = self.feat.layer1(x)
        res3 = self.feat.layer2(res2)
        res4 = self.feat.layer3(res3)
        res5 = self.feat.layer4(res4)
        cpmhead = self.last_conv(res5)
        if self._resize:
            return F.contrib.BilinearResize2D(cpmhead, like=input, mode="like")
        else:
            return cpmhead


class CPMVGGNet(mx.gluon.nn.HybridBlock):
    def __init__(self, resize=False, pretrained=True):
        super(CPMVGGNet, self).__init__()
        inputs = mx.sym.var(name="data")
        sym = get_vgg_cpm_symbol(inputs, number_of_parts=19, number_of_pafs=19)
        self.cpm_head = mx.gluon.SymbolBlock(sym, inputs)
        params_head = self.cpm_head.collect_params()

        for p_name in params_head.keys():
            if "Mconv" in p_name:
                if p_name.endswith(('_bias')):
                    params_head[p_name].lr_mult = 8
                    logging.info("set {}'s lr_mult to {}.".format(p_name, params_head[p_name].lr_mult))
                if p_name.endswith(('_weight')):
                    params_head[p_name].lr_mult = 4
                    logging.info("set {}'s lr_mult to {}.".format(p_name, params_head[p_name].lr_mult))
            else:
                if p_name.endswith(('_bias')):
                    params_head[p_name].lr_mult = 2
                    logging.info("set {}'s lr_mult to {}.".format(p_name, params_head[p_name].lr_mult))
                if p_name.endswith(('_weight')):
                    params_head[p_name].lr_mult = 1
                    logging.info("set {}'s lr_mult to {}.".format(p_name, params_head[p_name].lr_mult))

        if pretrained:
            net_params_pretrained = np.load("pretrained/caffe_vgg_IR.npy", allow_pickle=True).item()
            for key in net_params_pretrained.keys():
                if key + "_weight" in params_head:
                    params_head[key + "_weight"]._load_init(mx.nd.array(net_params_pretrained[key]["weights"]).transpose((3, 2, 0, 1)), ctx=mx.cpu())
                    params_head[key + "_bias"]._load_init(mx.nd.array(net_params_pretrained[key]["bias"]), ctx=mx.cpu())
                    logging.info("loaded {} from pretrained model.".format(key))
                else:
                    logging.info("extra param {} is ignored.".format(key))
            logging.info("loading pretrained model finished.")
        self._resize = resize

    def hybrid_forward(self, F, x):
        input = F.transpose(x, (0, 3, 1, 2))
        input = input / 255.0 -0.5
        cpmhead = self.cpm_head(input)
        if self._resize:
            return [F.contrib.BilinearResize2D(x, like=input, mode="like") for x in cpmhead]
        else:
            return cpmhead