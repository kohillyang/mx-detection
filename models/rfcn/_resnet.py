# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument,missing-docstring
"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['ResNetV1',
           'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
           'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
           'se_resnet18_v1', 'se_resnet34_v1', 'se_resnet50_v1',
           'se_resnet101_v1', 'se_resnet152_v1',
           'se_resnet18_v2', 'se_resnet34_v2', 'se_resnet50_v2',
           'se_resnet101_v2', 'se_resnet152_v2',
           'get_resnet']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
import mxnet as mx


class DeformabelConv2D(nn.HybridBlock):
    def __init__(self, *args, **kwargs):
        super(DeformabelConv2D, self).__init__()
        init = mx.init.Zero()
        init.set_verbosity(True)
        with self.name_scope():
            self.conv_offset = nn.Conv2D(channels=72, kernel_size=3, strides=1, dilation=2, padding=2,
                                         weight_initializer=init, bias_initializer=init, prefix="offset_")

            init = mx.init.Normal()
            init.set_verbosity(True)
            self.weight = self.params.get('weight', shape=(512, 0, 3, 3),
                                          init=init,
                                          allow_deferred_init=True)

    def hybrid_forward(self, F, x, weight):
        offset = self.conv_offset(x)
        data = F.contrib.DeformableConvolution(name='fwd',
                                               data=x,
                                               offset=offset,
                                               num_filter=512, pad=(2, 2), kernel=(3, 3),
                                               num_deformable_group=4,
                                               stride=(1, 1), dilate=(2, 2), no_bias=True, weight=weight)
        return data


# Helpers
def _conv3x3(channels, stride, in_channels, use_dcn):
    if use_dcn:
        return DeformabelConv2D(channels, kernel_size=3, strides=stride, padding=2, dilation=2,
                     use_bias=False, in_channels=in_channels)
    else:
        return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1, dilation=1,
                     use_bias=False, in_channels=in_channels)

class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, use_dcn=False, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride))
        self.body.add(nn.BatchNorm(use_global_stats=True))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, 1, channels//4, use_dcn=use_dcn))
        self.body.add(nn.BatchNorm(use_global_stats=True))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels * 4, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if not last_gamma:
            self.body.add(nn.BatchNorm(use_global_stats=True))
        else:
            self.body.add(nn.BatchNorm(gamma_initializer='zeros', use_global_stats=True))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm(use_global_stats=True))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x


# Nets
class ResNetV1(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm(use_global_stats=True))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 or i == 3 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   use_dcn=i == 3))
            # self.features.add(nn.GlobalAvgPool2D())
            #
            # self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, use_dcn=False):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='', use_dcn=use_dcn))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='', use_dcn=use_dcn))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        # x = self.output(x)

        return x

# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, None]
resnet_block_versions = [{'basic_block': None, 'bottle_neck': BottleneckV1},
                         {'basic_block': None, 'bottle_neck': None}]


# Constructor
def get_resnet(version, num_layers, pretrained=False, ctx=cpu(),
               root='~/.mxnet/models', use_se=False, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        if not use_se:
            net.load_parameters(get_model_file('resnet%d_v%d' % (num_layers, version),
                                               tag=pretrained, root=root), ctx=ctx,
                                ignore_extra=True, allow_missing=True)
        else:
            net.load_parameters(get_model_file('se_resnet%d_v%d'%(num_layers, version),
                                               tag=pretrained, root=root), ctx=ctx)
        from gluoncv.data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net

def resnet18_v1(**kwargs):
    r"""ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 18, use_se=False, **kwargs)

def resnet34_v1(**kwargs):
    r"""ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 34, use_se=False, **kwargs)

def resnet50_v1(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 50, use_se=False, **kwargs)

def resnet101_v1(**kwargs):
    r"""ResNet-101 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 101, use_se=False, **kwargs)

def resnet152_v1(**kwargs):
    r"""ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 152, use_se=False, **kwargs)

def resnet18_v2(**kwargs):
    r"""ResNet-18 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 18, use_se=False, **kwargs)

def resnet34_v2(**kwargs):
    r"""ResNet-34 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 34, use_se=False, **kwargs)

def resnet50_v2(**kwargs):
    r"""ResNet-50 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 50, use_se=False, **kwargs)

def resnet101_v2(**kwargs):
    r"""ResNet-101 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 101, use_se=False, **kwargs)

def resnet152_v2(**kwargs):
    r"""ResNet-152 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 152, use_se=False, **kwargs)

# SE-ResNet
def se_resnet18_v1(**kwargs):
    r"""SE-ResNet-18 V1 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 18, use_se=True, **kwargs)

def se_resnet34_v1(**kwargs):
    r"""SE-ResNet-34 V1 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 34, use_se=True, **kwargs)

def se_resnet50_v1(**kwargs):
    r"""SE-ResNet-50 V1 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 50, use_se=True, **kwargs)

def se_resnet101_v1(**kwargs):
    r"""SE-ResNet-101 V1 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 101, use_se=True, **kwargs)

def se_resnet152_v1(**kwargs):
    r"""SE-ResNet-152 V1 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(1, 152, use_se=True, **kwargs)

def se_resnet18_v2(**kwargs):
    r"""SE-ResNet-18 V2 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 18, use_se=True, **kwargs)

def se_resnet34_v2(**kwargs):
    r"""SE-ResNet-34 V2 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 34, use_se=True, **kwargs)

def se_resnet50_v2(**kwargs):
    r"""SE-ResNet-50 V2 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 50, use_se=True, **kwargs)

def se_resnet101_v2(**kwargs):
    r"""SE-ResNet-101 V2 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 101, use_se=True, **kwargs)

def se_resnet152_v2(**kwargs):
    r"""SE-ResNet-152 V2 model from `"Squeeze-and-Excitation Networks"
    <https://arxiv.org/abs/1709.01507>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(2, 152, use_se=True, **kwargs)
