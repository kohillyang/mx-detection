### classification on ImageNet
The model is copied from <https://github.com/HRNet/HRNet-Image-Classification> with the following two modifications:

```bash
1. The Upsample module is replaced by BilinearResize2D because there is no native support of nearest resizing in mxnet.
2. All item of HybridSequential must be a sub-class of HybridBlock, so it is not possible to put None into a
HybridSequential. I workaround this by a new class named NoneHybridBlock.
Please see models/backbones/hrnet/cls_hrnet_mx.py.
```

The keys of the original pytorch model and the converted mxnet model are almost same. So it is possible to load the original
pretrained model with a little extra codes. And because of that, I do not provide converted params. You can get the converted params
by `model.save_parameters()` if you like.

I strongly suggest validating the classification accuracy before using the converted models. In order to achieve this goal,
you need to :<br>
`1.` follow <https://github.com/pytorch/examples/tree/master/imagenet> and <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>
to prepare the imagenet validation dataset with 50k images. <br>
`2.` Download pytorch params from <https://github.com/HRNet/HRNet-Image-Classification>. <br>
`3.` Change the dataset path, config path, and params path in script [models/backbones/hrnet/cls_hrnet_mx.py](models/backbones/hrnet/cls_hrnet_mx.py). <br>
After having done these things, you can run `python models/backbones/hrnet/cls_hrnet_mx.py` directly, and you will see the Top1 acc result.

The following is my testing results:
|  ModelName   | Top1 Acc|Dataset          | config path |
| --------     | :-----:          |:-----:          |:-----:      |
| HRNetv2-W32  |  0.78246     |ImageNet val 50k |  [cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml](models/backbones/hrnet/torch_hrnet_cfgs/cls/cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml) |
