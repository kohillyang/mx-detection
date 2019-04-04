# mx-detection
`mx-detection` is a detection library implement based on Gluon and [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets). More detection framework will be added if I have enough time.

I have tried all my best to optimize the code readability, so, it will take you less time to implement your idea. 

I will be more than happy if someone can help me to improve it.

## Requirements
In theory any Nvidia GPUs with at least 4GB memory will be OK. but If you want to re-train the network on VOC, I suggest two 1080Ti, and on COCO2017 at least 4 GPUS will be needed.
Curently this repository only supports python2. 

Before training or validation, please run the following command.
```bash
cd utils/bbox/ && python setup_linux.py build_ext --inplace
cd utils/nms/ && python setup_linux.py build_ext --inplace
python -m pip install mxnet_cu90>=1.3.1
python -m pip install tqdm opencv-python gluoncv>=0.3.0
```


 ## Main results
 Results on VOC2007
 1. DCN+Resnet50, mAP@IoU=0.5 is 0.804<br>
 Download the pre-trained model from [OneDrive](https://pkamc-my.sharepoint.com/:u:/g/personal/by3410_office365vip_tech/EY1Ta2f54aZNklq4zjjek3wBXRy1uEMWYVhTymsPCfqvmA?e=BUbGGR)

 2. DCN+Resnet101, mAP@IoU=0.5 is 0.825<br>
 Download the pre-trained model from [OneDrive](https://pkamc-my.sharepoint.com/:u:/g/personal/by3410_office365vip_tech/EW8hZtillhNJmIiq8A4OMpsB0NdkBxNVwEfGU0TMT7qvVA?e=cVQCO9)

Results on COCO2017
1. SEResNext50_32x4d, 5 epochs, 1xlr, 4 GPUS, one image per GPU. <br>
Download the pre-trained model from [OneDrive](https://pkamc-my.sharepoint.com/:f:/g/personal/by3410_office365vip_tech/EpmNIECcTrtIk_wnf3oG4j4BAT_zynKiMptYPYxLYHCeBg?e=ZwzRmu).
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.595
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
```

## How to train
`1.` Prepare the VOC dataset and make sure the directory structure looks like this:

```
VOC2007
    Annotations
    ImageSets
    JPEGImages
    SegmentationClass
    SegmentationObject
VOC2012
    Annotations
    ImageSets
    JPEGImages
    SegmentationClass
    SegmentationObject
```
`2.` Change the item `dataset_path` in `configs/voc/resnet_v1_50_voc0712_rfcn_dcn_end2end_ohem_one_gpu.yaml`.

`3.` Just run `python2 scripts/train_voc.py`

## Differences between train_coco_fpn.py and train_voc.py
   `train_voc.py` is based on `RFCN`, and `train_coco_fpn.py` is based on FPN. Backbone of FPN return several feature maps 
   but rfcn return C4 for RPN proposal and C5 for PSROIPooling, and backbone of Faster R-CNN return C4 for RPN and ROIPooling, 
   and C5 is used as the backbone for bounding box classification and regression. In most cases, the accuracy of Faster R-CNN is 
   higher than RFCN and lower than FPN. And RFCN is faster than Faster R-CNN. But it's really case-by-case.
   
   As the outputs of the backbone are different, the rpn assign processes implemented in transformer are different and the proposal process implemented in the class `RCNNWithCriterion`
   are different. There are no other differences between the two scripts.
   
## Some useful notes.
1. Training with background image which has no positive bounding box is supported by `scripts/train_coco_fpn.py`, see the class RCNNWithCriterion
and its forward function for detail implementation. All you need to do is return a empty bbox when implementing your custom dataset.

2. `mx-detection` is not optimized for speed enough, but for code readability. If you have requirements for speed, 
you can hybridize the whole network by yourselves. Also, according to my test, tensorrt is a very useful way to speed the inference up.

3. Its very easy to change the backbone by yourselves, there are some examples like `models/fpn/resnext.py` which may be helpful.
