### Pre-trained Models
You can download pre-trained models from <https://drive.google.com/drive/folders/1LQnVHb5Xo6fKknUiOa1fXmGI_MCucGTC?usp=sharing>

|  ModelName   | Dataset  | Backbone         |with DCN | with Sync BN | Target Size | Max Size | IM_PER_IMAGE | Number of GPUs | Epochs | mAP   |
| --------     | :-----:  | :----:           |  :----: |      :----:  |      :----: |   :----: |       :----: |         :----: | :----: |:----: |
| FCOS         | COCO2017 | ResNet50         | False   |     True     |     800     |   1333   |     3        |      4         |   6    | 0.352 |
| FCOS         | COCO2017 | ResNet50         | True    |     False    |     800     |   1000   |     2        |      3         |   14   | -     |
| RetinaNet    | COCO2017 | ResNet50         | True    |     True     |     600     |   1333   |     2        |      3         |   14   | 0.324 |
| OpenPose     | COCO2017 | Dilated-ResNet50 | False   |     False    |     368     |   368    |     4        |      3         |   40   | 0.564 |
| OpenPose     | COCO2017 | VGG16            | False   |     False    |     368     |   368    |     4        |      3         |   40   | 0.561 |
| RFCN         | VOC12+07 | Dilated-ResNet101| Only 3  |     False    |     800     |   1280   |     1        |      3         |   40   | 0.825 |
| RFCN         | VOC12+07 | Dilated-ResNet50 | Only 3  |     False    |     800     |   1280   |     1        |      3         |   40   | 0.804 |
| FPN(MS)      | COCO2017 | SEResNext50_32x4d| True    |     False    |     800     |   1280   |     1        |      4         |   5    | 0.376 |
| FPN(MS)      | COCO2017 | Dilated-ResNet101| True    |     False    |     800     |   1280   |     1        |      4         |   5    | 0.412 |

Notes:<br>
RFCN trained on VOC is reported as mAP@IoU=0.5 according to VOC Metric, and it is slightly different from mAP @IoU=0.5 of COCO.

MS means the model is using multi-scaling when training.

FPN(MS) and RFCN are bought from <https://github.com/msracver/Deformable-ConvNets> and rewritten by new Gluon API,
their performance should be same with the model from <https://github.com/msracver/Deformable-ConvNets>.

### Acknowledgements

Greatly thanks to <https://github.com/wkcn/MobulaOP> by @wkcn.

If you have any question or suggestion, please feel free to send me a mail or create an issue.

### Todo List:
- [ ] Find the reason why FCOS map is lower than the official implementation.
- [ ] Train OpenPose with [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- [ ] [PolarMask](https://arxiv.org/1909.13226) 