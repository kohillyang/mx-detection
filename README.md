### Pre-trained Models
You can download pre-trained models from <https://drive.google.com/drive/folders/1LQnVHb5Xo6fKknUiOa1fXmGI_MCucGTC?usp=sharing>

|  ModelName   | Dataset  | Backbone         | mAP   | with DCN | with Sync BN | Target Size | Max Size | IM_PER_IMAGE | Number of GPUs | Epochs |
| --------     | :-----:  | :----:           |:----: |   :----: |      :----:  |      :----: |   :----: |       :----: |         :----: | :----: |
|FCOS No Tricks| COCO2017 | ResNet50         | 0.367 |  False   |     False    |     800     |   1333   |     4        |      4         |   6    |
| FCOS         | COCO2017 | ResNet50         | -     |  True     |     False    |     800     |   1000   |     2        |      3         |   14   |
| HRNet-cls    | -        | -                | See [HRNet](docs/hrnet.md)|  -    |     -    |     -     |   -   |     -        |      -         |   -   |
| RetinaNet    | COCO2017 | ResNet50         | 0.324 |  True    |     True     |     600     |   1333   |     2        |      3         |   14   |
| OpenPose     | COCO2017 | Dilated-ResNet50 | 0.564 |  False   |     False    |     368     |   368    |     4        |      3         |   40   |
| OpenPose     | COCO2017 | VGG16            | 0.561 |  False   |     False    |     368     |   368    |     4        |      3         |   40   |
| RFCN         | VOC12+07 | Dilated-ResNet101| 0.825 |  Only 3  |     False    |     800     |   1280   |     1        |      3         |   6   |
| RFCN         | VOC12+07 | Dilated-ResNet50 | 0.804 |  Only 3  |     False    |     800     |   1280   |     1        |      3         |   6   |
| FPN(MS)      | COCO2017 | SEResNext50_32x4d| 0.376 |  True    |     False    |     800     |   1280   |     1        |      4         |   5    |
| FPN(MS)      | COCO2017 | Dilated-ResNet101| 0.412 |  True    |     False    |     800     |   1280   |     1        |      4         |   5    |

Notes:<br>
FCOS No Tricks means the setting is same as original paper, i.e., centerness is on cls branch, GN is added, use P5 instead of C5,
and other setting like norm_on_bbox, centerness_on_reg, center_sampling is set to False. The mAP reported by the original paper is 0.371.
For more information about FCOS, please see [fcos.md](docs/fcos.md)

RFCN trained on VOC is reported as mAP@IoU=0.5 according to VOC Metric, and it is slightly different from mAP @IoU=0.5 of COCO.

MS means the model is using multi-scaling when training.

FPN(MS) and RFCN are bought from <https://github.com/msracver/Deformable-ConvNets> and rewritten by new Gluon API,
their performance should be same with the model from <https://github.com/msracver/Deformable-ConvNets>.

### Acknowledgements

Greatly thanks to <https://github.com/wkcn/MobulaOP> by @wkcn.

If you have any question or suggestion, please feel free to send me a mail or create an issue.

### Todo List:
- [ ] Train OpenPose with [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- [ ] [PolarMask](https://arxiv.org/1909.13226) 