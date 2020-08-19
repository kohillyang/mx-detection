## Main results
`1`. VOC2012 trainval + VOC2007 trainval, test on VOC2007 test, DCN+ResNet50, mAP@IoU=0.5 is 0.804<br>
Download the pre-trained model from [OneDrive](https://pkamc-my.sharepoint.com/:u:/g/personal/by3410_office365vip_tech/EY1Ta2f54aZNklq4zjjek3wBXRy1uEMWYVhTymsPCfqvmA?e=BUbGGR)

`2`. VOC2012 trainval + VOC2007 trainval, test on VOC2007 test, DCN+ResNet101, mAP@IoU=0.5 is 0.825<br>
Download the pre-trained model from [OneDrive](https://pkamc-my.sharepoint.com/:u:/g/personal/by3410_office365vip_tech/EW8hZtillhNJmIiq8A4OMpsB0NdkBxNVwEfGU0TMT7qvVA?e=cVQCO9)

`3`. COCO2017, FPN + ResNet50 + FCOS[Work in Progress]<https://arxiv.org/abs/1904.01355><br>
Download the pre-trained model from [GoogleDrive](https://drive.google.com/drive/folders/1UocX6i1P-_xSUpW9inB8dk5dF2gCdK5L?usp=sharing).
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.375
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.396
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.472
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.292
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.449
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.466
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.249
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.520
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.605

```
`4` OpenPose, COCO2017 ResNet50 <https://arxiv.org/pdf/1812.08008.pdf>

Training command:
```bash
PYTHONPATH=. /data2/kohill/jye_sanka/anaconda3/bin/python3 scripts/train_openpose.py --dataset-root=/data1/coco --gpus=0,1,2 --disable-fusion --backbone=res50
```

```bash
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.564
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.787
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.610
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.555
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.588
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.601
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.800
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.641
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.564
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.658
```

### Acknowledgements

Greatly thanks to <https://github.com/wkcn/MobulaOP> by @wkcn.