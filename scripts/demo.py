from models.fpn.fpn_dcn import PyramidRFCN
from models.fpn.resnext import SEResNext50_32x4d
from utils.config import update_config, config
import mxnet as mx
from utils.im_detect import im_detect_bbox_aug
import os
import numpy as np
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import easydict
import json
from data.bbox.mscoco import COCODetection


def evaluate(json_label, json_predict, classes=COCODetection.CLASSES):
    args = easydict.EasyDict()
    args.label = json_label
    args.predict = json_predict
    cocoGt = COCO(args.label)
    # create filename to imgid
    catIds = cocoGt.getCatIds()
    # A remapping is needed because of the differences of the order.
    cat_name2id={x["name"]: x["id"] for x in cocoGt.loadCats(catIds)}
    catid2catbane = {entry["id"]: entry["name"] for entry in cocoGt.cats.values()}

    imgIds = cocoGt.getImgIds()
    imgs = cocoGt.loadImgs(imgIds)
    filename2imgid = {entry["file_name"]: entry["id"] for entry in imgs}
    submit_validataion = json.load(open(args.predict, "rt"), encoding="utf-8")["results"]
    coco_results = []
    for onefile in submit_validataion:
        # {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
        filename = onefile["filename"]
        for rect in onefile["rects"]:
            coco_results.append({"image_id": filename2imgid[filename],
                                 "category_id": cat_name2id[classes[rect["label"]-1]],
                                 "bbox": [rect["xmin"], rect["ymin"], rect["xmax"] - rect["xmin"] + 1,
                                          rect["ymax"] - rect["ymin"] + 1],
                                 "score": rect["confidence"]
                                 })
    json.dump(coco_results, open("output/tmp.json", "wt"))
    cocoEval = COCOeval(cocoGt, cocoGt.loadRes("output/tmp.json"), "bbox")
    cocoEval.params.imgIds = imgIds
    mAP_eachclasses = {}
    # for catId in catIds:
    #     print(u"Evaluate %s" % (catid2catbane[catId]))
    #     cocoEval.params.catIds = [catId]
    #     cocoEval.evaluate()
    #     cocoEval.accumulate()
    #     cocoEval.summarize()
    #     mAP_eachclasses[catid2catbane[catId]] = cocoEval.stats[1]
    print(u"Evaluate all classes.")
    cocoEval.params.catIds = catIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    mAP_eachclasses[u"mAP@IoU=0.5"] = cocoEval.stats[1]
    print("************summary***************")
    for k in mAP_eachclasses.keys():
        print (k, mAP_eachclasses[k])


if __name__ == '__main__':
    update_config("configs/coco/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem.yaml")
    backbone = SEResNext50_32x4d()
    net = PyramidRFCN(config, backbone)
    params_pretrained = mx.nd.load("output/fpn_coco-5-0.0.params")
    for k in params_pretrained:
        params_pretrained[k.replace("arg:", "").replace("aux:", "")] = params_pretrained.pop(k)
    params = net.collect_params()
    for k in params.keys():
        if k in params_pretrained.keys():
            params[k]._load_init(params_pretrained[k], ctx=mx.cpu())
        else:
            print (k)


    results = {}
    results["results"] = []
    net.collect_params().reset_ctx(mx.gpu(8))
    import tqdm,json
    for root_dir, _, names in os.walk("/data3/zyx/yks/coco2017/val2017"):
        for name in tqdm.tqdm(names):
            one_img = {}
            one_img["filename"] = os.path.basename(name)
            one_img["rects"] = []
            im = mx.image.imread(os.path.join(root_dir, name)).asnumpy()
            pred_bboxes, pred_scores, pred_clsid = im_detect_bbox_aug(net, im=im,
                                                                      scales=[(800, 1280)],
                                                                      ctx=[mx.gpu(8)],
                                                                      bbox_stds=config.TRAIN.BBOX_STDS, viz=False,
                                                                      pad=32, hflip=False, vflip=False, vhflip=False,
                                                                      class_agnostic=config.CLASS_AGNOSTIC)

            for bbox, score, label in zip(pred_bboxes, pred_scores, pred_clsid):
                one_rect = {}
                xmin, ymin, xmax, ymax = bbox[:4]
                one_rect["xmin"] = int(np.round(xmin))
                one_rect["ymin"] = int(np.round(ymin))
                one_rect["xmax"] = int(np.round(xmax))
                one_rect["ymax"] = int(np.round(ymax))
                one_rect["confidence"] = score
                one_rect["label"] = label
                one_img["rects"].append(one_rect)
            results["results"].append(one_img)

    save_path = 'output/coco/results_se_resnext50_32x4d_6_0.json'
    json.dump(results, open(save_path, "wt"))
    evaluate(json_label="/data3/zyx/yks/coco2017/annotations/instances_val2017.json", json_predict=save_path)