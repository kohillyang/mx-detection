import json

import easydict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data.bbox.mscoco import COCODetection


def evaluate_coco(json_label, json_predict, classes=COCODetection.CLASSES):
    args = easydict.EasyDict()
    args.label = json_label
    args.predict = json_predict
    cocoGt = COCO(args.label)
    # create filename to imgid
    catIds = cocoGt.getCatIds()
    # A remapping is needed because of the differences of the order.
    cat_name2id = {x["name"]: x["id"] for x in cocoGt.loadCats(catIds)}
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
                                 "category_id": cat_name2id[classes[rect["label"] - 1]],
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
        print(k, mAP_eachclasses[k])
