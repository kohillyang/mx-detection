import datetime
import json
import os

import cv2
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon.data import Dataset
from tqdm import tqdm
import mxnet as mx
from PIL import Image


class DetectionDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(DetectionDataset, self).__init__()

    def __len__(self):
        return 0

    def viz(self, indexes=None):
        from gluoncv.utils.viz import plot_bbox
        import matplotlib.pyplot as plt
        if indexes is None:
            indexes = range(len(self))
        for index in indexes:
            x = self.at_with_image_path(index)
            img_path = x[0]
            img = cv2.imread(img_path)[:, :, ::-1]
            plot_bbox(img, x[1][:, :4], labels=x[1][:, 4], class_names=self.classes)
            plt.show()

    def viz_seg(self, indexes=None):
        from gluoncv.utils.viz import plot_bbox
        import matplotlib.pyplot as plt
        if indexes is None:
            indexes = range(len(self))
        for index in indexes:
            x = self.at_with_image_path(index)
            img_path = x[0]
            img = cv2.imread(img_path)[:, :, ::-1]
            plot_bbox(img, x[1][:, :4], labels=x[1][:, 4], class_names=self.classes)
            for k in range(len(x[2])):
                p = plt.Polygon(zip(x[2][k][0][:, 0], x[2][k][0][:, 1]), alpha=.4)
                plt.gca().add_patch(p)
            plt.show()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_class(self):
        return self.num_classes

    def at_with_image_path(self, idx):
        '''
        return img_path and bbox, implement this if you want to use to_coco.
        Note the bbox in this case cannot be transformed.
        '''
        raise NotImplementedError()

    def to_roidb(self, db_path):
        '''
        Compressing this dataset to a pickeled roidb format file, which may
        be helpful if you want to train your own Deformable Convnets Model.
        '''
        r = []
        for i in tqdm(range(len(self))):
            img_path_ori, bbox = self.at_with_image_path(i)
            bbox = np.array(bbox).astype(np.float32)
            image = Image.open(img_path_ori)
            width, height = image.size
            # img = cv2.imread(img_path_ori)

            onedb = {}
            onedb["boxes"] = bbox[:, :4].astype(np.int32)
            onedb["height"] = height
            onedb["width"] = width
            onedb["image"] = img_path_ori
            onedb["flipped"] = False

            num_objs = bbox.shape[0]
            assert num_objs > 0
            num_classes = self.num_classes + 1
            overlaps = np.zeros(shape=(num_objs, num_classes), dtype=np.float32)
            for idx in range(bbox.shape[0]):
                cls = bbox[idx, 4]
                overlaps[idx, int(cls)] = 1.0
            onedb["gt_classes"] = bbox[:, 4].astype(np.int32) + 1
            onedb["gt_overlaps"] = overlaps
            onedb["max_classes"] = overlaps.argmax(axis=1)
            onedb["max_overlaps"] = overlaps.max(axis=1)
            r.append(onedb)
        import pickle
        pickle.dump(r, open(db_path, "wb"), protocol=0)
        return r

    def to_coco(self, anno_path, INFO=None, LICENSES=None):

        # use command `pip install git+git://github.com/waspinator/pycococreator.git@0.2.0` to install pycococreator.
        '''
        Just convert this data set to coco detection format.
        Currently segmentation is not supported.
        '''
        from .pycococreatetools import create_annotation_info_detection_only, create_image_info
        classes = self.classes
        if INFO is None:
            INFO = {
                "description": "Example Dataset",
                "url": "https://github.com/waspinator/pycococreator",
                "version": "0.1.0",
                "year": 2018,
                "contributor": "kohill",
                "date_created": datetime.datetime.utcnow().isoformat(' ')
            }
        if LICENSES is None:
            LICENSES = [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }
            ]
        CATEGORIES = []
        for cat_id in range(len(classes)):
            CATEGORIES.append(
                {
                    'id': cat_id + 1,  # Make sure cat_id is not zero.
                    'name': classes[cat_id],
                    'supercategory': 'root',
                }
            )
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        objs = {}
        objs_id = {}
        for idx in range(len(self)):
            img_path, bbox = self.at_with_image_path(idx)
            objs_id[img_path] = idx
            for x0, y0, x1, y1, cls in bbox[:, :5]:
                try:
                    objs[img_path].append([x0, y0, x1, y1, cls])
                except KeyError:
                    objs[img_path] = [[x0, y0, x1, y1, cls]]
        anno_id = 1
        # filter for jpeg images
        for image_path in tqdm(objs.keys()):
            image_id = objs_id[image_path]
            # image = cv2.imread(image_path)
            image = Image.open(image_path)
            width, height = image.size
            image_info = create_image_info(
                image_id, os.path.basename(image_path), [width, height])
            coco_output["images"].append(image_info)

            bboxes = objs[image_path]
            for x0, y0, x1, y1, class_id in bboxes:
                category_id = int(class_id) + 1  # cat_id starts from 1.
                is_crowd = 0
                assert x1 - x0 >= 0, x0
                assert y1 - y0 >= 0, y0
                annotation_info = create_annotation_info_detection_only(anno_id, image_id,
                                                                        category_id=category_id,
                                                                        is_crowd=is_crowd,
                                                                        bounding_box=np.array(
                                                                            [x0, y0, x1 - x0, y1 - y0]))
                coco_output["annotations"].append(annotation_info)
                anno_id = anno_id + 1
        with open(anno_path, 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

    def pascal_write_box(self, xml_path, boxes, filepath):
        width, height = Image.open(filepath).size
        objs = ""
        for box in boxes[:, :5]:
            x0, y0, x1, y1, cls = box
            class_name = self.classes[int(cls)]
            one_obj = '''
            <object><name>{0}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
            <bndbox>
                <xmin>{1}</xmin>
                <ymin>{2}</ymin>
                <xmax>{3}</xmax>
                <ymax>{4}</ymax>
            </bndbox>
            </object>
            '''.format(class_name, x0, y0, x1, y1)
            objs += one_obj
        str2write = '''
        <annotation>
            <folder>{}</folder>
            <filename>{}</filename>
            <path>{}</path>
            <source>
            <database>Unknown</database>
            </source>
            <size>
                <width>{}</width>
                <height>{}</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
            {}
        </annotation>

        '''.format(filepath.strip().split('/')[-2],
                   os.path.basename(filepath),
                   filepath,
                   width, height, objs)
        with open(xml_path, "wt") as f:
            f.write(str2write)

    def to_pascal(self, dir):
        for i in tqdm(range(len(self))):
            filename, bboxes = self.at_with_image_path(i)
            xml_path = os.path.join(dir, os.path.splitext(os.path.basename(filename))[0] + ".xml")
            self.pascal_write_box(xml_path, bboxes, filename)

    def parser_pascal_voc_xml(self, xml_path, img_root):
        import xml.etree.ElementTree as ET
        import logging
        oneimg = {}
        oneimg['bndbox'] = []
        try:
            dom = ET.parse(xml_path)
        except Exception as e:
            logging.error("{}_{}".format(e, xml_path))
            return None
        root = dom.getroot()
        filename = root.findall('filename')[0].text
        oneimg['path'] = os.path.join(img_root, filename)
        oneimg['filename'] = filename
        for objects in root.findall('object'):
            name = objects.find('name').text
            points = list(objects.find('bndbox'))
            if len(points) != 4:
                logging.warning("find illegal label in file:{}.xml. ".format(filename))
                print(points)
                return None
            xmin = int(points[0].text)
            ymin = int(points[1].text)
            xmax = int(points[2].text)
            ymax = int(points[3].text)

            oneimg['bndbox'].append([xmin, ymin, xmax, ymax, name])
        return oneimg


class AspectGroupingDataset(object):
    def __init__(self, base_dataset, config, transformer=None, target_generator=None):
        self.base_dataset = base_dataset
        self.config = config
        self.aspects = [self.base_dataset.at_ratio(x) for x in range(len(self.base_dataset))]
        self.aspects_argsort = np.argsort(self.aspects)
        self.short_size = config.TRAIN.image_short_size
        self.max_size = config.TRAIN.image_max_long_size
        import data.transforms.bbox as bbox_t
        if transformer is None:
            self.transformer = bbox_t.Resize(target_size=self.short_size, max_size=self.max_size)
        else:
            self.transformer = transformer
        if target_generator is None:
            assert target_generator is not None, "target_generator should be not None."
        else:
            self.target_generator = target_generator
        self.batch_size = config.TRAIN.batch_size

    def __len__(self):
        return len(self.base_dataset) // self.batch_size

    def __getitem__(self, idx):
        images_list = []
        bboxes_list = []
        for i in range(self.batch_size):
            idx_mapped = self.aspects_argsort[idx * self.batch_size + i]
            image, bbox = self.base_dataset[idx_mapped]
            image, bbox = self.transformer(image, bbox)
            bbox = bbox[:, :5]
            images_list.append(image)
            bboxes_list.append(bbox)
        # pad = lambda x: x if x % 128 == 0 else x + 128 - x % 128
        pad = lambda x:x
        max_h = pad(max([x.shape[0] for x in images_list]))
        max_w = pad(max([x.shape[1] for x in images_list]))
        for i in range(len(images_list)):
            image = images_list[i]
            image0_padded = np.zeros(shape=(max_h, max_w, image.shape[2]), dtype=image.dtype)
            image0_padded[:image.shape[0], :image.shape[1], :] = image
            images_list[i] = image0_padded
        images_and_targets_list = []
        for i in range(len(images_list)):
            images_and_targets = self.target_generator(images_list[i], bboxes_list[i])
            images_and_targets_list.append(images_and_targets)
        r = []
        for i in range(len(images_and_targets_list[0])):
            r.append(np.concatenate([x[i][np.newaxis] for x in images_and_targets_list], axis=0))
        return tuple((mx.nd.array(x) for x in r))
