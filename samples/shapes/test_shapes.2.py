
# coding: utf-8

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import yaml
import keras
from PIL import Image
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from coco import coco

# 参考https://blog.csdn.net/l297969586/article/details/79140840/

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_my.h5")
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# ## Configurations

class MyDataset(utils.Dataset):
    """自定义训练集
    """

    # def __init__(self):
    #     self.iter_num = 0

    def from_yaml_get_class(self, image_id):
        """解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
        """
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        """生成mask,通过不同的颜色索引"""
        # print("image_id:",image_id)
        # print("self.image_info:",self.image_info)
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了mask_path 、yaml_path
    def load_shapes(self, count, height, width, img_floder, imglist):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        # 分类，id要从1开始
        self.add_class("shapes", 1, "0")
        self.add_class("shapes", 2, "1")
        self.add_class("shapes", 3, "2")
        self.add_class("shapes", 4, "3")
        self.add_class("shapes", 5, "4")
        self.add_class("shapes", 6, "5")
        self.add_class("shapes", 7, "6")
        self.add_class("shapes", 8, "7")
        self.add_class("shapes", 9, "8")
        self.add_class("shapes", 10, "9")

        for i in range(count):
            if os.path.isfile(os.path.join(img_floder, imglist[i])):
                filestr = imglist[i].split(".")[0]
                # filestr = filestr.split("_")[1]
                # mask_path = mask_floder + "/" + filestr + ".png"
                mask_path = os.path.join(
                    img_floder, "json/"+filestr+"_json/label.png")
                yaml_path = os.path.join(
                    img_floder, "json/"+filestr+"_json/info.yaml")
                self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                               width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        通过ID读图片
        """
        info = self.image_info[image_id]
        # print("info:",info)
        # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        # image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        # image = image * bg_color.astype(np.uint8)
        # for shape, color, dims in info['shapes']:
        #     image = self.draw_shape(image, shape, dims, color)
        image = keras.preprocessing.image.load_img(
            info["path"], target_size=(info['height'], info['width']))
        image = keras.preprocessing.image.img_to_array(image)
        return image

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        # ID对应的图片信息
        info = self.image_info[image_id]
        count = 1  # number of object
        # 通过路径打开图片
        img = Image.open(info['mask_path'])
        # 图片大小resize成固定大小
        img = img.resize((info['width'], info['height']))
        # 获得物体数量,图片是P模式，里面的值是颜色表的索引，范围0-255。所以np.max(img)能拿到物体数量
        # print("load_mask img",np.array(img))
        num_obj = np.max(img)
        # print("load_mask num_obj",num_obj)
        # 初始化mask矩阵
        mask = np.zeros(
            [info['height'], info['width'], num_obj], dtype=np.uint8)
        # 画mask图
        mask = self.draw_mask(num_obj, mask, img, image_id)
        # 取最后一个mask做逻辑非运算
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        # 数量大于等于2时，倒序循环遍历（该循环好像没意义）
        for i in range(count - 2, -1, -1):
            # 求交集
            mask[:, :, i] = mask[:, :, i] * occlusion
            # 做非运算，在于上一次结果做并运算，与上面的结果完全一样
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("0") == 0:
                labels_form.append("0")
            elif labels[i].find("1") == 0:
                labels_form.append("1")
            elif labels[i].find("2") == 0:
                labels_form.append("2")
            elif labels[i].find("3") == 0:
                labels_form.append("3")
            elif labels[i].find("4") == 0:
                labels_form.append("4")
            elif labels[i].find("5") == 0:
                labels_form.append("5")
            elif labels[i].find("6") == 0:
                labels_form.append("6")
            elif labels[i].find("7") == 0:
                labels_form.append("7")
            elif labels[i].find("8") == 0:
                labels_form.append("8")
            elif labels[i].find("9") == 0:
                labels_form.append("9")
        # 种类对应的序号
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # 分类数量
    NUM_CLASSES = 1 + 10  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1280
    # IMAGE_MIN_DIM = 1088
    # IMAGE_MAX_DIM = 1920
    # IMAGE_MIN_DIM = 768
    # IMAGE_MAX_DIM = 1408
    # IMAGE_MIN_DIM = 320
    # IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # 控制识别图片的大小
    RPN_ANCHOR_SCALES = (8*6, 16*6, 32*6, 64*6, 128*6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple训练步数
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small验证数量
    VALIDATION_STEPS = 5

    # RPN_ANCHOR_SCALES = (32, 128, 512)

    # BACKBONE = "resnet50"


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# 基础设置
parser = argparse.ArgumentParser()
parser.add_argument('labels_path')
args = parser.parse_args()

dataset_root_path = args.labels_path
img_floder_val = os.path.join(dataset_root_path, "test")
# 识别图片列表
imglist_val = os.listdir(img_floder_val)
count_val = len(imglist_val)
width = inference_config.IMAGE_MAX_DIM
height = inference_config.IMAGE_MIN_DIM
print("width", width)
print("height", height)

# Validation dataset 识别数据集
dataset_val = MyDataset()
dataset_val.load_shapes(count_val, height, width, img_floder_val,
                        imglist_val)
dataset_val.prepare()

model_path = os.path.join(MODEL_DIR, "mask_rcnn_my.h5")
# model_path = COCO_MODEL_PATH
# model.keras_model.save_weights(model_path)


# ## Detection 识别

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

print("dataset_val.class_names", dataset_val.class_names)
# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                 'bus', 'train', 'truck', 'boat', 'traffic light',
#                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                 'teddy bear', 'hair drier', 'toothbrush']
# Test on a random image
for image_id in dataset_val.image_ids:
    # 重新调整图片大小为正方形，边长等于最长边
    # original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
    #                                                                                    image_id, use_mini_mask=False)

    # log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                             dataset_train.class_names, figsize=(8, 8))

    # In[13]:
    original_image = dataset_val.load_image(image_id)  # 跳过调整大小直接读取图片
    results = model.detect([original_image], verbose=1)

    r = results[0]
    # print("识别：", r)
    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
    #                             dataset_val.class_names, r['scores'], ax=get_ax())
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], figsize=(8, 8))


# ## Evaluation

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
# image_ids = np.random.choice(dataset_val.image_ids, 10)
# APs = []
# for image_id in image_ids:
#     # Load image and ground truth data
#     image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
#                                                                               image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(
#         modellib.mold_image(image, inference_config), 0)
#     # Run object detection
#     results = model.detect([image], verbose=0)
#     r = results[0]
#     # Compute AP
#     AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                                                          r["rois"], r["class_ids"], r["scores"], r['masks'])
#     APs.append(AP)

# print("mAP: ", np.mean(APs))
