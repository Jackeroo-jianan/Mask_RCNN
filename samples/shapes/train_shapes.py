
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
#
#
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
#
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster.

# In[1]:


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

# get_ipython().run_line_magic('matplotlib', 'inline')

# 参考https://blog.csdn.net/l297969586/article/details/79140840/

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_my.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[2]:


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
    NUM_CLASSES = 1 + 11  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1280

    # Use smaller anchors because our image and objects are small
    # 控制识别图片的大小
    RPN_ANCHOR_SCALES = (8*6, 16*6, 32*6, 64*6, 128*6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple训练步数
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small验证数量
    VALIDATION_STEPS = 5

    BACKBONE = "resnet50"


config = ShapesConfig()
config.display()


# ## Notebook Preferences

# In[3]:


# def get_ax(rows=1, cols=1, size=8):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.

#     Change the default size attribute to control the size
#     of rendered images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax


# ## Dataset
#
# Create a synthetic dataset
#
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
#
# * load_image()
# * load_mask()
# * image_reference()

# In[4]:


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
        # 分类
        self.add_class("shapes", 0, "0")
        self.add_class("shapes", 1, "1")
        self.add_class("shapes", 2, "2")
        self.add_class("shapes", 3, "3")
        self.add_class("shapes", 4, "4")
        self.add_class("shapes", 5, "5")
        self.add_class("shapes", 6, "6")
        self.add_class("shapes", 7, "7")
        self.add_class("shapes", 8, "8")
        self.add_class("shapes", 9, "9")
        self.add_class("shapes", 10, "x")

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
            elif labels[i].find("x") == 0:
                labels_form.append("x")
        # 种类对应的序号
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


# In[5]:

# 基础设置
parser = argparse.ArgumentParser()
parser.add_argument('labels_path')
# parser.add_argument('-o', '--out', default=None)
args = parser.parse_args()

dataset_root_path = args.labels_path
img_floder = os.path.join(dataset_root_path, "train")
img_floder_val = os.path.join(dataset_root_path, "test")
# mask_floder = dataset_root_path+"mask"
#yaml_floder = dataset_root_path
# 训练图片列表
imglist = os.listdir(img_floder)
# 识别图片列表
imglist_val = os.listdir(img_floder_val)
count = len(imglist)
count_val = len(imglist_val)
width = config.IMAGE_MAX_DIM
height = config.IMAGE_MIN_DIM
print("width", width)
print("height", height)

# Training dataset 训练数据集
dataset_train = MyDataset()
dataset_train.load_shapes(count, height, width,
                          img_floder, imglist)
dataset_train.prepare()

# Validation dataset 识别数据集
dataset_val = MyDataset()
dataset_val.load_shapes(count_val, height, width, img_floder_val,
                        imglist_val)
dataset_val.prepare()


# In[6]:


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    print("mask", mask.shape)
    # print("class_ids",class_ids)
    visualize.display_top_masks(
        image, mask, class_ids, dataset_train.class_names)


# ## Ceate Model

# In[ ]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[7]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    if os.path.isfile(COCO_MODEL_PATH):
        print("加载模型：", COCO_MODEL_PATH)
        # model.load_weights(COCO_MODEL_PATH, by_name=True,
        #                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
        #                             "mrcnn_bbox", "mrcnn_mask"])
        model.load_weights(COCO_MODEL_PATH, by_name=True)
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training
#
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
#
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# In[8]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')


# In[9]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")


# In[10]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_my.h5")
model.keras_model.save_weights(model_path)


# ## Detection

# In[11]:


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[12]:


# Test on a random image
for image_id in dataset_val.image_ids:
    # image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                                       image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                             dataset_train.class_names, figsize=(8, 8))

    # In[13]:

    results = model.detect([original_image], verbose=1)

    r = results[0]
    # print("识别：", r)
    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
    #                             dataset_val.class_names, r['scores'], ax=get_ax())
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], figsize=(8, 8))


# ## Evaluation

# In[14]:


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                              image_id, use_mini_mask=False)
    molded_images = np.expand_dims(
        modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
