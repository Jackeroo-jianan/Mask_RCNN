import cv2
import numpy as np
import os
import sys
import random
import math
import skimage.io
import time
import matplotlib.pyplot as plt
import threading
import tensorflow as tf

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config

# To find local version
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

import coco


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255*np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    if not n_instances:
        print('No instances to display')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_colors(n_instances)
    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = names[ids[i]]
        score = scores[i] if scores is not None else None

        caption = '{}{:.2f}'.format(label, score) if score else label
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


if __name__ == '__main__':
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_my.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        print('cannot find coco_model')

    class InferenceConfig(Config):
        # Give the configuration a recognizable name,给配置一个名称用于构造网络
        NAME = "shapes"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_MIN_DIM = 768
        IMAGE_MAX_DIM = 1280
        # IMAGE_MIN_DIM = 1088
        # IMAGE_MAX_DIM = 1920
        # IMAGE_MIN_DIM = 576
        # IMAGE_MAX_DIM = 704
        # 分类数量
        NUM_CLASSES = 1 + 10  # background + 1 shapes
        # 控制识别图片的大小
        RPN_ANCHOR_SCALES = (8*6, 16*6, 32*6, 64*6, 128*6)  # anchor side in pixels

    config = InferenceConfig()
    config.display()
    # config2 = coco.CocoConfig()
    # config2.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    # class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #                'bus', 'train', 'truck', 'boat', 'traffic light',
    #                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    #                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    #                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    #                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    #                'kite', 'baseball bat', 'baseball glove', 'skateboard',
    #                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    #                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    #                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    #                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    #                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    #                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    #                'teddy bear', 'hair drier', 'toothbrush']
    class_names = ['BG', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    capture = cv2.VideoCapture(
        "rtsp://admin:admin123@192.168.3.76:554/Streaming/Channels/1")
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 704)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)

    graphNow = tf.get_default_graph()

    def detectModel(model, frame):
        frame=cv2.resize(frame,(512,512))
        print("线程启动",threading.currentThread().ident,frame.shape)
        start = time.time()
        with graphNow.as_default():
            results = model.detect([frame], verbose=0)
            r = results[0]

            # frame = display_instances(
            #     frame, r['rois'], r['masks'], r['class_ids'],
            #     class_names, r['scores']
            # )
        end = time.time()
        print("识别时间：")
        print(end-start)
        print("线程结束",threading.currentThread().ident)

    t1 = None
    i=0
    # 先识别一次，首次识别耗时较长
    ret, frame = capture.read()
    results = model.detect([frame], verbose=0)

    while ret:
        start = time.time()

        ret, frame = capture.read()
        if t1 == None or t1.isAlive() == False:
        # if i%7==0:
            t1 = threading.Thread(target=detectModel, args=(model, frame))
            t1.start()
        i+=1

        end = time.time()

        # print("视频每帧时间：")
        # print(end-start)
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cv2.destroyAllWindows()
    capture.release()
