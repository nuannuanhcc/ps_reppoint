import mmcv
import numpy as np
from numpy import random
import torchvision
from PIL import Image
import math
import random
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self, range_ratio=0.2, range_overlaps=(0.1, 0.9)):
        # 1: return ori img
        self.range_overlaps = range_overlaps
        self.range_ratio = range_ratio

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            # import cv2
            # import neptune
            # im2 = img.copy()
            # for i in range(boxes.shape[0]):
            #     cv2.rectangle(im2, tuple(map(int, boxes[i][:2])), tuple(map(int, boxes[i][-2:])), (255, 0, 0), 4)
            # neptune.log_image('mosaics', im2)

            if random.randint(0, 1):
                return img, boxes, labels

            for i in range(50):
                # keep the labeled id
                idx_labeled = labels[:, 1] > -1
                if not any(idx_labeled):
                    idx_labeled = ~idx_labeled
                boxes_labeled = boxes[idx_labeled]
                x1 = min(boxes_labeled[:, 0])
                y1 = min(boxes_labeled[:, 1])
                x2 = max(boxes_labeled[:, 2])
                y2 = max(boxes_labeled[:, 3])

                # based on above, random choose coord
                new_x1 = random.uniform(0, x1)
                new_y1 = random.uniform(0, y1)
                new_x2 = random.uniform(x2, w)
                new_y2 = random.uniform(y2, h)
                new_w = new_x2 - new_x1
                new_h = new_y2 - new_y1

                # keep the nearby ratio
                if new_h / new_w < h / w - self.range_ratio or new_h / new_w > h / w + self.range_ratio:
                    continue

                patch = np.array((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
                overlaps = bbox_overlaps(patch.reshape(-1, 4), boxes.reshape(-1, 4), mode='iof1').reshape(-1)

                if any((self.range_overlaps[0] < overlaps) & (overlaps < self.range_overlaps[1])):
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


class ColorJitter(object):

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0, box_mode=False):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )
        self.box_mode = box_mode

    def __call__(self, img, boxes, labels):
        if random.randint(0, 1):
            return img, boxes, labels
        img_ori = img.copy()
        pil_img = Image.fromarray(np.uint8(img))
        pil_img = self.color_jitter(pil_img)
        img = np.array(pil_img, copy=True)

        if not self.box_mode:
            return img, boxes, labels

        for i, box in enumerate(boxes.astype(int)):
            img[box[1]:box[3], box[0]:box[2]] = img_ori[box[1]:box[3], box[0]:box[2]]
        return img, boxes, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 colorjitter=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))
        if colorjitter is not None:
            self.transforms.append(ColorJitter(**colorjitter))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels
