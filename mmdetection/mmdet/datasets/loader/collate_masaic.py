import collections

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from mmcv.parallel.data_container import DataContainer
from mmcv.parallel import DataContainer as DC
import mmcv
import numpy as np
import random

def DC_to_tensor(data):
    if torch.is_tensor(data['img']._data):
        imgs = data['img']._data.numpy().copy()
    else:
        imgs = data['img']._data.copy()
    bboxes = data['gt_bboxes']._data.clone()
    labels = data['gt_labels']._data.clone()
    img_meta = data['img_meta']._data.copy()
    return imgs, bboxes, labels, img_meta


def tensor_to_DC(imgs, bboxes, labels, img_meta):
    data = dict(
        img=DC(imgs, stack=True),
        img_meta=DC(img_meta, cpu_only=True),
        gt_bboxes=DC(bboxes),
        gt_labels=DC(labels))
    return data


def concat(b1, b2, type='hconcat'):
    assert type in ['hconcat', 'wconcat']
    imgs_1, bboxes_1, labels_1, img_meta_1 = DC_to_tensor(b1)
    imgs_2, bboxes_2, labels_2, img_meta_2 = DC_to_tensor(b2)

    # h * w * c
    dim = 0 if type == 'hconcat' else 1
    if imgs_1.shape[dim] > imgs_2.shape[dim]:
        imgs_2, imgs_1 = imgs_1, imgs_2
        bboxes_2, bboxes_1 = bboxes_1, bboxes_2
        labels_2, labels_1 = labels_1, labels_2

    h1, w1 = imgs_1.shape[:2]
    h2, w2 = imgs_2.shape[:2]
    scale_factor = h2 / h1 if type == 'hconcat' else w2 / w1
    h1, w1 = int(h1 * float(scale_factor) + 0.5), int(w1 * float(scale_factor) + 0.5)
    imgs_1 = mmcv.imresize(imgs_1, (w1, h1), interpolation='bilinear')  # input is (w, h)
    bboxes_1 = bboxes_1 * scale_factor

    if type == 'hconcat':
        bboxes_2[:, [0, 2]] += w1
    if type == 'wconcat':
        bboxes_2[:, [1, 3]] += h1

    imgs = np.concatenate((imgs_1, imgs_2), axis=1 if type == 'hconcat' else 0)
    bboxes = torch.cat([bboxes_1, bboxes_2], dim=0)
    labels = torch.cat([labels_1, labels_2], dim=0)
    return tensor_to_DC(imgs, bboxes, labels, img_meta_1)


def transform(data, scale=(1333, 900)):
    imgs, bboxes, labels, img_metas = DC_to_tensor(data)
    # transform -- resize + normalize + pad
    ori_shape = imgs.shape
    imgs, scale_factor = mmcv.imrescale(imgs, scale, return_scale=True)
    img_shape = imgs.shape
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    imgs = mmcv.imnormalize(imgs, mean, std, to_rgb)

    imgs = mmcv.impad_to_multiple(imgs, 32)
    pad_shape = imgs.shape
    imgs = imgs.transpose(2, 0, 1)

    bboxes = bboxes * scale_factor

    img_metas.update(ori_shape=ori_shape)
    img_metas.update(img_shape=img_shape)
    img_metas.update(pad_shape=pad_shape)
    img_metas.update(scale_factor=scale_factor)

    return tensor_to_DC(torch.from_numpy(imgs), bboxes, labels, img_metas)


def collate_masaic(batch, do_collate=False, img_scale=(1333, 900), samples_per_gpu=1, p=0.6):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if do_collate and isinstance(batch[0], collections.Mapping):
        if random.random() > p:
            # no masaic
            new_batch = []
            for i in range(3):
                new_batch.append(transform(batch[i], img_scale))
            batch = new_batch
        else:
            # # two image masaic
            # h, w = batch[0]['img']._data.shape[:2]
            # concat_type = 'hconcat' if w < h else 'wconcat'
            # new_batch = []
            # for i in range(len(batch)):
            #     imgs_concat = concat(batch[i % len(batch)], batch[(i+1) % len(batch)], concat_type)
            #     new_batch.append(transform(imgs_concat, img_scale))
            # batch = new_batch

        # # four image masaic
            new_batch = []
            for i in range(3):
                idx = random.sample(range(len(batch)), k=4)
                imgs_concat_w1 = concat(batch[idx[0]], batch[idx[1]], 'hconcat')
                imgs_concat_w2 = concat(batch[idx[2]], batch[idx[3]], 'hconcat')
                imgs_concat = concat(imgs_concat_w1, imgs_concat_w2, 'wconcat')
                new_batch.append(transform(imgs_concat, img_scale))
            batch = new_batch

    else:
        batch = batch

    if not isinstance(batch, collections.Sequence):
        raise TypeError("{} is not supported.".format(batch.dtype))

    if isinstance(batch[0], DataContainer):
        # assert len(batch) % samples_per_gpu == 0
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_masaic(samples, do_collate, img_scale, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        return {
            key: collate_masaic([d[key] for d in batch], do_collate, img_scale, samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)
