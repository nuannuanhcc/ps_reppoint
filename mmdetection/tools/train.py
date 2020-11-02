from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)

    do_test = True
    if do_test:
        print('\nDoing inference')
        from mmdet.datasets import build_dataloader
        import mmcv
        from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
        from mmcv.runner import get_dist_info, load_checkpoint

        def single_gpu_test(model, data_loader, show=False):
            model.eval()
            results = []
            dataset = data_loader.dataset
            prog_bar = mmcv.ProgressBar(len(dataset))
            for i, data in enumerate(data_loader):
                with torch.no_grad():
                    result = model(return_loss=False, rescale=not show, **data)
                results.append(result)

                if show:
                    model.module.show_result(data, result, dataset.img_norm_cfg)

                batch_size = data['img'][0].size(0)
                for _ in range(batch_size):
                    prog_bar.update()
            return results

        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        dataset = [build_dataset(cfg.data.test)]
        if cfg.data.test.with_reid:
            dataset.append(build_dataset(cfg.data.query))
        data_loader = [
            build_dataloader(
                ds,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            for ds in dataset]

        for i in range(cfg.total_epochs, cfg.total_epochs-1, -1):
            model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            ckpt = os.path.join(cfg.work_dir, 'epoch_' + str(i) + '.pth')
            load_checkpoint(model, ckpt, map_location='cpu')
            model = MMDataParallel(model, device_ids=[0])
            outputs = [single_gpu_test(model, dl) for dl in data_loader]

            print('\nStarting evaluate {}'.format(ckpt))
            result = dataset[0].evaluate(outputs, dataset)
            with open(os.path.join(cfg.work_dir, "eva_result.txt"), "a") as fid:
                fid.write(ckpt + '\n')
                fid.write(result+'\n')


if __name__ == '__main__':
    main()
