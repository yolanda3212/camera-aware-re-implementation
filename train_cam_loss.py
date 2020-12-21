import sys

sys.path.append('../..')

import os
import time
import glob
import copy
import torch
import torch.nn as nn
import numpy as np
import tensorboardX

from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict

from tools.dataset_wrapper import CustomDataset, RefinedDataset
from tools.clusterer import Clusterer
from tools.load_config import load_config
from tools.init_gpus import init_gpus
from tools.sampler import ClusterSampler
from tools.split_for_proxy import split_for_proxy
from models.memory_bank import MemoryBank
from models.proxy_memory_bank import ProxyMemoryBank
from models.model import ReidNet
from settings import Settings

from fastreid.solver.lr_scheduler import WarmupMultiStepLR
from fastreid.data.datasets import VeRi, Market1501
from fastreid.modeling.backbones.resnet import build_resnet_backbone
from fastreid.solver.optim import Adam
from fastreid.modeling.heads.embedding_head import EmbeddingHead

def init_memory_bank(memory, centroids):
    '''
    Initialize memory bank with cluster centroids.

    Args:
        memory: The memory bank.
        centroids: list, the centroids of DBSCAN clustering.

    Returns:
        Initialized memory bank.
    '''
    for c in centroids:
        pid = c['cluster_id']
        feat = c['feature']
        memory.features[pid,:] = feat.unsqueeze(0) # init feature
        memory.labels[pid] = pid # init label
    print('>>> Memory bank is initiated with feature shape {}, label shape {}.'.format(memory.features.shape, memory.labels.shape))
    return memory

def build_dataloader(img_shape, dataset, batch_size, workers, mode='train'):
    '''
    Build a dataloader with fast re-id style dataset. When enumerating on it, it returns\
    (imgs, fnames, vids, camids) as a tuple.

    Args:
        img_shape: tuple, (height, width) of each input image. Depends on the model.
        dataset: Fast re-id style dataset.
        batch_size: int, batch size of each iteration.
        workers: int, workers of pytorch DataLoader.
        mode: str, 'train' | 'test', decide which dataset to use.

    Returns:
        A dataloader.
    '''

    if mode == 'train':
        custom_dataset = CustomDataset(img_shape, dataset, mode='train')
    elif mode == 'test':
        custom_dataset = CustomDataset(img_shape, dataset, mode='test')
    else:
        raise ValueError('Wrong argument value of mode!')
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

def extract_global_features(img_shape, batch_size, workers, model, dataset, mode='train', is_cuda=False):
    '''
    Extract global features from dataset.

    Args:
        img_shape: tuple, (height, width) of each input image. Depends on the model.
        batch_size: int, batch size of each iteration.
        workers: int, workers of pytorch DataLoader.
        model: Pytorch model.
        dataset: Fast re-id style dataset.
        mode: str, 'train' or 'test'.
        is_cuda: boolean, whether to use GPU.
    
    Returns:
        features: OrderedDict, global features.
        v_labels: OrderedDict, vehicle id labels.
        cam_labels: OrderedDict, camera id labels.
    '''
    if is_cuda:
        model = model.cuda()

    data_loader = build_dataloader(img_shape, dataset, batch_size=batch_size, workers=workers, mode=mode)
    
    # Containers of features and labels
    features = OrderedDict()
    v_labels = OrderedDict()
    cam_labels = OrderedDict()

    model = model.eval()
    with torch.no_grad():
        for i, (imgs, fnames, vids, camids) in enumerate(data_loader):
            if is_cuda:
                imgs = imgs.cuda()
            batch_feats = model(imgs).data.cpu() # extract batch of features
            for fname, batch_feat, vid, camid in zip(fnames, batch_feats, vids, camids):
                features[fname] = batch_feat
                v_labels[fname] = vid
                cam_labels[fname] = camid
    model = model.train()
    return features, v_labels, cam_labels
    
def merge_features_from_dict(features_dict):
    '''
    Merge features from dict to tensor.

    Args:
        features_dict: OrderedDict, features dict.

    Returns:
        Pytorch tensor of all features in the dict.
    '''
    tensor = torch.cat([v.unsqueeze(0) for _, v in features_dict.items()], dim=0)
    return tensor

def refine_dataset(img_shape, dataset, pseudo_labels):
    '''
    Refine original dataset with pseudo labels from clustering. Remove outliers.

    Args:
        img_shape: tuple, expected input image shape.
        dataset: Fast re-id style dataset.
        pseudo_labels: ndarray, output of clustering.

    Returns:
        Refined fast re-id dataset with outliers (-1 labeled) removed. When enumerating on its corresponding dataloader, it will return (pseudo_label, fname, vid, camid) as a tuple.
    '''
    refined_dataset = copy.deepcopy(dataset) # clone a new dataset
    good_indices = np.argwhere(pseudo_labels != -1).reshape((-1,))
    refined_dataset.train = [refined_dataset.train[i] for i in good_indices]
    print('>>> Remove {} outliers, re-arrange dataset with {} normal samples.'.format(len(np.argwhere(pseudo_labels==-1).reshape((-1,))), len(good_indices)))
    return RefinedDataset(img_shape, refined_dataset, pseudo_labels[good_indices])


def find_latest_checkpoint(ckpts):
    '''
    Find the latest checkpoint file name.

    Args:
        ckpts: list, list of checkpoint names.

    Returns:
        The latest checkpoint file name.
    '''
    ckpts.sort(key=lambda ckpt: int(ckpt.split('-')[-1].split('.')[0]))
    return ckpts[-1]

def opt_to_gpu(opt, is_cuda):
    if not is_cuda:
        return opt
    for state in opt.state.values():
        for k, v, in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    return opt


def train(cfg, model, dataset, optimizer, scheduler=None, logger=None, is_continue=False, use_pretrained=False, cluster_vis_path=None):
    
    save_to = cfg.TRAIN.CHECKPOINT_PATH
    epochs = cfg.TRAIN.EPOCHS
    batch_size = cfg.TRAIN.BATCHSIZE

    if logger is None:
        print('>>> No tensorboard logger used in training.')
    else:
        print('>>> Logger is used in training.')
        counter = 0

    if len(save_to) == 0:
        print('>>> No checkpoints will be saved.')

    start_ep = 0 # initiate start epoch number
    
    # 继续训练至预定epoch全部完成
    if is_continue:
        print('>>> Continue training from the latest checkpoint.')
        if save_to is None:
            print('>>> Without checkpoint folder, cannot continue training!')
            exit(0)
        ckpts = glob.glob(os.path.join(save_to, '*.pth'))
        if len(ckpts) == 0:
            print('>>> No earlier checkpoints, train from the beginning.')
        else:
            start_ckpt = find_latest_checkpoint(ckpts)
            print('>>> Found earlier checkpoints, continue training with {}.'.format(start_ckpt))

            # load latest model
            start_ep = torch.load(os.path.join(save_to, start_ckpt))['epoch']
            model_state = torch.load(os.path.join(save_to, start_ckpt))['model_state_dict'] # 加载权重、优化器、scheduler等信息
            opt_state = torch.load(os.path.join(save_to, start_ckpt))['optimizer_state_dict']
            model.load_state_dict(model_state)
            optimizer.load_state_dict(opt_state)
            optimizer = opt_to_gpu(optimizer, torch.cuda.is_available())
            if scheduler is not None:
                scheduler_state = torch.load(os.path.join(save_to, start_ckpt))['scheduler_state_dict']
                scheduler.load_state_dict(scheduler_state)
            if logger is not None:
                counter = torch.load(os.path.join(save_to, start_ckpt))['logger_counter']
    
    # 仅使用pretrained权重从头开始训练
    if use_pretrained:
        print('>>> Use pretrained model weights to start a new training.')
        model_state = torch.load(cfg.TRAIN.PRETRAINED_PATH)['model_state_dict'] # 只加载模型权重
        model.load_state_dict(model_state)

    if torch.cuda.is_available():
        model = model.cuda()

    # training loop
    for epoch in range(start_ep, epochs):
        # extract global features
        print('>>> Extracting global features ...')
        features, v_labels, cam_labels = extract_global_features(
            img_shape=(256,256),
            batch_size=batch_size, workers=4,
            model=model, dataset=dataset, mode='train',
            is_cuda=torch.cuda.is_available()
        )

        # clustering
        print('>>> Start clustering ...')
        features = merge_features_from_dict(features)
        pseudo_labels, num_ids, centroids = Clusterer(features, eps=0.5,
            is_cuda=torch.cuda.is_available()).cluster(visualize_path=cluster_vis_path, epoch=epoch+1)


        # create non-outlier refined dataset
        print('>>> Refining dataset ...')
        good_dataset = refine_dataset((256,256), dataset, pseudo_labels)



        # ################################## NOTE: new pipeline below! #################
        # get pseudo proxy labels
        proxy_dataset = split_for_proxy(good_dataset)

        import ipdb; ipdb.set_trace()

        # create dataloader with proxy-labeled data
        dataloader = DataLoader(dataset=proxy_dataset, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=False) # TODO: adopt proxy-balanced sampling


        # TODO: proxy memory bank initialization
        memory = ProxyMemoryBank(feature_dims=2048, cam_proxy_map=proxy_dataset.cam_proxy_map)
        memory.init_storage(proxy_dataset, features) # initialize memory bank with proxy centroids, features have been L2-normalized



        # TODO: memory bank construction test
        import ipdb; ipdb.set_trace()
        quit()



        # TODO: camera-aware training pipeline
        for i, (imgs, cluster_labels, proxy_labels, camids) in enumerate(dataloader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                memory = memory.cuda()
            optimizer.zero_grad()
            features = model(imgs)
            intra_loss, inter_loss = memory(features, camids, proxy_labels) # TODO: intra-cam and inter-cam loss
            loss = intra_loss + 0.5 * inter_loss # NOTE: total loss with balancing factor
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0: # print loss each 50 iters
                print('[epoch: {}/{}][iter: {}/{}] loss: {}'.format(epoch+1, epochs, i+1, len(dataloader), loss))
            
            # update logger
            if logger is not None:
                logger.add_scalar('total_loss', loss.item(), global_step=counter)
                logger.add_scalar('intra_cam_loss', intra_loss.item(), global_step=counter)
                logger.add_scalar('inter_cam_loss', inter_loss.item(), global_step=counter)
                logger.add_scalar('cluster_centroids', memory.num_samples, global_step=counter)
                logger.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=counter)
                counter += 1
        
        # update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # save checkpoint
        if len(save_to) != 0 and (epoch+1) % cfg.TRAIN.SAVE_INTERVAL == 0:
            save_name = os.path.join(save_to, 'backbone-epoch-{}.pth'.format(epoch+1))
            state_dict = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'logger_counter': counter if logger is not None else None
            }
            torch.save(state_dict, save_name)
            print('>>> Checkpoint is saved as {}.'.format(save_name))

def init_dataset(dataset_name, dataset_root, mode):
    if dataset_name == 'VeRi':
        dataset = VeRi(root=dataset_root, mode=mode)
    elif dataset_name == 'Market1501':
        dataset = Market1501(root=dataset_root, mode=mode)
    else:
        raise ValueError('Wrong dataset name!')
    return dataset

def config_summary(settings, cfg, dataset):
    start_time = time.localtime()
    print('--- Training Configuration Summary ---')
    print('Start time: {}-{}-{} {}:{}:{}'.format(start_time.tm_year, start_time.tm_mon, start_time.tm_mday, start_time.tm_hour, start_time.tm_min, start_time.tm_sec))
    print('Training nums:', len(dataset.train))
    print('Using GPU:', settings.gpu_ids)
    print('Epochs:', cfg.TRAIN.EPOCHS)
    print('Batch size:', cfg.TRAIN.BATCHSIZE)
    print('Learning rate:', cfg.TRAIN.LR)
    print('Weight decay:', cfg.TRAIN.WEIGHT_DECAY)
    print('Checkpoint save interval:', cfg.TRAIN.SAVE_INTERVAL)
    print('Pretrained weight:', cfg.TRAIN.PRETRAINED_PATH)    
    print('--------------------------------------')

def main():
    # load configurations
    Settings.init()
    if Settings.debug:
        print('>>> Debug mode.')
        import ipdb
        ipdb.set_trace()
    init_gpus(Settings.gpu_ids)
    cfg = load_config(Settings.conf)
    dataset = init_dataset(dataset_name='VeRi', dataset_root=cfg.DATASET.PATH, mode='train')
    # dataset = init_dataset(dataset_name='Market1501', dataset_root=cfg.DATASET.PATH, mode='train')
    dataset.train = dataset.train[:100] # shorten list for test
    
    # initiate model
    model = ReidNet(cfg)
    if len(Settings.gpu_ids.split(',')) > 1:
        print('>>> Using multi-GPUs, enable DataParallel')
        model = torch.nn.DataParallel(model)
    optim = torch.optim.SGD(params=model.parameters(), lr=cfg.TRAIN.LR, momentum=0.8)
    scheduler = WarmupMultiStepLR(optim, milestones=[29,49], gamma=0.1, warmup_iters=10, warmup_method='linear')

    # training monitor
    if len(cfg.TRAIN.LOG_PATH) != 0:
        logger = tensorboardX.SummaryWriter(cfg.TRAIN.LOG_PATH)
    else:
        logger = None


    # print config summary
    config_summary(Settings, cfg, dataset)

    # training
    train(
        cfg, model, dataset, optim, scheduler=scheduler,
        logger=logger, is_continue=Settings.is_continue, use_pretrained=Settings.use_pretrained,
        cluster_vis_path=None
    )

    # close logger
    if logger is not None:
        logger.close()

if __name__ == "__main__":
    main()