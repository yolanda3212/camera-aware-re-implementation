import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 12.20 dataset with camera-id and pseudo proxy label
class ProxyDataset(Dataset):
    def __init__(self, img_shape, samples, proxy_nums):
        super(ProxyDataset, self).__init__()
        self.samples = samples
        self.img_shape = img_shape
        self.proxy_nums = proxy_nums
        self.cam_proxy_map = self._create_cam_proxy_map()

    def _create_cam_proxy_map(self):
        res = {}
        for item in self.proxy_nums:
            camid, proxy_num = item['camid'], item['proxy_num']
            res[camid] = proxy_num
        return res

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, camid, cluster_label, proxy_label = self.samples[index]
        h, w = self.img_shape
        img = Image.open(fname).convert('RGB')
        img = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(h+10,w+10)),
            torchvision.transforms.RandomCrop(size=(h,w), pad_if_needed=True, padding_mode='edge'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # SpCL config
            torchvision.transforms.RandomErasing()
        ])(img)

        return img, cluster_label, proxy_label, camid

# 11.30 evaluation wrapper
class EvalDataset(Dataset):
    def __init__(self, img_shape, old_dataset, mode='query'):
        super(EvalDataset, self).__init__()
        self.img_shape = img_shape
        if mode == 'query':
            self.datalist = old_dataset.query
        elif mode == 'gallery':
            self.datalist = old_dataset.gallery
        else:
            raise ValueError('Argument mode should be "query" | "gallery"!')

    def __len__(self):
        return len(self.datalist)

    def _get_item_with_img(self, index):
        fname, vid, camid = self.datalist[index]
        h, w = self.img_shape
        img = Image.open(fname).convert('RGB')

        # For evaluation, no data augmentation
        img = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(h,w)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # SpCL config
        ])(img)

        return {'images': img, 'targets':vid, 'camids':camid}

    def __getitem__(self, index):
        return self._get_item_with_img(index)

# 11.23 refined dataset
class RefinedDataset(Dataset):
    def __init__(self, img_shape, old_dataset, good_labels):
        super(RefinedDataset, self).__init__()
        self.good_labels = good_labels
        self.img_shape = img_shape
        self.train = old_dataset.train

    def __len__(self):
        return len(self.train)

    def _get_item_with_img(self, index):
        fname, vid, camid = self.train[index]
        h, w = self.img_shape
        img = Image.open(fname).convert('RGB')
        img = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(h+10,w+10)),
            torchvision.transforms.RandomCrop(size=(h,w), pad_if_needed=True, padding_mode='edge'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # SpCL config
            torchvision.transforms.RandomErasing()
        ])(img)

        return img, self.good_labels[index], fname, vid, camid

    def __getitem__(self, index):
        return self._get_item_with_img(index)

# 11.19 rebuild
# 12.11 abandon transformation on CustomDatset -> for global feature extraction
class CustomDataset(Dataset):
    def __init__(self, img_shape, dataset, mode='train'):
        super(CustomDataset, self).__init__()
        self.img_shape = img_shape
        self.datalist = self._init_datalist(dataset, mode)

    def _init_datalist(self, dataset, mode):
        if mode == 'train':
            return dataset.train
        elif mode == 'test':
            return list(set(dataset.query) | set(dataset.gallery))
        else:
            raise ValueError('Wrong argument value of mode!')

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        return self._get_item_with_img(index)

    def _get_item_with_img(self, index):
        fname, vid, camid = self.datalist[index]
        h, w = self.img_shape
        img = Image.open(fname).convert('RGB')
        img = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(h,w)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # SpCL config
        ])(img)
        return img, fname, vid, camid
