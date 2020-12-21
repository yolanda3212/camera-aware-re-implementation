import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd

class ProxyMemoryBank(nn.Module):
    '''
    Memory bank of proxy feature centroids.

    Args:
        cam_proxy_map: dict, a dict recording number of proxies of each camera.
        temp: float, temperature factor of the contrastive loss.
        momentum: float, momentum factor in updating of the memory bank.

    Returns:
        A proxy memory bank instance.
    '''
    def __init__(self, feature_dims, cam_proxy_map, temp=0.05, momentum=0.2):
        super(ProxyMemoryBank, self).__init__()
        self.feature_dims = feature_dims
        self.cam_proxy_map = cam_proxy_map
        self.temp = temp
        self.momentum = momentum
        self.storage = self._init_storage() # memory bank storage structure

    def init_entities(self, dataset, features):
        '''
        Initialize proxy memory bank entities with given dataset and features. Both \
        two arguments share the same order of samples. After the initialization, the\
        ProxyMemoryBank.storage is filled with proxy centroids.

        Args:
            dataset: ProxyDataset, containing all refined samples after clustering.
            features: Tensor, containing all features extracted by the backbone model.
        '''
        # Step 1: enumerate on each camera
        for camid, proxy_num in self.cam_proxy_map.items():
            same_cam_samples = self._get_same_cam_samples(camid, dataset)
            
            # Step 2: get each proxy's centroid and initialize memory
            for i in range(proxy_num):
                centroid = self._cal_proxy_centroid(same_cam_samples, i, features)
                self.storage[camid][i:] = centroid

    # TODO
    def _get_same_cam_samples(self, camid, dataset):
        pass

    # TODO
    def _cal_proxy_centroid(self,samples, proxy_id, features):
        pass

    def _init_storage(self):
        storage = {}
        for camid, proxy_num in self.cam_proxy_map.items():
            storage[camid] = torch.zeros((proxy_num, self.feature_dims))
        return storage


    def forward(self, input_features, camids, proxy_labels):
        # TODO: intra-camera loss
        pass

        # TODO: inter-camera loss
        pass