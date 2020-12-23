import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd

__all__ = ['ProxyMemoryBank', 'get_abs_proxy_labels']

def get_abs_proxy_labels(camids, proxy_labels, cam_proxy_map):
    res = []
    for cid, plabel in zip(camids, proxy_labels):
        proxy_cls_label = cam_proxy_map[cid]['cam_index'] + plabel
        res.append(proxy_cls_label)
    res = torch.tensor(res)
    # if torch.cuda.is_available():
    #     res = res.cuda()
    return res

class UpdateFunctionIntra(autograd.Function):
    @staticmethod
    def forward(ctx, input_features, stored_features, abs_proxy_indices, proxy_labels, momentum):
        # proxy_centroids = _get_centroids(stored_features)
        # ctx.proxy_centroids = proxy_centroids
        ctx.stored_features = stored_features
        ctx.momentum = momentum
        ctx.abs_proxy_indices = abs_proxy_indices
        ctx.save_for_backward(input_features, proxy_labels)
        # return input_features.mm(ctx.proxy_centroids.t())
        return input_features.mm(ctx.stored_features.t())

    @staticmethod
    def backward(ctx, grad_outputs):
        input_features, proxy_labels = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            # grad_inputs = grad_outputs.mm(ctx.proxy_centroids)
            grad_inputs = grad_outputs.mm(ctx.stored_features)

        # update memory bank storage
        for feat_in, abs_plabel, plabel in zip(input_features, ctx.abs_proxy_indices, proxy_labels):
            ctx.stored_features[abs_plabel] = ctx.momentum * ctx.stored_features[abs_plabel] + (1 - ctx.momentum) * feat_in
            ctx.stored_features[abs_plabel] /= ctx.stored_features[abs_plabel].norm() # L2-normalization

        return grad_inputs, None, None, None, None

def _get_centroids(stored_features):
    res = []
    ordered_keys = sorted(stored_features.keys(), key=lambda k: int(k.split('_')[-1]))
    for camid in ordered_keys:
        centroid = stored_features[camid]
        res.append(centroid)
    return torch.cat(res, dim=0)

def _update_memory_intra(input_features, stored_features, camids, proxy_labels, momentum):
    return UpdateFunctionIntra.apply(input_features, stored_features, camids, proxy_labels, torch.Tensor([momentum]).to(input_features.device))

# TODO
def _update_memory_inter(input_features, stored_features, camids, proxy_labels, momentum):
    pass

class ProxyMemoryBank(nn.Module):
    '''
    Memory bank of proxy feature centroids.\
    ProxyMemoryBank.storage records all proxy centroids. It is a dict like:\
    
    ```
        {
            'veri_0': torch.Tensor([[...]]),
            'veri_1': torch.Tensor([[...]]),
            ...
        }
    ```

    In the dict, each key denotes a camera id and the value is the proxies under this \
    camera. For example, if you want to choose the 3rd proxy centroid of camera-5, you \
    should act like this:

    ```
        camid = 'veri_5'
        centroid = memory.storage[camid][3] # The 3rd proxy centroid of camera 5
    ```

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
        self.register_buffer('storage', self._init_storage()) # memory bank storage structure

    def init_storage(self, dataset, features):
        '''
        Initialize proxy memory bank entities with given dataset and features. Both \
        two arguments share the same order of samples. After the initialization, the\
        ProxyMemoryBank.storage is filled with proxy centroids.

        Args:
            dataset: ProxyDataset, containing all refined samples after clustering.
            features: Tensor, containing all features extracted by the backbone model.
        '''
        # Step 1: enumerate on each camera
        for camid, item in self.cam_proxy_map.items():
            same_cam_samples = self._get_same_cam_samples(camid, dataset)
            
            # Step 2: get each proxy's centroid and initialize memory
            for i in range(item['proxy_num']):
                centroid = self._cal_proxy_centroid(same_cam_samples, i, features)
                # self.storage[camid][i,:] = centroid
                self.storage[i+dataset.cam_proxy_map[camid]['cam_index'],:] = centroid

    def _get_same_cam_samples(self, camid, dataset):
        indices_and_camid = [(idx, sample[3]) for idx, sample in enumerate(dataset.samples) if sample[1] == camid]
        return indices_and_camid

    def _cal_proxy_centroid(self, same_cam_samples, proxy_id, features):
        feature_proposals = [features[idx,:].view((1, -1)) for idx, proxy_label in same_cam_samples if proxy_id == proxy_label]
        feature_proposals = torch.cat(feature_proposals, dim=0)
        res = torch.mean(feature_proposals, dim=0, keepdim=True)
        return torch.nn.functional.normalize(res, dim=1) # L2 normalization

    def _init_storage(self):
        storage = []
        for camid, item in self.cam_proxy_map.items():
            storage.append(torch.zeros((item['proxy_num'], self.feature_dims)))
        return torch.cat(storage, dim=0)


        # storage = {}
        # for camid, item in self.cam_proxy_map.items():
        #     storage[camid] = torch.zeros((item['proxy_num'], self.feature_dims))
        # return storage



    def forward(self, input_features, camids, proxy_labels, abs_proxy_labels):
        # intra-camera loss
        # input_intra = _update_memory_intra(input_features, self.storage, camids, proxy_labels, self.momentum)
        input_intra = _update_memory_intra(input_features, self.storage, abs_proxy_labels, proxy_labels, self.momentum)
        input_intra /= self.temp
        logits_intra = F.log_softmax(input_intra, dim=1) # shape: (bsize, proxy_num)
        # targets_intra = self._get_abs_proxy_labels(camids, proxy_labels) # need a global absolute proxy label [0, proxy_num-1] to assign classfication labels
        
        loss_intra = F.nll_loss(logits_intra, abs_proxy_labels)
        # import ipdb; ipdb.set_trace()

        # return loss_intra, 0
        
        # TODO: inter-camera loss
        pass