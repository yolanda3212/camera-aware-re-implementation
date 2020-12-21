import numpy as np
import torch, random
from torch.utils.data import Sampler

class ClusterSampler(Sampler):
    def __init__(self, data_source, cluster_num=8, samples_per_cluster=4):
        super(ClusterSampler, self).__init__(data_source)
        self.data_source = data_source
        self.cluster_num = cluster_num
        self.samples_per_cluster = samples_per_cluster

    def __iter__(self):
        # Step 1: choose cluster_num clusters
        batch_size = self.cluster_num * self.samples_per_cluster
        iterations = len(self.data_source) // batch_size if len(self.data_source) % batch_size == 0 else len(self.data_source) // batch_size + 1

        # Step 2: choose several samples from selected clusters
        iter_list = []
        labels = self.data_source.good_labels.reshape(-1)
        for i in range(iterations):
            cluster_cls_num = len(set(labels.tolist()))
            chosen_labels = random.sample(range(cluster_cls_num), min(self.cluster_num, cluster_cls_num)) # better implementation
            for l in chosen_labels:
                indices = torch.where(torch.tensor(labels)==l)[0]
                ids = random.sample(range(len(indices)), min(self.samples_per_cluster, len(indices))) # valid sampling
                iter_list.extend(indices[ids].tolist())
        return iter(iter_list)

    def __len__(self):
        return len(self.data_source)

if __name__ == "__main__":
    pass