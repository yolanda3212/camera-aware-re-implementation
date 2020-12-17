import torch, sys
sys.path.append('/home/ljc/works/fast-reid')
from dataset_wrapper import CustomDataset
from fastreid.data.datasets import Market1501, VeRi

dataset = Market1501(root='/home/ljc/datasets', mode='train')
dataset = CustomDataset((256,256), dataset, mode='train')
loader = torch.utils.data.DataLoader(dataset, batch_size=1)

for i, (imgs, paths, pids, camids) in enumerate(loader):
    print(paths)
    input()