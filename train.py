import torch
from dataset import VOCDetection
from utils import SSDAugmentation
from config import voc, MEANS

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

dataset_root = "./data/11.21/marked/"


def train():
    cfg = voc
    voc_dataset = VOCDetection(root=dataset_root,
                       transform=SSDAugmentation(cfg['min_dim'],
                                                 MEANS))
