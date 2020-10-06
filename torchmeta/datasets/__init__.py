from torchmeta.datasets.triplemnist import TripleMNIST
from torchmeta.datasets.doublemnist import DoubleMNIST
from torchmeta.datasets.cub import CUB
from torchmeta.datasets.cifar100 import CIFARFS, FC100
from torchmeta.datasets.miniimagenet import MiniImagenet
from torchmeta.datasets.omniglot import Omniglot
from torchmeta.datasets.tieredimagenet import TieredImagenet
from torchmeta.datasets.tcga import TCGA

from torchmeta.datasets.vgg_flower import VggFlower
from torchmeta.datasets.aircraft import AirCraft
from torchmeta.datasets.traffic_sign import TrafficSign
from torchmeta.datasets.svhn import SVHN
from torchmeta.datasets.cars import CARS

from torchmeta.datasets import helpers

__all__ = [
    'TCGA',
    'Omniglot',
    'MiniImagenet',
    'TieredImagenet',
    'CIFARFS',
    'FC100',
    'CUB',
    'DoubleMNIST',
    'TripleMNIST',
    'VggFlower',
    'AirCraft',
    'TrafficSign',
    'SVHN',
    'CARS',
    'helpers'
]
