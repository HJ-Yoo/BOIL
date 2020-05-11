import os
import pickle
from PIL import Image
import h5py
import json

import numpy as np
from tqdm import tqdm
import requests
import tarfile
import glob
import shutil
import collections
from scipy.io import loadmat

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_file_from_google_drive


class AirCraft(CombinationMetaDataset):
    """
    The Mini-Imagenet dataset, introduced in [1]. This dataset contains images 
    of 100 different classes from the ILSVRC-12 dataset (Imagenet challenge). 
    The meta train/validation/test splits are taken from [2] for reproducibility.

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `miniimagenet` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way" 
        classification.

    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`, 
        `meta_val` and `meta_test` if all three are set to `False`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes 
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root 
        directory (under the `miniimagenet` folder). If the dataset is already 
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from [this repository]
    (https://github.com/renmengye/few-shot-ssl-public/). The meta train/
    validation/test splits are over 64/16/20 classes.

    References
    ----------
    .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016). 
           Matching Networks for One Shot Learning. In Advances in Neural 
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)

    .. [2] Ravi, S. and Larochelle, H. (2016). Optimization as a Model for 
           Few-Shot Learning. (https://openreview.net/forum?id=rJY0-Kcll)
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = AirCraftClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(AirCraft, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)

class AirCraftClassDataset(ClassDataset):
    folder = 'aircraft'

    tar_url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(AirCraftClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('AirCraft integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return AirCraftDataset(index, data, class_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        chunkSize = 1024
        r = requests.get(self.tar_url, stream=True)
        with open(self.root+'/fgvc-aircraft-2013b.tar.gz', 'wb') as f:
            pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize):
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

        filename = os.path.join(self.root, 'fgvc-aircraft-2013b.tar.gz')
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)

        splits = {}
        splits['train'] = [
            "A340-300", "A318", "Falcon 2000", "F-16A/B", "F/A-18", "C-130", 
            "MD-80", "BAE 146-200", "777-200", "747-400", "Cessna 172", "An-12", 
            "A330-300", "A321", "Fokker 100", "Fokker 50", "DHC-1", "Fokker 70", 
            "A340-200", "DC-6", "747-200", "Il-76", "747-300", "Model B200", 
            "Saab 340", "Cessna 560", "Dornier 328", "E-195", "ERJ 135", "747-100", 
            "737-600", "C-47", "DR-400", "ATR-72", "A330-200", "727-200", "737-700", 
            "PA-28", "ERJ 145", "737-300", "767-300", "737-500", "737-200", "DHC-6", 
            "Falcon 900", "DC-3", "Eurofighter Typhoon", "Challenger 600", "Hawk T1", 
            "A380", "777-300", "E-190", "DHC-8-100", "Cessna 525", "Metroliner", 
            "EMB-120", "Tu-134", "Embraer Legacy 600", "Gulfstream IV", "Tu-154", 
            "MD-87", "A300B4", "A340-600", "A340-500", "MD-11", "707-320", 
            "Cessna 208", "Global Express", "A319", "DH-82"
            ]
        splits['test'] = [
            "737-400", "737-800", "757-200", "767-400", "ATR-42", "BAE-125", 
            "Beechcraft 1900", "Boeing 717", "CRJ-200", "CRJ-700", "E-170", 
            "L-1011", "MD-90", "Saab 2000", "Spitfire"
            ]
        splits['val'] = [
            "737-900", "757-300", "767-200", "A310", "A320", "BAE 146-300", 
            "CRJ-900", "DC-10", "DC-8", "DC-9-30", "DHC-8-300", "Gulfstream V", 
            "SR-20", "Tornado", "Yak-42"
            ]

        # Cropping images with bounding box same as meta-dataset.
        bboxes_path = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images_box.txt')
        with open(bboxes_path, 'r') as f:
            names_to_bboxes = [line.split('\n')[0].split(' ') for line in f.readlines()]
            names_to_bboxes = dict((name, map(int, (xmin, ymin, xmax, ymax))) for name, xmin, ymin, xmax, ymax in names_to_bboxes)
            
        # Retrieve mapping from filename to cls
        cls_trainval_path = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images_variant_trainval.txt')
        with open(cls_trainval_path, 'r') as f:
            filenames_to_clsnames = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]

        cls_test_path = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images_variant_test.txt')
        with open(cls_test_path, 'r') as f:
            filenames_to_clsnames += [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
            
        filenames_to_clsnames = dict(filenames_to_clsnames)
        clss_to_names = collections.defaultdict(list)
        for filename, cls in filenames_to_clsnames.items():
            clss_to_names[cls].append(filename)
                
        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))

            images = np.array([])
            classes = {}
            pre_idx = 0
            post_idx = 0

            for class_id, cls_name in enumerate(tqdm(splits[split])):
                pre_idx = post_idx

                cls_data = []
                for filename in sorted(clss_to_names[cls_name]):
                    file_path = os.path.join(self.root,
                                            'fgvc-aircraft-2013b',
                                            'data',
                                            'images',
                                            '{}.jpg'.format(filename))
                    img = Image.open(file_path)
                    bbox = names_to_bboxes[filename]
                    img = np.asarray(img.crop(bbox).resize((32, 32)))
                    cls_data.append(img)
                cls_data = np.array(cls_data)
                if images.shape[0] == 0:
                    images = cls_data
                else:
                    images = np.concatenate((images, cls_data), axis=0)

                post_idx = pre_idx + len(cls_data)
                classes[str(cls_id)] = list(range(pre_idx, post_idx))

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)
                
class AirCraftDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(AirCraftDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
