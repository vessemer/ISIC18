import pickle 
import os
from glob import glob
import cv2
import numpy as np
import joblib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import ToTensor, Normalize, Compose



MEAN = [0.57053024, 0.54612805, 0.76352107]
STD = [0.09505704, 0.0857232 , 0.09826472]

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.57053024, 0.54612805, 0.76352107], std=[0.09505704, 0.0857232 , 0.09826472])
])

APPEARENCE = [1113, 6705, 514, 327, 1099, 115, 142]
ATTRIBUTES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

INIT_APPEARENCE = [1.] * 7


class ISIC_Dataset(Dataset):
    def __init__(self, root, df=None, augmentator=None, aug_params={},
                 appearence_mean=INIT_APPEARENCE, is_test=False, 
                 part=0, partsamount=1, exclude=False, seed=None, 
                 apply_transform=True):
        self.augmentator = augmentator
        self.aug_params = aug_params.copy()
        self.apply_transform = apply_transform
        self.augmentations = None
        if augmentator is not None:
            self.augmentations = augmentator(**self.aug_params)

        self.is_test = is_test
        self.paths = {}
        
        template=os.path.join(root, '*.jpg')
        paths = sorted(glob(template))

        if seed is not None:
            rs = np.random.RandomState(seed=seed)
            rs.shuffle(paths)

        step = len(paths) // partsamount

        if exclude:
            paths = paths[:part * step] + paths[(part + 1) * step:]
        else:
            paths = paths[part * step : (part + 1) * step]

        for path in tqdm(paths):
            key = os.path.basename(path).split('.')[0]
            cls = np.where(df.query('image==@key').values[0, 1:])[0] if df is not None else None
            self.paths[key] = {
                'image': path,
                'class': cls
            }

        self.keys = list(self.paths.keys())
        self.appearence_mean = appearence_mean

    def aug_strength_decay(self, decay=1.):
        self.aug_params['strength'] *= decay
        self.augmentations = self.augmentator(**self.aug_params)

    def converge_appearence(self, decay=1.):
        self.appearence_mean = [
            (1 - decay) * sa + decay * ga 
            for sa, ga in zip(self.appearence_mean, APPEARENCE)
        ]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = idx
        if isinstance(idx, int): 
            key = self.keys[idx]

        img = cv2.imread(self.paths[key]['image'])
        if self.augmentations is not None:
            img = self.augmentations(img, is_test=self.is_test)
        return (
            img_transform(img) if self.apply_transform else img, 
            self.paths[key]['class'] 
        )
        #{
        #    'images': img_transform(img), 
        #    'class': self.paths[key]['class']#torch.tensor(self.paths[key]['class']).long()
        #}


class EqualizedSampler(Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, df, rate=.8):
        self.data_source = data_source
        self.keys = data_source.keys
        keys = self.keys.copy()
        df_ = df.query('image==@keys')
        self.attributes_keys = [
            df_.query('{}==1'.format(attr)).image.values 
            for attr_id, attr in enumerate(ATTRIBUTES)
        ]
        self.rate = rate

    def __iter__(self):
        idxs = list()
        self.recompute_probs()
        iters = int(len(self.data_source) * self.rate)
        for i in range(iters):
            attr_id = np.random.choice(np.arange(len(self.probs)), p=self.probs)
            key = np.random.choice(self.attributes_keys[attr_id])
            idxs.append(
                self.data_source.keys.index(key)
            )

        return iter(idxs)

    def __len__(self):
        return int(len(self.data_source) * self.rate)

    def recompute_probs(self):
        print('appearence recomputed')
        self.probs = np.array(self.data_source.appearence_mean)
        #self.probs = np.array(self.probs) ** .3
        self.probs /= self.probs.sum()
        print(self.probs)


class ValSampler(Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, df):
        self.data_source = data_source
        self.keys = data_source.keys.copy()

    def __iter__(self):
        return iter(np.arange(len(self.keys)))

    def __len__(self):
        return len(self.data_source)
