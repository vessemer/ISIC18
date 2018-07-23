import os
from glob import glob
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


MEAN = [0.53659743, 0.58239675, 0.708252] 
STD = [0.12663189, 0.11222798, 0.09813331]

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.53659743, 0.58239675, 0.708252], std=[0.12663189, 0.11222798, 0.09813331])
])


class ISIC_Dataset(Dataset):
    def __init__(self, root, prefix='', postfix='_segmentation.png', augmentations=None, is_test=False, part=0, partsamount=1, exclude=False, seed=None):
        self.is_test = is_test
        self.paths = {}

        template = os.path.join(root, '*{}')
        mask_paths = sorted(glob(template.format(prefix + postfix)))

        if seed is not None:
            rs = np.random.RandomState(seed=seed)
            rs.shuffle(mask_paths)

        step = len(mask_paths) // partsamount

        if exclude:
            mask_paths = mask_paths[:part * step] + mask_paths[(part + 1) * step:]
        else:
            mask_paths = mask_paths[part * step : (part + 1) * step]
        
        for mpath in mask_paths:
            key = os.path.basename(mpath.split(postfix)[0])
            self.paths[key] = {
                'mask': mpath,
                'image': mpath.replace(postfix, '.jpg')
            }
        self.keys = list(self.paths.keys())
        self.augmentations = augmentations
        self.postfix = postfix

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[self.keys[idx]]['image'])
        mask = cv2.imread(self.paths[self.keys[idx]]['mask'], 0)
        mask = np.expand_dims(mask, -1)
        if self.augmentations is not None:
            img, mask = self.augmentations(img, mask, self.is_test)
        return {
            'images': img_transform(img), 
            'masks': torch.from_numpy(np.rollaxis(mask, 2, 0))
        }

    def shrink_all(self, prefix='_shrinked'):
        for key, el in tqdm(self.paths.items()):
            image = cv2.imread(el['image'])
            mask = cv2.imread(el['mask'])
            shape = np.array(image.shape[:2])
            coeff = shape.min() / 576#max(576, min(shape.min() // 2))
            shape = (shape / coeff).astype(np.int)
            image = cv2.resize(image, tuple(shape[::-1].tolist()), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, tuple(shape[::-1].tolist()), interpolation=cv2.INTER_AREA)

            cv2.imwrite(el['image'].replace('.jpg', '.'.join([prefix, 'jpg'])), image)
            cv2.imwrite(el['mask'].replace(self.postfix, prefix + self.postfix), mask)
