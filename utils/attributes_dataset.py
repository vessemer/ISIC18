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



MEAN = [0.46764078, 0.52520324, 0.67566734]
STD = [0.13644579, 0.12332337, 0.09122486]

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.46764078, 0.52520324, 0.67566734], std=[0.13644579, 0.12332337, 0.09122486])
])

APPEARENCE = [0.23245952, 0.26291442, 0.07324595, 0.58712413, 0.0385505, 0.19814958]
INIT_APPEARENCE = [1.] * 5 + [.1]

attributes = [
    '_attribute_globules_{}.png',            # 255, 0, 0
    '_attribute_milia_like_cyst_{}.png',     # 255, 255, 0
    '_attribute_negative_network_{}.png',    # 255, 255, 255
    '_attribute_pigment_network_{}.png',     # 255, 0, 255
    '_attribute_streaks_{}.png',             # 0, 255, 0
    '_attribute_segmentation_{}.png'         # 0, 0, 255
]

# attributes = [
#     '_attribute_globules.png',            # 255, 0, 0
#     '_attribute_milia_like_cyst.png',     # 255, 255, 0
#     '_attribute_negative_network.png',    # 255, 255, 255
#     '_attribute_pigment_network.png',     # 255, 0, 255
#     '_attribute_streaks.png',             # 0, 255, 0
#     '_attribute_segmentation.png'         # 0, 0, 255
# ]

class ISIC_Dataset(Dataset):
#     def __init__(self, root, augmentator=None, aug_params={}, 
#                  amounts=None, appearence_mean=INIT_APPEARENCE, is_test=False, part=0, partsamount=1, exclude=False, seed=None):
#         self.augmentator = augmentator
#         self.aug_params = aug_params
#         self.augmentations = augmentator(**self.aug_params)

#         self.is_test = is_test
#         self.paths = {}
        
#         template=os.path.join(root, '*.jpg')
#         paths = sorted(glob(template))

#         for path in tqdm(paths):
#             key = os.path.basename(path).split('_')[1].split('.')[0]
#             tmp = os.path.join(root, ''.join(['ISIC_', key, '.jpg']))
#             self.paths[int(key)] = {
#                 'attributes': [
#                     path[:-4] + attr
#                     for attr in attributes
#                 ],
#                 'image': tmp,
#             }

#         self.keys = list(self.paths.keys())

    def __init__(self, root, augmentator=None, aug_params={}, 
                 amounts=None, appearence_mean=INIT_APPEARENCE, is_test=False, part=0, partsamount=1, exclude=False, seed=None):
        self.augmentator = augmentator
        self.aug_params = aug_params
        self.augmentations = augmentator(**self.aug_params)

        self.is_test = is_test
        self.paths = {}
        
        template=os.path.join(root, 'JPG', '*_0.jpg')
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
            key = os.path.basename(path).split('_')[1]
            tmp = os.path.join(root, 'JPG', '_'.join(['ISIC', key, '{}.jpg']))
            self.paths[int(key)] = {
                'attributes': [
                    path.replace('JPG', 'PNG')[:-6] + attr
                    for attr in attributes
                ],
                'image': tmp,
                'amount': amounts[int(key)] if (amounts is not None) 
                and (int(key) in amounts.keys()) else len(glob(tmp.format('*')))
            }

        self.keys = list(self.paths.keys())
        self.appearence_mean = appearence_mean
        self.appearence_keys = pickle.load(open('../data/attr_appearence_keys', 'rb'))
        self.appearence_keys = [
            [key for key in attr if key in self.keys] 
            for attr in self.appearence_keys
        ]

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

    def __getitem__(self, key_idx):
        key, idx = key_idx
        img = cv2.imread(self.paths[key]['image'].format(idx))
        mask = np.dstack([cv2.imread(attr.format(idx), 0) for attr in self.paths[key]['attributes']])
        if self.augmentations is not None:
            img, mask = self.augmentations(img, mask, is_test=self.is_test)
        return {
            'key': key,
            'images': img_transform(img), 
            'masks': torch.from_numpy(np.rollaxis(mask, 2, 0))
        }

    def shrink_all(self, out_dir):
        for key, el in tqdm(self.paths.items()):
            image = cv2.imread(el['image'])
            mask = [cv2.imread(attr, 0) for attr in el['attributes']]

            shape = np.array(image.shape[:2])
            coeff = shape.min() / max(576, shape.min() // 2)
            shape = (shape / coeff).astype(np.int)
            image = cv2.resize(image, tuple(shape[::-1].tolist()), interpolation=cv2.INTER_AREA)
            mask = [cv2.resize(m, tuple(shape[::-1].tolist()), interpolation=cv2.INTER_AREA) for m in mask]

            cv2.imwrite(os.path.join(out_dir, os.path.basename(el['image'])), image)
            for i, attr in enumerate(el['attributes']):
                cv2.imwrite(os.path.join(out_dir, os.path.basename(attr)), mask[i])

    def _get_crop(self, image, masks, window):
        coords = np.array(np.where(masks))[:2]
        coord_min, coord_max = coords.min(axis=1), coords.max(axis=1)

        point = None
        if np.prod((coord_max - coord_min) <= window):
            point = (coord_max + coord_min) // 2 - window // 2
            point = np.clip(point, 0, np.array(image.shape[:2]) - window)

        return self._crop(image, point, window), self._crop(masks, point, window)

    def _crop(self, image, point, window):
        if point is not None:
            return image[
                point[0]: point[0] + window[0], 
                point[1]: point[1] + window[1]
            ]
        else:
            return None

    def _get_patches_old(self, image, side=576, any_attr=False, subdivisions=2):
        step = int(side / subdivisions)
        aug = int(round(side * (1. - 1. / subdivisions)))
        more_borders = ((aug, aug), (aug, aug), (0, 0))
#         any_attr =  
        image = np.pad(image, pad_width=more_borders, mode='reflect')

        x_len = image.shape[0]
        y_len = image.shape[1]

        subdivs = []
        for i in range(0, x_len - side + 1, step):
            for j in range(0, y_len - side + 1, step):
                patch = image[i: i + side, j: j + side]
                subdivs.append(patch)
        return np.array(subdivs)

    def _get_patches(self, image, side=576, any_attr=False, subdivisions=2):
        step = int(side / subdivisions)
        shape = np.array(image.shape[:2])
        steps_nb = 1 + np.ceil((shape - side) / step).astype(np.int)
        x_points = np.linspace(0, shape[0] - side, num=steps_nb[0]).astype(np.int)
        y_points = np.linspace(0, shape[1] - side, num=steps_nb[1]).astype(np.int)
        
        subdivs = []
        for x in x_points:
            for y in y_points:
                patch = image[x: x + side, y: y + side]
                subdivs.append(patch)
        return np.array(subdivs)

    def _meta_crop(self, out_dir, el, side=576, is_full=False):
            images = cv2.imread(el['image'])
            masks = np.dstack([cv2.imread(attr, 0) for attr in el['attributes']])
            images, masks = self._apply_crops(images, masks, side, is_full=is_full)

            for i, (img, mask) in enumerate(zip(images, masks)):
                name = '_{}.'.join(os.path.basename(el['image']).split('.'))
                cv2.imwrite(os.path.join(out_dir, 'JPG', name.format(i)), img)
                for j, attr in enumerate(el['attributes']):
                    name = '_{}.'.join(os.path.basename(attr).split('.'))
                    cv2.imwrite(os.path.join(out_dir, 'PNG', name.format(i)), mask[..., j])

    def _apply_crops(self, image, masks, side=576, is_full=False):
        window = np.array([side, side])
        if not is_full:
            img, msk = self._get_crop(image, masks, window)
            if img is not None:
                return [img], [msk]

            any_attr = masks[..., :-1].sum()
            if any_attr:
                img, msk = self._get_crop(np.dstack([image, masks]), masks[..., :-1], window)
                if img is not None:
                    return [img[..., :3]], [img[..., 3:]]

        subdivs = self._get_patches(np.dstack([image, masks]), side, any_attr, subdivisions=2)
        if is_full:
            return subdivs[..., :3], subdivs[..., 3:]

        if any_attr:
            idxs = np.where(subdivs[..., 3: -1].sum(axis=(1, 2, 3)))
        else:
            idxs = np.where(subdivs[..., 3:].sum(axis=(1, 2, 3)))
        return subdivs[idxs, ..., :3][0], subdivs[idxs, ..., 3:][0]

    def crop_attributes(self, out_dir, side=576, is_full=False):
        joblib.Parallel(n_jobs=12)(
            joblib.delayed(self._meta_crop)(out_dir, el, side, is_full) 
            for key, el in train_dataset.paths.items()
        )

    def shrink_masks(self, scale_size=4):
        for key, el in tqdm(self.paths.items()):
            mask = [cv2.imread(attr, 0) for attr in el['attributes']]

            shape = np.array(mask[0].shape[:2])
            shape = (shape / 4).astype(np.int)
            mask = [cv2.resize(m, tuple(shape[::-1].tolist()), interpolation=cv2.INTER_AREA) for m in mask]

            for i, attr in enumerate(el['attributes']):
                cv2.imwrite(attr, mask[i])


class EqualizedSampler(Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, attribute_areas):
        self.data_source = data_source
        self.appearence_keys = self.data_source.appearence_keys
        self.attribute_areas = attribute_areas

    def __iter__(self):
        keys = list()
        self.recompute_probs()
        for i in range(len(self.data_source)):
            attr_id = np.random.choice(np.arange(len(self.probs)), p=self.probs)
            key = np.random.choice(self.appearence_keys[attr_id])

            idx = np.random.choice(self.data_source.paths[key]['amount'], p=self.attribute_areas[key][attr_id])
            keys.append((key, idx))

        return iter(keys)

    def __len__(self):
        return len(self.data_source)

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

    def __init__(self, data_source, attribute_areas):
        self.data_source = data_source
        self.keys = data_source.keys
        self.attribute_areas = attribute_areas

    def __iter__(self):
        indexes = list()
        for key in self.keys:
            areas = self.attribute_areas[key]
            if areas[:-1].sum():
                idx = np.argmax(areas[:-1].sum(0))
            else:
                idx = np.argmax(areas.sum(0))
            indexes.append((key, idx))
        return iter(indexes)

    def __len__(self):
        return len(self.data_source)
