import numpy as np
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

from pyblur import LinearMotionBlur_random, DefocusBlur_random


def convex_coeff(alpha, orig, augmented):
    if alpha == 0:
        return orig
    alpha = np.random.uniform(alpha[0], alpha[1])
    return ((1 - alpha) * orig + (alpha * augmented)).astype(np.uint8)


class MotionBlur():
    def __init__(self, alpha=0):
        self.alpha = alpha

    def augment_image(self, im):
        augmented = np.dstack([LinearMotionBlur_random(im[..., i]) for i in range(3)])
        return convex_coeff(self.alpha, im, augmented)

    def augment_images(self, images, parents=None, hooks=None):
        return [self.augment_image(im) for im in images]
    

class CLAHE():
    def __init__(self, alpha=0):
        self.alpha = alpha

    def augment_image(self, im):
        lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2LRGB)
        return convex_coeff(self.alpha, im, augmented)

    def augment_images(self, images, parents=None, hooks=None):
        return [self.augment_image(im) for im in images]
    

class OneOf():
    def __init__(self, imgaugs, probs=None):
        self.probs = probs
        self.imgaugs = imgaugs

    def augment_image(self, im):
        imgaug = np.random.choice(self.imgaugs, p=self.probs)
        return imgaug.augment_image(im)

    def augment_images(self, images, parents=None, hooks=None):
        return [self.augment_image(im) for im in images]
    
    
class Crop():
    def __init__(self, window, central=False):
        self.window = np.array(window)
        self.central = central

    def augment_image(self, im, coords=None):
        if self.central:
            point  = np.array([
                (im.shape[0] - self.window[0]) // 2,
                (im.shape[1] - self.window[1]) // 2
            ])
        elif coords is not None:
            point = coords[np.random.randint(coords.shape[0])] - self.window // 2
            point = np.clip(point, 0, np.array(im.shape[:2]) - self.window)
        else:
            point = np.array([
                    np.random.randint(0, max(1, im.shape[0] - self.window[0] + 1)),
                    np.random.randint(0, max(1, im.shape[1] - self.window[1] + 1))
                ])

        return im[
            point[0]: point[0] + self.window[0], 
            point[1]: point[1] + self.window[1]
        ]

    def augment_images(self, images, parents=None, hooks=None):
        return [self.augment_image(im) for im in images]


class Rot90():
    def augment_image(self, im):
        return np.rot90(im, k=np.random.randint(0, 4))

    def augment_images(self, images, parents=None, hooks=None):
        return [self.augment_image(im) for im in images]


class Augmentation:
    def __init__(self, side, strength=1.):
        coeff = int(3 * strength)
        k = max(1, coeff if coeff % 2 else coeff - 1)
        median_blur = iaa.MedianBlur(k=(1, k))

        self.photometric = iaa.Sequential([
            iaa.Sometimes(
                .4, 
                iaa.OneOf([
                    CLAHE(alpha=(0 * strength, 1.0 * strength)),
                    iaa.Multiply((1 - .3 * strength, 1 + .3 * strength), per_channel=False),
                    iaa.ContrastNormalization(alpha=(.5 * strength, 1.5 * strength), per_channel=False)
                ])
            ),
            iaa.Sometimes(
                .3, 
                iaa.OneOf([
                  iaa.Sharpen(alpha=(0 * strength, .5 * strength), lightness=(1 - .25 * strength, 1 + .4 * strength)),
                  iaa.Emboss(alpha=(0 * strength, .5 * strength), strength=(1.0 * strength))
                ])
            ),
            iaa.Sometimes(
                .4, 
                [OneOf([
                    MotionBlur(alpha=(1., 1.)),
                    iaa.GaussianBlur((0 * strength, 3.0 * strength)),
                    median_blur,
                ], [.2, .3, .5])]
            ),
            iaa.Sometimes(.3, iaa.AddToHueAndSaturation((int(-10 * strength), int(10 * strength)), per_channel=.5)), # change hue and saturation
            iaa.Sometimes(.3, iaa.Grayscale(alpha=(0.0 * strength, .5 * strength))),
            iaa.Sometimes(.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0 * strength, 0.05 * 255 * strength), per_channel=0.5)),
        ])
        self.photometric = iaa.Sometimes(.9, self.photometric)


        self.geometric = iaa.Sequential([
            iaa.Sometimes(
                .6,
                iaa.OneOf([
                    #iaa.WithChannels([0, 1, 2], iaa.ElasticTransformation(alpha=(0, 1.))),
                    iaa.PerspectiveTransform(scale=(0 * strength, .25 * strength)),
                    iaa.PiecewiseAffine(scale=(0 * strength, .025 * strength)),
                ])
            ),
            iaa.Sometimes(
                .7,
                iaa.Affine(
                    scale={"x": (1 - .3 * strength, 1 + .2 * strength), "y": (1 - .3 * strength, 1 + .2 * strength)},
                    translate_percent={"x": (-0.2 * strength, 0.2 * strength), "y": (-0.2 * strength, 0.2 * strength)},
                    rotate=(-45, 45),
                    shear=(-16 * strength, 16 * strength),
                    order=[0, 1],
                    mode=['symmetric']
                ),
            ),
        ])

        self.geometric = iaa.Sequential([
            iaa.Fliplr(.5),
            iaa.Flipud(.5),
            Rot90(),
            iaa.Sometimes(.9, self.geometric),

        ])
        self.crop = Crop(window=(side, side), central=False)
        self.valid_geometric = Crop(window=(side, side), central=True)

    def __call__(self, image, mask, is_test=False, coords=None):
        patch = np.dstack([image, mask])
        if not is_test:
            patch = self.crop.augment_image(patch, coords)
        else:
            nlen = coords.shape[0] // 2
            patch = self.crop.augment_image(patch, coords[nlen: nlen + 1])

        if not is_test:
            patch = self.geometric.augment_image(patch)
        else:
            patch = self.valid_geometric.augment_image(patch)
        image, mask = patch[..., :3], patch[..., 3:]
        if not is_test:
            image = self.photometric.augment_image(image)
        return image, mask
