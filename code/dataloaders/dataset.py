import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from skimage.color import rgb2gray


def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform

        # self.all_slices = os.listdir(self._base_dir + "/train")
        # print(f"Contents of train directory: {self.all_slices}")

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/train")
            self.sample_list = []
            for file_name in self.all_slices:
                if os.path.splitext(file_name)[1] == '.h5':
                    self.sample_list.append(file_name)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/val")
            self.sample_list = []
            for file_name in self.all_volumes:
                if os.path.splitext(file_name)[1] == '.h5':
                    self.sample_list.append(file_name)
        elif self.split == 'test':
            self.all_volumes = os.listdir(
                self._base_dir + "/test")
            self.sample_list = []
            for file_name in self.all_volumes:
                if os.path.splitext(file_name)[1] == '.h5':
                    self.sample_list.append(file_name)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/train/{}".format(case), 'r')
        elif self.split == "val":
            h5f = h5py.File(self._base_dir +
                            "/val/{}".format(case), 'r')
        elif self.split == "test":
            h5f = h5py.File(self._base_dir +
                        "/test/{}".format(case), 'r')


        image = h5f['image'][:]
        image = rgb2gray(image)/255.0
        
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': label}
        sample['label'][sample['label'] == 255] = 2
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 2 in np.unique(label):
                image, label = random_rotate(image, label, cval=2)
            else:
                image, label = random_rotate(image, label, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
