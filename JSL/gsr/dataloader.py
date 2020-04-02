from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import pdb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

# from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import json
import cv2
from PIL import Image
torch.multiprocessing.set_sharing_strategy('file_system')


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, verb_info, is_training, inference=False, inference_verbs=None, transform=None, is_visualizing=False):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.inference = inference
        self.inference_verbs = inference_verbs
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        self.is_visualizing = is_visualizing
        self.is_training = is_training

        self.color_change = transforms.Compose([transforms.ColorJitter(hue=.05, saturation=.05, brightness=0.05), transforms.RandomGrayscale(p=0.3)])

        with open(self.class_list, 'r') as file:
            self.classes, self.idx_to_class = self.load_classes(csv.reader(file, delimiter=','))

        if not self.inference:
            self.labels = {}
            for key, value in self.classes.items():
                self.labels[value] = key

            with open(self.train_file) as file:
                SWiG_json = json.load(file)
            self.image_data = self._read_annotations(SWiG_json, verb_info, self.classes)
            self.image_names = list(self.image_data.keys())
        else:
            self.image_names = []
            with open(train_file) as f:
                for line in f:
                    self.image_names.append(line.split('\n')[0])


        self.verb_to_idx = {}
        self.idx_to_verb = []
        with open('./global_utils/verb_indices.txt') as f:
            k = 0
            for line in f:
                verb = line.split('\n')[0]
                self.idx_to_verb.append(verb)
                self.verb_to_idx[verb] = k
                k += 1

        self.image_to_image_idx = {}
        i = 0
        for image_name in self.image_names:
            self.image_to_image_idx[image_name] = i
            i += 1


    def load_classes(self, csv_reader):
        result = {}
        idx_to_result = []
        for line, row in enumerate(csv_reader):
            line += 1
            class_name, class_id = row
            class_id = int(class_id)
            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
            idx_to_result.append(class_name.split('_')[0])

        return result, idx_to_result


    def make_dummy_annot(self):
        annotations = np.zeros((0, 7))


        # parse annotations
        for idx in range(6):
            annotation = np.zeros((1, 7))  # allow for 3 annotations

            annotations = np.append(annotations, annotation, axis=0)

        return annotations


    def __len__(self):
        #return 16
        return len(self.image_names)


    def __getitem__(self, idx):

        img = self.load_image(idx)
        if self.inference:
            verb_idx = self.inference_verbs[self.image_names[idx]]
            annot = self.make_dummy_annot()
            sample = {'img': img, 'annot': annot, 'img_name': self.image_names[idx], 'verb_idx': verb_idx}
            if self.transform:
                sample = self.transform(sample)
            return sample

        annot = self.load_annotations(idx)
        verb = self.image_names[idx].split('/')[2]
        verb = verb.split('_')[0]

        verb_idx = self.verb_to_idx[verb]
        sample = {'img': img, 'annot': annot, 'img_name': self.image_names[idx], 'verb_idx': verb_idx}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):

        im = Image.open(self.image_names[image_index])
        im = im.convert('RGB')

        if self.is_training:
            im = np.array(self.color_change(im))
        else:
            im = np.array(im)


        return im.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 7))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            annotation = np.zeros((1, 7))  # allow for 3 annotations

            annotation[0, 0] = a['x1']
            annotation[0, 1] = a['y1']
            annotation[0, 2] = a['x2']
            annotation[0, 3] = a['y2']

            annotation[0, 4] = self.name_to_label(a['class1'])
            annotation[0, 5] = self.name_to_label(a['class2'])
            annotation[0, 6] = self.name_to_label(a['class3'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations


    def _read_annotations(self, json, verb_orders, classes):
        result = {}

        for image in json:
            total_anns = 0
            verb = json[image]['verb']
            order = verb_orders[verb]['order']
            img_file = './images_512/' + image
            result[img_file] = []
            for role in order:
                total_anns += 1
                [x1, y1, x2, y2] = json[image]['bb'][role]
                class1 = json[image]['frames'][0][role]
                class2 = json[image]['frames'][1][role]
                class3 = json[image]['frames'][2][role]
                if class1 == '':
                    class1 = 'blank'
                if class2 == '':
                    class2 = 'blank'
                if class3 == '':
                    class3 = 'blank'
                if class1 not in classes:
                    class1 = 'oov'
                if class2 not in classes:
                    class2 = 'oov'
                if class3 not in classes:
                    class3 = 'oov'
                result[img_file].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class1': class1, 'class2': class2, 'class3': class3})

            while total_anns < 6:
                total_anns += 1
                [x1, y1, x2, y2] = [-1, -1, -1, -1]
                class1 = 'Pad'
                class2 = 'Pad'
                class3 = 'Pad'
                result[img_file].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class1': class1, 'class2': class2, 'class3': class3})

        return result


    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return 1

    def num_nouns(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    shift_0 = [s['shift_0'] for s in data]
    shift_1 = [s['shift_1'] for s in data]

    scales = [s['scale'] for s in data]
    img_names = [s['img_name'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]

    batch_size = len(imgs)
    max_width = 704
    max_height = 704

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, shift_0[i]:shift_0[i]+img.shape[0], shift_1[i]:shift_1[i]+img.shape[1], :] = img


    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 7)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 7)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'img_name': img_names, 'verb_idx': verb_indices,
            'widths': torch.tensor(widths).float(), 'heights': torch.tensor(heights).float(), 'shift_0': shift_0, 'shift_1': shift_1}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, is_for_training):
        self.is_for_training = is_for_training


    def __call__(self, sample, min_side=512, max_side=700):
        image, annots, image_name = sample['img'], sample['annot'], sample['img_name']

        rows_orig, cols_orig, cns_orig = image.shape
        smallest_side = min(rows_orig, cols_orig)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows_orig, cols_orig)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        if self.is_for_training:
            scale_factor = random.choice([1, 0.75, 0.5])
            scale = scale*scale_factor

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows_orig * scale)), int(round((cols_orig * scale)))))
        rows, cols, cns = image.shape

        new_image = np.zeros((rows, cols, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        shift_1 = int((704 - cols) * .5)
        shift_0 = int((704 - rows) * .5)

        annots[:, :4][annots[:, :4] > 0] *= scale

        annots[:, 0][annots[:, 0] > 0] = annots[:, 0][annots[:, 0] > 0] + shift_1
        annots[:, 1][annots[:, 1] > 0] = annots[:, 1][annots[:, 1] > 0] + shift_0
        annots[:, 2][annots[:, 2] > 0] = annots[:, 2][annots[:, 2] > 0] + shift_1
        annots[:, 3][annots[:, 3] > 0] = annots[:, 3][annots[:, 3] > 0] + shift_0


        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'img_name': image_name, 'verb_idx': sample['verb_idx'], 'shift_1': shift_1, 'shift_0': shift_0}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, flip_x=0.5):

        image, annots, img_name = sample['img'], sample['annot'], sample['img_name']
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0][annots[:, 0] > 0] = cols - x2[annots[:, 0] > 0]
            annots[:, 2][annots[:, 2] > 0] = cols - x_tmp[annots[:, 2] > 0]

            sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx']}

        sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx']}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, 'img_name': sample['img_name'], 'verb_idx': sample['verb_idx']}



class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
            # self.std = [1, 1, 1]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]