import os
import re
import numpy as np
import pandas as pd
import json

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from lib.utils.utils import downIuDataset


class IuDataset(Dataset):
    def __init__(self, mode, tokenizer, imageSize=(768, 768), **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.root = 'iu-dataset'
        self.imagePath = os.path.join(self.root, 'images')
        if os.path.exists(self.root) == False:
            downIuDataset()
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(imageSize),
                transforms.RandomCrop(imageSize[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.dataFile = os.path.join(self.root, 'iu_xray_data.json')
        self.imageSize = imageSize
        self.mode = mode if mode != 'validate' else 'val'
        self.tokenizer = tokenizer
        self.data = self._loadData()
        self.keys = list(self.data.keys())
        only_findings = getattr(self, 'only_findings', False)
        path_tags = 'tags/iu_disease.txt' if only_findings else f'tags/iu.txt'
        with open(path_tags, 'r') as f:
            self.tags = f.read().splitlines()

    def _loadData(self):
        with open(self.dataFile) as f:
            data = json.load(f)
        data = data[self.mode]
        return data

    def cleanCaption(self, caption):
        def captionCleaner(t): return t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        def sentCleaner(t): return re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                          replace('\\', '').replace("'", '').strip().lower())
        tokens = [sentCleaner(sent) for sent in captionCleaner(
            caption) if sentCleaner(sent) != []]
        caption = ' . '.join(tokens) + ' .'
        return caption

    def __getitem__(self, index):
        imageName = self.keys[index]
        row = self.data[imageName]
        img = Image.open(os.path.join(
            self.imagePath, imageName)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        tag = [0]*len(self.tags)
        for i, label_name in enumerate(self.tags):
            tagKey = [i.lower() for i in row['tags_key']]
            if label_name in tagKey:
                tag[i] = 1.
        caption = row['caption']
        impression, findings = caption.split("<#findings#>")
        caption = findings+" "+impression
        caption = self.cleanCaption(caption)
        caption = self.tokenizer(caption)[:self.seq_length]
        mask = [1]*len(caption)
        return img, caption, mask, len(caption), tag

    def __len__(self):
        return len(self.keys)


class AtHotDataset(Dataset):
    def __init__(self, mode, tokenizer, imageSize=(768, 768), **kwargs):
        self.root = 'vn-dataset'
        self.imagePath = os.path.join(self.root, 'images')
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(imageSize),
                transforms.RandomCrop(imageSize[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.dataFile = os.path.join(self.root, 'ann.json')
        self.imageSize = imageSize
        self.mode = mode if mode != 'validate' else 'val'
        self.tokenizer = tokenizer
        self.data = self._loadData()
        self.keys = list(self.data.keys())
        with open('tags/vn.txt', 'r') as f:
            self.tags = f.read().splitlines()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _loadData(self):
        with open(self.dataFile) as f:
            data = json.load(f)
        data = data[self.mode]
        return data

    def __getitem__(self, index):
        imageName = self.keys[index]
        row = self.data[imageName]
        img = Image.open(os.path.join(
            self.imagePath, imageName)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        tag = [0]*len(self.tags)
        for i, label_name in enumerate(self.tags):
            tagKey = [i.lower() for i in row['tag']]
            if label_name in tagKey:
                tag[i] = 1.
        findings = row['finding']
        impression = row['impression']
        caption = findings+" "+impression
        caption = self.tokenizer(caption)[:self.seq_length]
        mask = [1]*len(caption)
        return img, caption, mask, len(caption), tag

    def __len__(self):
        return len(self.keys)


def collate_fn(data):
    images, captionIds, captionsMasks, seqLengths, tags = zip(*data)
    images = torch.stack(images, 0)
    maxSeqLength = max(seqLengths)

    targets = np.zeros((len(captionIds), maxSeqLength), dtype=int)
    targetsMasks = np.zeros((len(captionIds), maxSeqLength), dtype=int)

    for i, reportIds in enumerate(captionIds):
        targets[i, :len(reportIds)] = reportIds

    for i, reportMasks in enumerate(captionsMasks):
        targetsMasks[i, :len(reportMasks)] = reportMasks
    return images, torch.LongTensor(targets), torch.FloatTensor(targetsMasks), torch.FloatTensor(tags)
