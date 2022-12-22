from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, config, images_dir, labels_dir):
        
        self.config = config
        self.subjects = []

        if (config.in_class == 3) and (config.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(config.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(config.fold_arch))

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)
        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(config.fold_arch))

            artery_labels_dir = Path(labels_dir+'/artery')
            self.artery_label_paths = sorted(artery_labels_dir.glob(config.fold_arch))

            lung_labels_dir = Path(labels_dir+'/lung')
            self.lung_label_paths = sorted(lung_labels_dir.glob(config.fold_arch))

            trachea_labels_dir = Path(labels_dir+'/trachea')
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(config.fold_arch))

            vein_labels_dir = Path(labels_dir+'/vein')
            self.vein_label_paths = sorted(vein_labels_dir.glob(config.fold_arch))


            for (image_path, artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths,self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    atery=tio.LabelMap(artery_label_path),
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
                )
                self.subjects.append(subject)


        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)
        
        self.sampler = tio.data.UniformSampler(config.patch_size)

        self.queue_dataset = Queue(
            self.training_set,
            max_length=config.queue_length,
            num_workers=config.num_workers,
            sampler=self.sampler,
            samples_per_volume=config.samples_per_volume,
        )


    def transform(self):

        if self.config.mode == '3d':
            if self.config.data_aug:
                training_transform = Compose([
                # ToCanonical(),
                CropOrPad((self.config.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),])
            else:
                training_transform = Compose([
                CropOrPad((self.config.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])
        elif self.config.mode == '2d':
            if self.config.data_aug:
                training_transform = Compose([
                # CropOrPad((config.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),])
            else:
                training_transform = Compose([
                # CropOrPad((config.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])

        else:
            raise Exception('no such kind of mode!')


        return training_transform




class MedData_test(torch.utils.data.Dataset):
    def __init__(self, config, images_dir, labels_dir):
        self.config = config

        self.subjects = []

        if (config.in_class == 3) and (config.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(config.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(config.fold_arch))

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)
        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(config.fold_arch))

            artery_labels_dir = Path(labels_dir+'/artery')
            self.artery_label_paths = sorted(artery_labels_dir.glob(config.fold_arch))

            lung_labels_dir = Path(labels_dir+'/lung')
            self.lung_label_paths = sorted(lung_labels_dir.glob(config.fold_arch))

            trachea_labels_dir = Path(labels_dir+'/trachea')
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(config.fold_arch))

            vein_labels_dir = Path(labels_dir+'/vein')
            self.vein_label_paths = sorted(vein_labels_dir.glob(config.fold_arch))


            for (image_path, artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths,self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    atery=tio.LabelMap(artery_label_path),
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
                )
                self.subjects.append(subject)


        # self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=None)




