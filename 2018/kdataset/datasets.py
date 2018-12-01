
import os, sys, pdb, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from tensorboard_logger import configure, log_value
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torchvision
from torchvision import transforms, datasets, models
import random
from shutil import copyfile
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time
from shutil import copyfile
from os.path import isfile, join, abspath, exists, isdir, expanduser
from os import listdir, makedirs, getcwd, remove
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torchvision
from torchvision import transforms, datasets, models
import random
import sys
from glob import glob
import fnmatch

class GenericDataset(Dataset):
  def __init__(self, labels, root_dir, subset=False, transform=None):
    self.labels = labels
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    img_name = self.labels.iloc[idx, 0]  # file name
    fullname = join(self.root_dir, img_name)
    image = Image.open(fullname).convert('RGB')
    labels = self.labels.iloc[idx, 2]  # category_id
    #         print (labels)
    if self.transform:
      image = self.transform(image)
    return image, labels

  @staticmethod
  def find_classes(fullDir):
    classes = [d for d in os.listdir(fullDir) if os.path.isdir(os.path.join(fullDir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    num_to_class = dict(zip(range(len(classes)), classes))

    train = []
    for index, label in enumerate(classes):
      path = fullDir + label + '/'
      for file in listdir(path):
        train.append(['{}/{}'.format(label, file), label, index])

    df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])

    return classes, class_to_idx, num_to_class, df

  @staticmethod
  def find_classes_melanoma(fullDir):

    df_labels = pd.read_csv("melanoma_labels.csv", sep=',')
    class_to_idx= {'benign': 0, 'malignant': 1}
    num_to_class= {0: 'benign', 1: 'malignant'}
    #
    classes = [d for d in os.listdir(fullDir) if os.path.isdir(os.path.join(fullDir, d))]
    print('Classes: {}'.format(classes))
    print('class_to_idx: {}'.format(class_to_idx))
    print('num_to_class: {}'.format(num_to_class))

    train = []
    for index, row in (df_labels.iterrows()):
      id = (row['image_id'])
      # currImage = os.path.join(fullDir, num_to_class[(int(row['melanoma']))] + '/' + id + '.jpg')
      currImage_on_disk = os.path.join(fullDir, id + '.jpg')
      if os.path.isfile(currImage_on_disk):
        if (int(row['seborrheic_keratosis']))==1 or (int(row['melanoma']))==1:
          trait='malignant'
        else:
          trait = 'benign'

        train.append(['{}'.format(currImage_on_disk), trait, class_to_idx[trait]])

    # train.append(['{}'.format(currImage_on_disk), num_to_class[(int(row['seborrheic_keratosis']))],
    #               class_to_idx[num_to_class[(int(row['seborrheic_keratosis']))]]])
    #
    df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])
    df.to_csv('full_melanoma_labels.csv', index=None)
    return classes, class_to_idx, num_to_class, df


  @staticmethod
  def find_classes_breast(fullDir):
    imageList = glob(fullDir + '/**/*.png', recursive=True)

    benign = '*class0.png'
    malignant = '*class1.png'
    classZero = fnmatch.filter(imageList, benign)
    classOne = fnmatch.filter(imageList, malignant)
    #         print("benign:",classZero[0:5],'\n')
    #         print("malignant:",classOne[0:5])

    class_to_idx = {'benign': 0, 'malignant': 1}
    num_to_class = {0: 'benign', 1: 'malignant'}
    classes = ['benign', 'malignant']

    print('Classes: {}'.format(classes))
    print('class_to_idx: {}'.format(class_to_idx))
    print('num_to_class: {}'.format(num_to_class))

    train = []
    for currImage_on_disk in classZero:
      if os.path.isfile(currImage_on_disk):
        train.append(['{}'.format(currImage_on_disk), 'benign', class_to_idx['benign']])

    for currImage_on_disk in classOne:
      if os.path.isfile(currImage_on_disk):
        train.append(['{}'.format(currImage_on_disk), 'malignant', class_to_idx['malignant']])

    df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])
    df.to_csv('full_labels_breast.csv', index=None)
    return classes, class_to_idx, num_to_class, df

