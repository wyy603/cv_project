import os, sys, subprocess, shutil
DIR = os.path.dirname(__file__)

from tqdm import tqdm
import numpy as np
from PIL import Image
import pickle, yaml, math, random, json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

def get_params():
    with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params

PARAMS = get_params()

def image_to_tensor(image):
    process = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    numpy_img = np.array(image)
    tensor = torch.from_numpy(numpy_img).permute(2, 0, 1).float() / 255.0
    tensor = process(tensor)
    return tensor

def images_to_tensor(images):
    tensor_list = []
    for image in images:
        tensor_list.append(image_to_tensor(image))
    return tensor_list

def patchify_tensor(imgs, p, w, h):
    b = imgs.shape[0]
    x = imgs.reshape(shape=(b, 3, w, p, h, p))
    x = torch.einsum('abcdef->acebdf', x)
    x = x.reshape((b, w * h, 3, p, p))
    return x