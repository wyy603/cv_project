from PIL import Image
import torch
import os, shutil

from .utils import image_to_tensor
from .model import Model

def pachify(from_name, to_folder, n, p):
    image = Image.open(from_name)
    if(image.mode != 'RGB'):
        print(f"Warning. The mode {image.mode} is not RGB. We convert it to RGB.")
        image = image.convert('RGB')

    if(min(image.size[0], image.size[1]) < n * p):
        print(f"The image's size ({image.size[0]}, {image.size[1]}) is less than {n * p}.")
        return 1
    if os.path.exists(to_folder):
        shutil.rmtree(to_folder)
    os.makedirs(to_folder)
    perm = torch.randperm(n * n).tolist()
    for i in range(n):
        for j in range(n):
            image_crop = image.crop((i * p, j * p, (i + 1) * p, (j + 1) * p))
            image_crop.save(os.path.join(to_folder, f"{perm[i * n + j]}.png"))
    return 0

def load_model(model_path, p, n):
    model = Model(p, n)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to("cpu")
    return model

def inference(from_folder, model, n, p):
    x = torch.zeros((1, n * n, 3, p, p))
    for i in range(n * n):
        image = Image.open(os.path.join(from_folder, f"{i}.png"))
        if(image.mode != 'RGB'):
            print(f"Warning. The mode {image.mode} of the {i}'th image is not RGB. We convert it to RGB.")
            image = image.convert('RGB')
        x[0, i] = image_to_tensor(image)
    perm = model.predict_hungarian(x)
    return perm