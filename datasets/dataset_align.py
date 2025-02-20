from PIL import Image
import os, sys, subprocess, shutil, random, json
from tqdm import tqdm
DIR = os.path.dirname(__file__)

def dataset_align(from_name, patch_size, patch_num, train_ratio = 0.8):
    from_path = os.path.join(DIR, from_name)
    dataset_name = []
    with tqdm(os.listdir(from_path)) as pbar:
        index = 0
        for filename in pbar:
            pbar.set_description(f"index: {index}")
            splitext = os.path.splitext(filename)
            if splitext[1] in {'.jpg', '.png', '.JPEG'}:
                file_path = os.path.join(from_path, filename)
                image = Image.open(file_path)
                w = image.size[0] // patch_size
                h = image.size[1] // patch_size
                if(image.mode != "RGB" or w < patch_num or h < patch_num):
                    continue
                dataset_name.append(filename)

    for index, name in enumerate(dataset_name):
        if(index <= len(dataset_name) * train_ratio):
            dataset_name[index] = {"name": name, "split": "train"}
        else:
            dataset_name[index] = {"name": name, "split": "valid"}
    with open(os.path.join(DIR, from_path, "name.json"), "w") as f:
        json.dump(dataset_name, f)

if __name__ == "__main__":
    #from_name = "imagenet1k_0"
    from_name = "DIV2K"
    dataset_align(from_name, 64, 5, train_ratio=0)