from .utils import *

class PatchDataset(Dataset):
    def __init__(self, data_path, split, patch_size, patch_num, min_patch_num = 2, maxlen = None):
        self.patch_size = patch_size
        self.patch_num = patch_num
        
        self.image_name = []
        dir_path = os.path.join(DIR, f"../datasets/{PARAMS['dataset']}")
        with open(os.path.join(dir_path, "name.json"), "r") as f:
            dataset_name = json.load(f)
        for elem in dataset_name:
            if(elem["split"] == split):
                self.image_name.append(os.path.join(dir_path, elem["name"]))

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        image = Image.open(self.image_name[index])
        H, W = image.size
        p = self.patch_size
        w, h = self.patch_num, self.patch_num
        #w, h = H // p, W // p
        #w, h = min(w, h, self.patch_num), min(w, h, self.patch_num)
        perm = torch.randperm(w * h)
        x = random.randint(0, H - w * p)
        y = random.randint(0, W - h * p)
        image = image.crop((x, y, x + w * p, y + h * p))
        tensors = patchify_tensor(image_to_tensor(image).unsqueeze(0), p, w, h)[0]
        return tensors[perm], perm

    def collate_fn(self, collection):
        images = [i[0] for i in collection]
        perms = [i[1] for i in collection]
        return torch.stack(images, dim = 0), torch.stack(perms, dim = 0)

if __name__ == '__main__':
    dataset = PatchDataset(PARAMS['dataset'], 'train', 64, 5)
    output = dataset[0]
    loader = DataLoader(
        dataset,
        batch_size = 8,
        shuffle=True,
        num_workers = 2,
        collate_fn=dataset.collate_fn,
    )
    for images,perms in loader:
        print(images.shape,perms.shape)