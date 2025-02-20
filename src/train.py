from .utils import *
from .model import Model
from .evaluation import evaluate
from .dataset import PatchDataset

from torch import optim
import random
import numpy as np
import torch

def train(dataset_name,
    patch_size,
    patch_num,
    lr = 0.001, 
    weight_decay = 0,
    train_batch_size = 8,
    val_batch_size = 8,
    num_epoch = 20,
    num_workers = 2,
    save_interval = 1):
    train_dataset = PatchDataset(dataset_name, 'train', patch_size, patch_num)
    val_dataset = PatchDataset(dataset_name, 'valid', patch_size, patch_num)

    print(len(train_dataset), len(val_dataset))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(patch_size, patch_num, ).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    
    optimizer = optim.Adam(model.parameters(), lr=lr, 
                           weight_decay = weight_decay)
    train_loader = DataLoader(
        train_dataset,
        batch_size = train_batch_size,
        shuffle=True,
        num_workers = num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = val_batch_size,
        shuffle=True,
        num_workers = num_workers,
        collate_fn=val_dataset.collate_fn,
    )
    losses = []
    for epoch in range(num_epoch):
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            for index, (imgs, perms) in enumerate(pbar):
                optimizer.zero_grad()
                loss = model.get_loss(imgs, perms)
                #if(index % 10 == 0):
                #    print(model.predict(imgs)[0], perms[0])
                loss.backward()
                #if(index % 1000 == 0):
                #    model.print_gradient()
                optimizer.step()
                
                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" %
                                     (epoch + 1, np.mean(losses),
                                      optimizer.param_groups[0]['lr']))

        if((epoch+1) % save_interval == 0):
            torch.save(
                model.state_dict(),
                os.path.join(os.path.dirname(__file__), f'{DIR}/../log/model/{epoch+1}_model.pt'))
            acc = evaluate(model, device, val_loader)
            pickle.dump((losses, acc), open(f'{DIR}/../log/model/{epoch+1}_data.pkl', "wb"))

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train(
        dataset_name = PARAMS["dataset"],
        patch_size = PARAMS["patch_size"],
        patch_num = PARAMS["patch_num"],
        lr = PARAMS["lr"],
        weight_decay = PARAMS["weight_decay"],
        train_batch_size = PARAMS["train_batch_size"],
        val_batch_size = PARAMS["valid_batch_size"],
        num_epoch = PARAMS["num_epoch"],
        num_workers = PARAMS["num_workers"],
        save_interval = PARAMS["save_interval"]
    )