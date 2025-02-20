from .model import Model
from .utils import *
from .dataset import PatchDataset
from .inference import load_model

def evaluate(model, device, loader):
    model.eval()
    score = 0
    sample_num = 0
    with tqdm(loader, desc="evaluate") as pbar:
        for index, (imgs, perms) in enumerate(pbar):
            #pred = model.predict(imgs).cpu()
            pred = model.predict_hungarian(imgs).cpu()
            score += ((pred != perms).sum(axis = 1) == 0).sum()
            sample_num += imgs.shape[0]
            pbar.set_description(f"acc: {score / sample_num}")
    return score / sample_num

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("src/checkpoint/model_random_crop_63.pt", 64, 5).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    valid_dataset = PatchDataset(PARAMS["dataset"], 'valid', PARAMS["patch_size"], PARAMS["patch_num"])
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = PARAMS["valid_batch_size"],
        shuffle=True,
        num_workers = PARAMS["num_workers"],
        collate_fn=valid_dataset.collate_fn,
    )
    evaluate(model, device, valid_loader)