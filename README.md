# Report

The report is `doc/CV_Project.pdf`.

# Environment

1.

```
pip install torch torchvision numpy timm matplotlib scipy pyyaml tqdm
```

2. (In main directory but not in `src`)

```
pip install -e .
```

# Demo

Open `src/demo/demo.ipynb` (need IDEs to open jupyter notebook).

# Dataset

Put images in `datasets/[dataset_name]`, and run `datasets/dataset_align.py` to create `name.json`, it filters images with size not less than 320 and split them to train and validation. You can modify the code to modify the `[dataset_name]` and the ratio of training data.

# Training

`src/config.yaml` controls the training parameters.

Run `python -m src.train` to start a training.

