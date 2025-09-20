# Install
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

# Train

To adjust training settings, view `settings.py`.

## DDPM
```
python ddpm.py
```

## DDIM + CFG
```
python ddim_cfg.py
```

# Sample

## DDPM

```
python sample_ddpm.py
```

## DDIM

```
python sample_ddim_cfg.py
```

# Evaluate

First, you need to create a small subset of the training set by running:

```
python data_extract.py
```

This subset consists of 2000 images, each class has 20 images.
Then, calculating the FID by running:

```
python -m pytorch_fid folder1 folder2
```

For example
```
python -m pytorch_fid cifar100_train pipeline_output_fixed_class
```

Here is the current result when comparing to original CIFAR-100's training set.

| Name   | block_out_channels        | layers_per_block | FID   |
|-------|-------------------------|----------------|-------|
| Val set | -                       | -              | 28.55 |
| DDPM   | (64, 128, 128, 256)     | 2              | 63.37 |
| DDIM+CFG   | (64, 128, 128, 256)     | 4              | 55.57 |

# Visualization

You can create a 16x16 grid by running

```
python make_grid.py
```