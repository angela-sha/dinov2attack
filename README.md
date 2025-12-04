# CSC 2503 Project Code for Attacking Vision Transformers 

Authors: Anthony DiMaggio and Angela (Yuxin) Sha

## Set up 

To run the code in this repository, there are a few set up steps and dependencies. First, clone a version of the [DINOv2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file) repository to your local machine:

```
git clone git@github.com:facebookresearch/dinov2.git
```

Then, simply run with local dependencies using `uv` (recommended), for example:
```
uv run model/dino.py
uv run model/poison.py
```

To run experiments specific to the WikiArt and ImageNet datasets, follow the directions in `data/README.md` to access the datasets. 