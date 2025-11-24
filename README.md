# CSC 2503 Project Code for Attacking Vision Transformers 

Authors: Anthony DiMaggio and Angela (Yuxin) Sha

## Set up 

To run the code in this repository, there are a few set up steps and dependencies. First, clone a version of the [DINOv2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file) repository to your local machine:

```
git clone git@github.com:facebookresearch/dinov2.git
```

Then, simply run with local dependencies using `uv` (recommended)
```
uv run model/model.py
```