# CSC 2503 Project Code for Attacking Vision Transformers 

Authors: Anthony DiMaggio and Angela (Yuxin) Sha

## Set up 

To run the code in this repository, there are a few set up steps and dependencies. First, clone a version of the [DINOv2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file) repository to your local machine:

```
git clone git@github.com:facebookresearch/dinov2.git
```

Then, simply run with local dependencies using `uv` (recommended). Installation guide for uv is [here](https://docs.astral.sh/uv/getting-started/installation/) and works out of the box:

```
uv run model/dino.py
uv run model/poison.py
```

## Our experiments 

- `dino.py` is a a local model instance loader for DINOv2 ViT with zero-shot classification definition
- `poison.py` is a script to run Unified Concept Editing (UCE) on ViTs. The sample code in the main body shows how to run poisoning experiment evaluation before and after the attack.
    - Run `run_poison_iteration` to run the entire UCE experiment pipeline with a source, target concept config.
- `gen_poison.py` generates adversarial examples using the Nightshade FGSM algorithm (our baseline)
    - Command `uv run model/gen_poison.py` generates images and saves them to `poisoned` directory. They can be classified with `uv run model/classify.py`
    - Note that the script is currently configured for ImageNet samples, but can be easily set to use WikiArt.

To run experiments for WikiArt and ImageNet datasets, follow the directions in `data/README.md` to access the datasets. 
