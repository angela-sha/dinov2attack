# Datasets for Concept Poisoning 

## WikiArt 

Download Wikiart dataset from the (Kaggle link)[https://www.kaggle.com/datasets/steubk/wikiart?select=classes.csv]

```
curl -L -o ./wikiart.zip https://www.kaggle.com/api/v1/datasets/download/steubk/wikiart
```
<!-- Download the subset of Wikiart dataset from the (Kaggle link)[https://www.kaggle.com/c/painter-by-numbers]:
```
uv pip install kaggle
kaggle competitions download -c painter-by-numbers -f test.zip
```

To do so you must also have accept terms of the dataset and have API access. -->

## COCO

Download COCO captions locally:

```
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

Unzip and use validation set.

 <!-- TODO (angela-sha): add readme on how to locally download the datasets that we use  -->
