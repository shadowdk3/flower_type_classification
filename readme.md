# Flower Type Classification

## Environment 

- Jetson ORIN AGX
- Pytorch

## Install dependencies

```
pip install -U pip
pip install -r requirments.txt
pip install wheels/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install wheels/torchvision-0.15.1-cp38-cp38-manylinux2014_aarch64.whl
```

## Run Script

```
cd /path/to/project
chmod +x srcipt/setup-venv.sh
./srcipt/setup-venv.sh
```

## Run program

```
source venv/bin/activate
python src/components/data_ingestion.py
```

## Dataset

https://www.kaggle.com/datasets/lara311/flowers-five-classes

- Daisy: 501 images
- Dandelion: 646 images
- Rose: 497 images
- Sunflower: 495 images
- Tulip: 607 images

## Flask

![demo](/ref/flower_type_predict.gif)

## [Classification with resnet50](https://github.com/shadowdk3/flower_type_classification/blob/master/notebook/classification.ipynb)

### Dataset

![](/ref/dataset.png)

### Train Model

![](/ref/train.png)

### Predict

![](/ref/predict.png)
