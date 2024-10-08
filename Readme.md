# Algal Bloom Prediction

##### Author: Levi (nguyenhongson.kstn.hust@gmail.com) et. al.

#### Credits: This problem is given by Assoc. Prof. Le Chi Ngoc. For more details, please contact Prof. Le Chi Ngoc.

## Description

This is an educational research framework for Algal Bloom Prediction problem. This work is closely related to this publication:

- [Application of Machine Learning for eutrophication analysis and algal bloom prediction in an urban river: A 10-year study of the Han River, South Korea](https://www.sciencedirect.com/science/article/abs/pii/S0048969721041127?fbclid=IwY2xjawE_-45leHRuA2FlbQIxMAABHUKnb-7glbIBPBUhYsdWqfZk1zfe6rURGLAfg1HlmiNtIoWnKYpe8HWAjw_aem_i5PeffJUSm-RtxDPKi2ZXA)

### Dataset

Dataset contains the concentration of multiple chemical substances. The ultimate goal is to predict algal bloom.

### Model development suggestion

This framework is designed for easy developing and experimenting the models and training strategies.
Therefore, the bellow steps are expected to be implemented already:

- Load data and split train test
- Create data loader
- Training/validating functions for different problem types: Regression and Classification

The framework is designed using PyTorch. All models inherit the [BaseModel](/models/base_pytorch_model.py).
The child class must implement two functions:

- _build_network: Initiate a model object to be trained.
- calculate_detailed_report: Return the detailed evaluation score (e.g. Precision/Recall/F1)

See [basic example](/tests/test_regression.py) for more details.

## To set up new dev environment

```shell
conda env create -f environment.yml -n water-quality
```

## To activate the environment

```shell
conda activate water-quality
export PYTHONPATH=.
```

https://docs.google.com/spreadsheets/d/1FQJqFbZk1K2fM3mbL_lvdDPBf7TbWIhlGlFnzy5-tzg/edit?gid=1849550514#gid=1849550514

