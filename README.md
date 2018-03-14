# Multi Task Learning with Homoscedastic Uncertainty

Multi Task Learning has been widely used in different applications. However, the performance of the model heavily depends on the method used to combine the losses between various objectives. This repository contains the implementation in *Tensorflow*.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need to install the libraries below:

```
numpy
pandas
tensorflow
scikit-learn
```

### Scripts

Below are the functions of the scripts:

- [`config.py`](https://github.com/hardianlawi/MTL-Homoscedastic-Uncertainty/blob/master/scripts/config.py): contains the hyperparameters of the model and the configuration for the training such as the labels used as the objectives.
- [`estimators.py`](https://github.com/hardianlawi/MTL-Homoscedastic-Uncertainty/blob/master/scripts/estimators.py): contains the neural networks that import the hyperparameters and the configuration from `config.py` and train an `estimator` which then would be exported to a `SavedModel` format.

### Training

To train your model, you will need to have a file containing your dataset which includes both features and labels. Then, run the command below:

```
python estimators.py [--train_data] [--model_path]
```

Arguments:
```
--train_data : path to the dataset
--model_path : directory to store the outputs.
```

Example:
```
python estimators.py --train_data ~/Downloads/train.csv --model_path /tmp/model-v1.0.0
```

### Evaluation

For evaluation, we have amazing tool called `tensorboard` built by tensorflow developers to help us observe the training progress or even debug our network. You can run the command below:

```
tensorboard [--logdir] [--host optional] [--port optional]
```

Argument:
```
--logdir : the directory used to store the outputs during the training.
--host : ip address of the machine
--port : port you want to serve on.
```

### Inference

For inference, you could load the exported `SavedModel` and do an inference like below:

```python
import json
from tensorflow.contrib import predictor
from sklearn.externals import joblib


# Read configuration of training
with open("../model_config.json", 'r') as f:
    params = json.load(f)

# Load scikit learn preprocessor
preprocessor = joblib.load('../preprocessor.pkl')

# Load tensorflow predictor
predict_fn = predictor.from_saved_model(MODEL_PATH + 'goboard/' + model_name)

# Suppose xTest is a 2D numpy array
inference_x = {'x': xTest}

# Generate predictions
predictions = predict_fn(inference_x)['output']
```

## Deployment

After training the model in *Python*, tensorflow serving could be used to deploy the model.

## Authors

* [**Hardian Lawi**](https://github.com/hardianlawi)

## Acknowledgments

Please read the [paper](https://arxiv.org/abs/1705.07115) for more details of the model or check the github of one of the authors [github](https://github.com/yaringal/multi-task-learning-example) for the implementation in `keras`.

## References

- [Identifying beneficial task relations for multi-task learning in deep neural networks](http://www.aclweb.org/anthology/E17-2026)
- [Fully-adaptive Feature Sharing in Multi-Task Networks with Applications in Person Attribute Classification](https://arxiv.org/pdf/1611.05377.pdf)
- [Cross-stitch Networks for Multi-task Learning](https://arxiv.org/pdf/1604.03539.pdf)
- [An Overview of Multi-Task Learning in Deep Neural Networks](http://ruder.io/multi-task/)
- [Deep Learning Is Not Good Enough, We Need Bayesian Deep Learning for Safe AI](https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/)
- [Uncertainty in Deep Learning (PhD Thesis) | Yarin Gal - Blog | Cambridge Machine Learning Group](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html)
- [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
- [Geometry and Uncertainty in Deep Learning for Computer Vision](https://alexgkendall.com/media/presentations/oxford_seminar.pdf)
