import gc
import os
import json
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import namedtuple
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from tensorflow.contrib.layers import xavier_initializer
from config import params, target_names

np.random.seed(0)
tf.set_random_seed(0)
tf.logging.set_verbosity(tf.logging.INFO)


def create_hidden_layer(n_hidden, prev_layer, name,
                        weights_initializer=xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        activation='relu'):
    W = tf.get_variable(
        'hidden_W' + name, shape=[prev_layer.shape.as_list()[1], n_hidden],
        initializer=weights_initializer, dtype=tf.float64)
    b = tf.get_variable(
        'hidden_b' + name, shape=[n_hidden], initializer=biases_initializer,
        dtype=tf.float64)
    if activation == 'relu':
        A = tf.nn.relu_layer(prev_layer, W, b, name=name)
    elif activation == 'tanh':
        A = tf.tanh(prev_layer @ W + b, name=name)
    else:
        A = tf.add(prev_layer @ W, b, name=name)
    return A


def BatchNormalization(x):
    mu, var = tf.nn.moments(x, axes=[0])
    return tf.nn.batch_normalization(
        x,
        mean=mu,
        variance=var,
        offset=0,
        scale=1,
        variance_epsilon=0.001,
    )


def mtl_model_fn(features, labels, mode, params):
    """Model function for MTL."""

    hidden_layers = params["hidden_layers"]
    starter_learning_rate = params["starter_learning_rate"]
    decay_steps = params["decay_steps"]
    decay_rate = params["decay_rate"]

    # Input Layer
    with tf.variable_scope("input_layer"):
        input_layer = tf.convert_to_tensor(features['x'], name='x')
        A = input_layer

    with tf.variable_scope("hidden_layers"):
        for key, item in hidden_layers.items():

            n_units = item['n_units']
            activation = item['activation']

            A = create_hidden_layer(n_units, A, "layer_" + key,
                                    activation=activation)

            if item["dropout"]:
                A = tf.layers.dropout(
                    inputs=A, rate=item["dropout"],
                    training=mode == tf.estimator.ModeKeys.TRAIN)

            if item["batch_normalization"]:
                A = BatchNormalization(A)

    loss = 0
    logits = {}
    predictions = {}
    uncertainties = {}
    eval_metric_ops = {}

    for target_name, label_type in target_names.items():

        with tf.variable_scope("uncertainties_"+target_name):
            # Uncertainty variable
            if label_type == "binary_classification":
                uncertainties[target_name] = tf.get_variable(
                    'uncertainties_'+target_name,
                    initializer=tf.constant(1.0, dtype=tf.float64),
                    trainable=params["with_uncertainty"]
                )
            elif label_type == "regression":
                uncertainties[target_name] = tf.get_variable(
                    "uncertainties_"+target_name,
                    initializer=tf.zeros_initializer(),
                    trainable=params["with_uncertainty"]
                )
            else:
                raise NotImplementedError

        tf.summary.scalar("uncertainties_"+target_name,
                          uncertainties[target_name])

        with tf.variable_scope("outputs_"+target_name):
            # Logits layer
            logits[target_name] = create_hidden_layer(
                1, A, activation='linear', name="logits_"+target_name
            )

            if label_type == "binary_classification":
                logits[target_name] = logits[target_name]/uncertainties[target_name]
                # Predictions output
                predictions[target_name] = tf.nn.sigmoid(
                    logits[target_name]/uncertainties[target_name],
                    name="probs_"+target_name
                )
            elif label_type == "regression":
                predictions[target_name] = logits[target_name]
            else:
                raise NotImplementedError

        tf.summary.histogram("Preds_"+target_name, predictions[target_name])

    export_outputs = {}
    for key, val in predictions.items():
        if key == params["default_target"]:
            export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                tf.estimator.export.PredictOutput(val)
        else:
            export_outputs[key] = tf.estimator.export.PredictOutput(val)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs=export_outputs)

    specific_losses = []
    for target_name, label_type in target_names.items():

        with tf.variable_scope("loss_"+target_name):

            if label_type == "binary_classification":
                specific_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels[target_name],
                        logits=logits[target_name]/uncertainties[target_name],
                    )
            elif label_type == "regression":
                specific_loss = tf.reduce_sum(
                    tf.exp(-uncertainties[target_name]) * tf.square(predictions[key] - labels[key]) + uncertainties[target_name], -1)
            else:
                raise NotImplementedError

            tf.summary.histogram("Loss_"+target_name, specific_loss)

            specific_loss = tf.reduce_mean(specific_loss)
            specific_losses.append(specific_loss)

            tf.summary.scalar("Loss_"+target_name, specific_loss)

        # Evaluation metrics
        eval_metric_ops[target_name] = tf.metrics.auc(
            labels=labels[target_name],
            predictions=predictions[target_name]
        )

        # Visualize AUROC score
        tf.summary.scalar("AUROC_"+target_name,
                          eval_metric_ops[target_name][1])

    with tf.variable_scope("total_loss"):
        # Accumulate loss for the whole tasks
        loss = tf.reduce_sum(specific_losses)
        tf.summary.scalar("Total_Loss", loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   decay_rate,
                                                   staircase=True)
        # Passing global_step to minimize() will increment it at each step.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_receiver_fn(num_features):
    """Build the serving inputs."""
    # The outer dimension (None) allows us to batch up inputs for
    # efficiency. However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.
    inputs = {"x": tf.placeholder(shape=[None, num_features],
                                  dtype=tf.float64)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def main():

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        filename='../logs/{}.log'.format(model_name))

    # Argparse format
    # Boilerplate for arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data", help="absolute path of train (input) data")
    parser.add_argument("model_path", help="absolute path to save models and preprocessors")

    # Parse arguments
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        # os.mkdir(model_path)

    # Training data
    train = pd.read_csv(args.train_data)

    logging.info("Train shape: ")
    logging.info(train.shape)

    feature_names = [x for x in train.columns if x not in target_names]

    Xtrain = train.as_matrix(columns=feature_names)

    # To impute with zeros
    Xtrain = np.nan_to_num(Xtrain)

    # By default Imputer() will impute using mean, but since we are imputing
    # with zeros, it is not doing anything. Comment `np.nan_to_num` to enable
    # Imputer()
    preprocessor = Pipeline([('imputer', Imputer()),
                             ('scaler', StandardScaler())])

    Xtrain = preprocessor.fit_transform(Xtrain)

    # Saving preprocessor
    joblib.dump(preprocessor, os.path.join(model_path, 'preprocessor.pkl'))

    logging.info("Target labels: {}".format(' '.join(target_names)))
    ytrain = {}
    for name in target_names:
        ytrain[name] = train.as_matrix(columns=[name]).astype('float64')

    del train
    gc.collect()

    with open(os.path.join(model_path, "model_config.json"), 'w') as f:
        json.dump(params, f)

    model = tf.estimator.Estimator(model_fn=mtl_model_fn,
                                   model_dir=model_path,
                                   params=params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": Xtrain},
        y=ytrain,
        batch_size=params["batch_size"],
        num_epochs=params["epochs"],
        shuffle=True,
    )

    model.train(input_fn=train_input_fn)

    def input_receiver_fn():
        return serving_input_receiver_fn(Xtrain.shape[1])

    export_dir = model.export_savedmodel(
        export_dir_base=model_path,
        serving_input_receiver_fn=input_receiver_fn
    )

    logging.info("SavedModel is exported to {}".format(export_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
