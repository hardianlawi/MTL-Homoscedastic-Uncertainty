import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def do_inference(hostport):
    """Tests PredictionService with concurrent requests.
    Args:
    hostport: Host:port address of the Prediction Service.
    Returns:
    pred values, ground truth label
    """
    # create connection
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # initialize a request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'

    # Randomly generate some test data
    temp_data = np.random.randn(100, 76).astype(np.float64)
    data = temp_data
    request.inputs['x'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=data.shape))

    # predict
    result = stub.Predict(request, 10.0)
    return result


def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return

    predictions = do_inference(FLAGS.server)
    print(predictions)


if __name__ == '__main__':
    tf.app.run()
