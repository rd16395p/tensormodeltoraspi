import time
import picamera
import picamera.array
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    rgb = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray  = (r+g+b)
    return gray

with picamera.PiCamera() as camera:
        camera.resolution = (100, 100)
        camera.framerate = 24
        time.sleep(2)
        output = np.empty((112 * 128 * 3,), dtype=np.uint8)
        camera.capture(output, 'rgb',resize=(100, 100))

output = output.reshape((112, 128, 3))
output = output[:100, :100, :]
output = output.reshape((100, 100, 3))
print(output.shape)
    #camera.capture(output, 'rgb')

X_input = output
X_input_gray = X_input.copy()

#X_input_gray = rgb2gray(X_input)
print(X_input_gray.shape)
X_input_gray = np.average(X_input_gray, axis=2)
print(X_input_gray.shape)

numberimages = 1
X_label = np.ones((numberimages,1),dtype=int)

X_test = X_input_gray

X_test = np.asarray(X_test, dtype=np.float)
X_test_f = X_test.reshape((10000,1))
X_test_f = np.transpose(X_test_f)
y_test = X_label#.values.ravel()
y_test = np.asarray(y_test, dtype=np.int32)
y_test = y_test.reshape((1,))
y_test = X_label.ravel()#.values.ravel()
print(X_test_f.shape, X_test_f.dtype)
print(y_test.shape, y_test.dtype)


def cnn_model_fn(features, labels, mode):

  # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 100, 100, 1])
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 25, 25, 1]
  # Output Tensor Shape: [batch_size, 25, 25, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[10, 10],
        padding="same",
        activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 25, 25, 32]
  # Output Tensor Shape: [batch_size, 12, 12, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=10)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 12, 12, 32]
  # Output Tensor Shape: [batch_size, 12, 12, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=92,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=3)


    conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=182,#92,
            kernel_size=[2, 2],
            padding="same",
            activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

      # Dense Layer
      # Flatten tensor into a batch of vectors
      # Input Tensor Shape: [batch_size, 6, 6, 64]
      # Output Tensor Shape: [batch_size, 6 * 6 * 64]
    pool3_flat = tf.reshape(pool3, [-1, 1 * 1 * 182])

    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

    if mode == tf.estimator.ModeKeys.PREDICT:
        predict_op = ()
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
         loss=loss,
         global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
       "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
       "auc": tf.metrics.auc(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          num_thresholds=200,
          metrics_collections=None,
          updates_collections=None,
          curve='ROC',
          name=None,
          summation_method='trapezoidal'
          ),
       "false_negatives": tf.metrics.false_negatives(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None
),
        "false_postives": tf.metrics.false_positives(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None
),
        "true_negatives":tf.metrics.true_negatives(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None
),
        "true_postivies":tf.metrics.true_positives(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None
)


}
    return tf.estimator.EstimatorSpec(
       mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

test_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="./model")

test_results=[len(y_test)]

x_example = X_test.shape
print(type(x_example))
totalentries = x_example[0]
print(totalentries)
print(type(totalentries))

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

def evaluate(X_test,y_test):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
    eval_results = test_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)
    test_results.append(eval_results)

predict_results=[len(y_test)]

x_example = X_test.shape
print(type(x_example))
totalentriesp = x_example[0]
print(totalentriesp)
print(type(totalentriesp))

def predict(X_test,y_test):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
    return list(test_classifier.predict(input_fn=predict_input_fn,yield_single_examples=True))

#works
with tf.Session() as sess:
    loader = tf.train.import_meta_graph('./model/model.ckpt-181000.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./model'))
    #evaluate(X_test,y_test)
    predict_results = predict(X_test_f,y_test)
    #predict_resultsl = list(predict_results)
    with open('results_withpredcnn2.csv', 'w') as f:
        for item in predict_results:#l:
            f.write("%s\n" % item)

print(type(test_results))
import pprint
print("Test results:")
pprint.pprint(test_results)
print("Predict results:")
pprint.pprint(predict_results)
