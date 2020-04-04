import tensorflow as tf
EPSILON = 1e-8
class Model(object):

    def __init__(self,RESOLUTION, learning_rate = 0.01):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Placeholder for input batch of images. It will later be bound to actual image data.
            self.images=tf.placeholder(dtype=tf.float32, shape=[None, RESOLUTION, RESOLUTION, 1],name="images")
            self.labels=tf.placeholder(dtype=tf.int64, shape=[None],name="labels")
            
            self.conv1=tf.layers.conv2d(inputs=self.images, kernel_size=[3,3], strides=1, activation=tf.nn.relu, name="conv1", padding="SAME", filters=32)
            self.pool1=tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2,2], strides=1, padding='valid',  name="pool1")
            self.conv2=tf.layers.conv2d(inputs=self.pool1, kernel_size=[3,3], strides=1, activation=tf.nn.relu, name="conv2", padding="SAME", filters=64)
            self.pool2=tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2,2], strides=1, padding='valid',  name="pool2")
            self.conv3=tf.layers.conv2d(inputs=self.pool2, kernel_size=[3,3], strides=1, activation=tf.nn.relu, name="conv3", padding="SAME", filters=128)
            self.pool3=tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2,2], strides=1, padding='valid',  name="pool3")
            self.flat=tf.layers.Flatten()(self.pool3)
            self.fc1=tf.layers.dense(inputs=self.flat, units=128, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
            self.drop1=tf.layers.dropout(inputs=self.fc1, rate=0.2)
            self.fc2=tf.layers.dense(inputs=self.drop1, units=256, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
            self.drop2=tf.layers.dropout(inputs=self.fc2, rate=0.2)
            self.output=tf.layers.dense(inputs=self.drop2, units=8, activation=None)
    
            # Creates a subnetwork that computes class predictions and their probabilities.
            # Softmax layer to convert scores to probabilities.
            self.prob = tf.nn.softmax(self.output, name="prob")
            # "Top-k" layer to get three most probable guesses, and their probabilities.
            (self.guess_prob, self.guess_class) = tf.nn.top_k(self.prob, k=3, name="top_k")
            
            #self.one_hot_labels = tf.one_hot(self.labels,7)
            self.loss_per_sample=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output, name='loss')
            self.loss=tf.reduce_mean(self.loss_per_sample)+tf.losses.get_regularization_loss()
            self.optimizer=tf.train.AdamOptimizer(learning_rate, epsilon=EPSILON).minimize(self.loss)
            # We want to track accuracy through one epoch
            is_correct = tf.equal(tf.argmax(self.prob, axis=1), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            # add summary
            self._add_summary()
            
    def _add_summary(self):
        """
        Creates a subnetwork that produces a summary with the following data:
        - 0-th input image from current batch.
        - Weights of each filter of the `conv1` layer.
        - Outputs of the same filters for the 0-th input image from current batch.
        """

        # Index of the filter of interest in the `conv1` layer.
        example_index = 0

        # Get input image of interest.
        image = tf.slice(self.images, begin=[example_index, 0, 0, 0], size=[1, -1, -1, -1])

        # Get a handle on the weight and output tensors of interest.
        weights = self.graph.get_tensor_by_name("conv1/kernel:0") # gets conv1 weights
        outputs = self.graph.get_tensor_by_name("conv1/Conv2D:0")

        summary_list = [
            self._image_summary(image, name="image"),
            self._conv_weight_summary(weights, name="conv1_weights"),
            self._conv_output_summary(outputs, example_index, name="conv1_outputs")
            ]
        self.summary = tf.summary.merge([s for s in summary_list if s is not None])

    
    def _image_summary(self, tensor, name):
        """
        Creates a subnetwork that computes a summary for a given batch of images.
        Uses [tf.summary.image](https://www.tensorflow.org/api_docs/python/tf/summary/image).
        :param tensor: Tensor of shape (<batch_size>, <height>, <width>) representing a batch of images.
        :param name: Name of the returned tensor.
        :returns: Tensor containing a summary object which represents the batch of images.
        """
        with tf.variable_scope(name):
            # Normalize to [0, 1].
            x_min = tf.reduce_min(tensor)
            x_max = tf.reduce_max(tensor)
            normalized = (tensor - x_min) / (x_max - x_min)
            return tf.summary.image("out", normalized, max_outputs=normalized.shape[0])

    def _conv_weight_summary(self, conv_weight_tensor, name):
        """
        Create a subnetwork that visualizes given tensor containing weights of a convolutional layer.
        :param conv_weight_tensor: Tensor to visualize.
        :param name: Name of the returned tensor.
        :returns: Tensor containing a summary object which represents a set of images depicting
            layer weights.
        """
        # Reduce along the input channel axis.
        images = tf.norm(conv_weight_tensor, axis=2, keepdims=True)
        # Transpose into a batch of single-channel images to visualize.
        images = tf.transpose(images, [3, 0, 1, 2])
        # Visualize as batch of images.
        return self._image_summary(images, name=name)

    def _conv_output_summary(self, conv_output_tensor, example_index, name):
        """
        Create a subnetwork that visualizes given tensor containing outputs of a convolutional
        layer, restricted to a given example.
        :param conv_output_tensor: Tensor containing layer outputs for all examples.
        :param example_index: Index of the example whose outputs should be visualized.
        :param name: Name of the returned tensor.
        :returns: Tensor containing a summary object which represents a set of images depicting
            outputs of all filters on the example of interest.
        """
        # Slice along the batch axis to select the example.
        images = tf.slice(conv_output_tensor, [example_index, 0, 0, 0], [1, -1, -1, -1])
        # Transpose into a batch of single-channel images to visualize.
        images = tf.transpose(images, [3, 1, 2, 0])
        # Visualize as batch of images.
        return self._image_summary(images, name=name)