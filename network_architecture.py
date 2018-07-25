class NeuralNetwork:

    def __init__(self):

        # Initializer for the layers in the Neural Network.
        # If you change the architecture of the network, particularly
        # if you add or remove layers, then you may have to change
        # the stddev-parameter here. The initial weights must result
        # in the Neural Network outputting Q-values that are very close
        # to zero - but the network weights must not be too low either
        # because it will make it hard to train the network.

        init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)
        ...

        # This builds the Neural Network using the tf.layers API,
        # which is very verbose and inelegant, but should work for everyone.

        # Note that the checkpoints for Tutorial #16 which can be
        # downloaded from the internet only support PrettyTensor.
        # Although the Neural Networks appear to be identical when
        # built using the PrettyTensor and tf.layers APIs,
        # they actually create somewhat different TensorFlow graphs
        # where the variables have different names, which means the
        # checkpoints are incompatible for the two builder APIs.

        # Padding used for the convolutional layers.
        # "VALID" only ever drops the right-most columns (or bottom-most rows).
        # "SAME" tries to pad evenly left and right, but if the amount of columns
        #  to be added is odd, it will add the extra column to the right.
        padding = 'SAME'

        # Activation function for all convolutional and fully-connected
        # layers, except the last.
        activation = tf.nn.relu

        # Reference to the lastly added layer of the Neural Network.
        # This makes it easy to add or remove layers.
        net = self.x

        # First convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                               filters=16, kernel_size=3, strides=2,
                               padding=padding,
                               kernel_initializer=init, activation=activation)

        # Second convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                               filters=32, kernel_size=3, strides=2,
                               padding=padding,
                               kernel_initializer=init, activation=activation)

        # Third convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                               filters=64, kernel_size=3, strides=1,
                               padding=padding,
                               kernel_initializer=init, activation=activation)

        # Flatten output of the last convolutional layer so it can
        # be input to a fully-connected (aka. dense) layer. Flattens
        # the input while maintaining the batch_size.Assumes that the
        # first dimension represents the batch.

        net = tf.contrib.layers.flatten(net)

        # First fully-connected (aka. dense) layer.
        net = tf.layers.dense(inputs=net, name='layer_fc1', units=1024,
                              kernel_initializer=init, activation=activation)

        # Second fully-connected layer.
        net = tf.layers.dense(inputs=net, name='layer_fc2', units=512,
                              kernel_initializer=init, activation=activation)

        # Final fully-connected layer. Activation function (callable).
        net = tf.layers.dense(inputs=net, name='layer_fc_out', units=num_actions,
                              kernel_initializer=init, activation=None)

        # The output of the Neural Network is the estimated Q-values
        # for each possible action in the game-environment.
        self.q_values = net