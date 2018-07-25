class NeuralNetwork:

    def __init__(self):

        ...

        # TensorFlow has a built-in loss-function for doing regression:
        # self.loss = tf.nn.l2_loss(self.q_values - self.q_values_new)
        # But it uses tf.reduce_sum() rather than tf.reduce_mean()
        # which is used by PrettyTensor. This means the scale of the
        # gradient is different and hence the hyper-parameters
        # would have to be re-tuned.
        squared_error = tf.square(self.q_values - self.q_values_new)
        sum_squared_error = tf.reduce_sum(squared_error, axis=1)
        self.loss = tf.reduce_mean(sum_squared_error)

        # Optimizer used for minimizing the loss-function.
        # Note the learning-rate is a placeholder variable so we can
        # lower the learning-rate as optimization progresses.
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Used for saving and loading checkpoints.
        self.saver = tf.train.Saver()

        # Create a new TensorFlow session so we can run the Neural Network.
        self.session = tf.Session()

        # Load the most recent checkpoint if it exists,
        # otherwise initialize all the variables in the TensorFlow graph.
        self.load_checkpoint()