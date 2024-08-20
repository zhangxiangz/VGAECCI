
from tensorflow.keras.layers import Layer

from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model

from .layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in list(kwargs.keys()):
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in list(kwargs.keys()):
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class VGAECCI(Model):
    """
    Referred to 
    """
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, hidden1_dim, hidden2_dim, hidden3_dim, **kwargs):
        super(VGAECCI, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.h1_dim = hidden1_dim
        self.h2_dim = hidden2_dim
        self.h3_dim = hidden3_dim
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        with tf.compat.v1.variable_scope('Encoder'):
            # First Graph Convolutional Layer
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=self.h1_dim,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)
            self.h1 = self.hidden1
            # Second Graph Convolutional Layer
            self.hidden2 = GraphConvolution(input_dim=self.h1_dim,
                                            output_dim=self.h2_dim,
                                            adj=self.adj,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            name='e_dense_2')(self.hidden1)
            
            # Third Graph Convolutional Layer
            self.hidden3 = GraphConvolution(input_dim=self.h2_dim,
                                            output_dim=self.h3_dim,
                                            adj=self.adj,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            name='e_dense_3')(self.hidden2)

            # Calculate mean and log standard deviation for the latent space
            self.z_mean = GraphConvolution(input_dim=self.h3_dim,
                                           output_dim=self.h2_dim,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_4')(self.hidden3)
            
            self.z_log_std = GraphConvolution(input_dim=self.h3_dim,
                                              output_dim=self.h2_dim,
                                              adj=self.adj,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging,
                                              name='e_dense_5')(self.hidden3)
            
            # Reparameterization trick
            self.z = self.z_mean + tf.random.normal([self.n_samples, self.h2_dim]) * tf.exp(self.z_log_std)
           
            # Reconstruction with Inner Product Decoder
            self.reconstructions = InnerProductDecoder(input_dim=self.h2_dim,
                                                        act=lambda x: x,
                                                        logging=self.logging)(self.z)
            # Set embeddings
            self.embeddings = self.z



def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.compat.v1.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.compat.v1.set_random_seed(1)
        weights = tf.compat.v1.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.compat.v1.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.compat.v1.get_variable("bias", shape=[n2], initializer=tf.compat.v1.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out



import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class Discriminator(Model):
    def __init__(self, input_dim, dc_hidden1_dim, dc_hidden2_dim, dc_hidden3_dim, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.act = tf.nn.relu
        self.input_dim = input_dim
        self.dc_h1_dim = dc_hidden1_dim
        self.dc_h2_dim = dc_hidden2_dim
        self.dc_h3_dim = dc_hidden3_dim

    def call(self, inputs, training=None):
        # Define layers
        dense1 = Dense(self.dc_h1_dim, activation=self.act, name='dc_den1')
        dense2 = Dense(self.dc_h2_dim, activation=self.act, name='dc_den2')
        dense3 = Dense(self.dc_h3_dim, activation=self.act, name='dc_den3')
        output_layer = Dense(1, name='dc_output')

        # Forward pass
        x = dense1(inputs)
        x = dense2(x)
        x = dense3(x)
        output = output_layer(x)
        
        return output

    def construct(self, inputs, reuse=False):
        # Use variable scope for reuse
        with tf.compat.v1.variable_scope('Discriminator', reuse=reuse):
            output = self.call(inputs)
        
        return output












