import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.nn_ops import leaky_relu

from utils.network_summary import count_parameters


def remove_duplicates(input_features):
    """
    Remove duplicate entries from layer list.
    :param input_features: A list of layers
    :return: Returns a list of unique feature tensors (i.e. no duplication).
    """
    feature_name_set = set()
    non_duplicate_feature_set = []
    for feature in input_features:
        if feature.name not in feature_name_set:
            non_duplicate_feature_set.append(feature)
        feature_name_set.add(feature.name)
    return non_duplicate_feature_set

class EncoderDecoderTemplate(object):
    def __init__(self, name, batch_size, activation, total_stages, num_inner_conv, num_filter_list,
                 batch_normalization):

        self.activation = leaky_relu
        self.total_stages = total_stages
        self.num_inner_conv = num_inner_conv
        self.num_filter_list = num_filter_list
        self.conv_layer_num = 0
        self.batch_size = batch_size
        self.build = True
        self.reuse = False
        self.batch_normalization = batch_normalization
        self.name = name

    def conv_layer(self, inputs, num_filters, strides, dropout_rate, training, deconv=False, w_size=2, h_size=2):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :param w_size: Used only for upscale, w_size to scale to.
        :param h_size: Used only for upscale, h_size to scale to.
        :return: Convolution features
        """
        self.conv_layer_num += 1

        if deconv:
            outputs = tf.layers.conv2d(inputs, num_filters, kernel_size=(3, 3),
                                       strides=strides,
                                       padding="SAME", activation=None)
            outputs = self.activation(outputs, alpha=0.1)
            outputs = tf.layers.dropout(inputs=outputs, rate=dropout_rate, training=training)
            if self.batch_normalization:
                outputs = batch_norm(outputs,
                                     decay=0.99, scale=True,
                                     center=True, is_training=training,
                                     renorm=True)
            outputs = self.upscale(outputs, h_size=h_size, w_size=w_size)
        else:
            outputs = tf.layers.conv2d(inputs, num_filters, kernel_size=(5, 5), strides=strides,
                                                 padding="SAME", activation=None)
            outputs = self.activation(outputs, alpha=0.1)
            outputs = tf.layers.dropout(inputs=outputs, rate=dropout_rate, training=training)
            if self.batch_normalization:
                outputs = batch_norm(outputs,
                                     decay=0.99, scale=True,
                                     center=True, is_training=training,
                                     renorm=True)

        return outputs

    def subpixel_shuffling(self, x, rh, rw):

        batch_size, h, w, c = x.get_shape().as_list()
        if batch_size is None:
            batch_size = -1

        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)

        out = tf.reshape(x, (batch_size, h, w, rh, rw, oc))
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        out = tf.reshape(out, (batch_size, oh, ow, oc))

        return out

    def upscale(self, x, h_size, w_size):
        """
        Upscales an image using nearest neighbour
        :param x: Input image
        :param h_size: Image height size
        :param w_size: Image width size
        :return: Upscaled image
        """
        #return tf.image.resize_nearest_neighbor(x, (h_size, w_size))
        return self.subpixel_shuffling(x, h_size, w_size)

class EncoderStandard(EncoderDecoderTemplate):
    def __init__(self, name, batch_size, activation, total_stages,
                 num_inner_conv, num_filter_list,
                 batch_normalization):
        super(EncoderStandard, self).__init__(name, batch_size, activation, total_stages,
                                        num_inner_conv, num_filter_list,
                                        batch_normalization)
    def __call__(self, inputs, training=False, dropout_rate=0.0):
        with tf.variable_scope(self.name, reuse=self.reuse):
            encoder_layers = []
            outputs = inputs
            encoder_layers.append(outputs)
            for idx_enc in range(len(self.num_filter_list)):
                for idx_inner_dec in range(self.num_inner_conv):
                    outputs = self.conv_layer(outputs, num_filters=self.num_filter_list[idx_enc], strides=1,
                                              dropout_rate=dropout_rate, training=training)

                outputs = self.conv_layer(inputs=outputs, num_filters=self.num_filter_list[idx_enc],
                                          strides=2, deconv=False, dropout_rate=dropout_rate, training=training)
            encoder_layers.append(outputs)
            shape_before_dense = outputs.get_shape()
            outputs = tf.layers.flatten(outputs)
            outputs = tf.layers.dense(outputs, 1024)
            outputs = tf.layers.dense(outputs, 4*4*1024)
            outputs = tf.reshape(outputs, shape=(shape_before_dense[0], 4, 4, 1024))

            outputs = self.conv_layer(outputs, num_filters=512, strides=1,
                                      deconv=True, dropout_rate=dropout_rate, training=training)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        # view_names_of_variables(self.variables)
        if self.build:
            print("Encoder layer number", self.conv_layer_num, "bottleneck_size", encoder_layers[-1].get_shape())
            count_parameters(network_variables=self.variables, name=self.name)

        self.build = False

        return outputs, encoder_layers

class DecoderStandard(EncoderDecoderTemplate):
    def __init__(self, name, batch_size, activation, total_stages, num_inner_conv, num_filter_list,
                 batch_normalization):
        super(DecoderStandard, self).__init__(name, batch_size, activation, total_stages, num_inner_conv, num_filter_list,
                 batch_normalization)
        self.activation = activation
        self.total_stages = total_stages
        self.num_inner_conv = num_inner_conv
        self.num_filter_list = num_filter_list
        self.conv_layer_num = 0
        self.batch_size = batch_size
        self.build = True
        self.reuse = False
        self.batch_normalization = batch_normalization
        self.name = name

    def __call__(self, inputs, encoder_layers, training=False, dropout_rate=0.0):
        num_channels = encoder_layers[0].get_shape().as_list()[-1]

        with tf.variable_scope(self.name, reuse=self.reuse):
            outputs = inputs
            decoder_features = []
            for idx_dec in range(len(self.num_filter_list)):
                for idx_inner_dec in range(self.num_inner_conv):
                    outputs = self.conv_layer(outputs, num_filters=self.num_filter_list[idx_dec], strides=1,
                                              deconv=False, w_size=2, h_size=2, dropout_rate=dropout_rate,
                                              training=training)

                outputs = self.conv_layer(outputs, num_filters=self.num_filter_list[idx_dec], strides=1,
                                          deconv=True, w_size=2, h_size=2, dropout_rate=dropout_rate,
                                          training=training)

                decoder_features.append(outputs)

            outputs = tf.layers.conv2d(outputs, 3, kernel_size=(5, 5), strides=1,
                                       padding="SAME", activation=None)
            outputs = tf.nn.tanh(outputs)

            print(outputs.get_shape())

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        # view_names_of_variables(self.variables)
        if self.build:
            print("Decoder layer num", self.conv_layer_num)
            count_parameters(network_variables=self.variables, name=self.name)

        self.build = False

        return outputs, decoder_features


