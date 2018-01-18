import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import pad_to_bounding_box, crop_to_bounding_box
from tensorflow.python.ops.losses.losses_impl import Reduction
from tensorflow.python.ops.nn_ops import leaky_relu

from neural_network_architectures import EncoderStandard, DecoderStandard

class AutoencoderFaceTransferGraph:
    def __init__(self, input_A, input_B, dropout_rate, learning_rate, batch_size=128, is_training=True, augment=True, num_gpus=1,
                 target_A=None, target_B=None):
        """
        Initializes a DAGAN object.
        :param input_x_i: Input image x_i
        :param input_x_j: Input image x_j
        :param dropout_rate: A dropout rate placeholder or a scalar to use throughout the network
        :param generator_layer_sizes: A list with the number of feature maps per layer (generator) e.g. [64, 64, 64, 64]
        :param discriminator_layer_sizes: A list with the number of feature maps per layer (discriminator)
                                                                                                   e.g. [64, 64, 64, 64]
        :param generator_layer_padding: A list with the type of padding per layer (e.g. ["SAME", "SAME", "SAME","SAME"]
        :param z_inputs: A placeholder for the random noise injection vector z (usually gaussian or uniform distribut.)
        :param batch_size: An integer indicating the batch size for the experiment.
        :param z_dim: An integer indicating the dimensionality of the random noise vector (usually 100-dim).
        :param num_channels: Number of image channels
        :param is_training: A boolean placeholder for the training/not training flag
        :param augment: A boolean placeholder that determines whether to augment the data using rotations
        :param discr_inner_conv: Number of inner layers per multi layer in the discriminator
        :param gen_inner_conv: Number of inner layers per multi layer in the generator
        :param num_gpus: Number of GPUs to use for training
        """
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self.encoder = EncoderStandard(name="encoder", batch_size=self.batch_size, activation=leaky_relu,
                                               total_stages=4, num_inner_conv=1,
                                               num_filter_list=[128/2, 256/2, 512/4, 1024/4], batch_normalization=True)

        self.decoder_A = DecoderStandard(name="decoder_A", batch_size=self.batch_size, activation=leaky_relu,
                                       total_stages=4, num_inner_conv=1,
                                       num_filter_list=[1024/4, 512/4, 256/2], batch_normalization=True)

        self.decoder_B = DecoderStandard(name="decoder_B", batch_size=self.batch_size, activation=leaky_relu,
                                         total_stages=4, num_inner_conv=1,
                                         num_filter_list=[1024/4, 512/4, 256/2], batch_normalization=True)
        self.input_A = input_A
        self.input_B = input_B

        self.target_A = target_A
        self.target_B = target_B

        self.dropout_rate = dropout_rate
        self.training_phase = is_training
        self.augment = augment
        self.learning_rate = learning_rate

    def rotate_data(self, image_a, image_b):
        """
        Rotate 2 images by the same number of degrees
        :param image_a: An image a to rotate k degrees
        :param image_b: An image b to rotate k degrees
        :return: Two images rotated by the same amount of degrees
        """
        random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None))
        image_a = tf.image.rot90(image_a, k=random_variable[0])
        image_b = tf.image.rot90(image_b, k=random_variable[0])
        return [image_a, image_b]

    def rotate_batch(self, batch_images_a, batch_images_b):
        """
        Rotate two batches such that every element from set a with the same index as an element from set b are rotated
        by an equal amount of degrees
        :param batch_images_a: A batch of images to be rotated
        :param batch_images_b: A batch of images to be rotated
        :return: A batch of images that are rotated by an element-wise equal amount of k degrees
        """
        shapes = map(int, list(batch_images_a.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked_a = tf.unstack(batch_images_a)
            batch_images_unpacked_b = tf.unstack(batch_images_b)
            new_images_a = []
            new_images_b = []
            for image_a, image_b in zip(batch_images_unpacked_a, batch_images_unpacked_b):
                rotate_a, rotate_b = self.augment_rotate(image_a, image_b)
                new_images_a.append(rotate_a)
                new_images_b.append(rotate_b)

            new_images_a = tf.stack(new_images_a)
            new_images_a = tf.reshape(new_images_a, (batch_size, x, y, c))
            new_images_b = tf.stack(new_images_b)
            new_images_b = tf.reshape(new_images_b, (batch_size, x, y, c))
            return [new_images_a, new_images_b]

    def augment_rotate(self, image_a, image_b):
        r = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None))
        rotate_boolean = tf.equal(0, r, name="check-rotate-boolean")
        [image_a, image_b] = tf.cond(rotate_boolean[0], lambda: self.rotate_data(image_a, image_b),
                        lambda: [image_a, image_b])
        return image_a, image_b

    def data_augment_batch(self, batch_images_a, batch_images_b): #maybe pass some angle feature and expect a different output
        """
        Apply data augmentation to a set of image batches if self.augment is set to true
        :param batch_images_a: A batch of images to augment
        :param batch_images_b: A batch of images to augment
        :return: A list of two augmented image batches
        """
        [images_a, images_b] = tf.cond(self.augment, lambda: self.rotate_batch(batch_images_a, batch_images_b),
                                       lambda: [batch_images_a, batch_images_b])
        return images_a, images_b

    def save_features(self, name, features):
        """
        Save feature activations from a network
        :param name: A name for the summary of the features
        :param features: The features to save
        """
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = 8
            x_channels = channels / y_channels

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                        y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            tf.summary.image('{}_{}'.format(name, i), activations_features)

    def huber_loss(self, logits, labels, max_gradient=1.0):
        err = tf.abs(labels - logits)
        mg = tf.constant(max_gradient)
        lin = mg * (err - 0.5 * mg)
        quad = 0.5 * err * err
        loss = tf.reduce_mean(tf.where(err < mg, quad, lin))
        #loss = tf.reduce_sum(err)
        return loss

    def mean_absolute_error(self, logits, labels):

        loss = tf.reduce_mean(tf.abs(logits-labels))

        return loss

    def euclidean_distance(self, vector_a, vector_b):
        diff_a_b = vector_a - vector_b
        square_diff_a_b = tf.square(diff_a_b)
        sum_square_diff_a_b = tf.reduce_sum(square_diff_a_b)

        return sum_square_diff_a_b

    def selected_loss(self, logits, labels):
        return self.huber_loss(logits=logits, labels=labels)

    # def add_noise_batch(self, batch_images):
    #     shapes = map(int, list(batch_images.get_shape()))
    #     batch_size, x, y, c = shapes
    #     with tf.name_scope('augment'):
    #         batch_images_unpacked = tf.unstack(batch_images)
    #         new_images = []
    #         for image in batch_images_unpacked:
    #             new_images.append(self.add_noise_data(image))
    #         new_images = tf.stack(new_images)
    #         new_images = tf.reshape(new_images, (batch_size, x, y, c))
    #         return new_images
    #
    # def add_noise_image(self, image):
    #     random_noise = tf.random_normal(shape=image.get_shape(), stddev=0.1)
    #     image = image + random_noise
    #     return image
    #
    # def add_noise_data(self, image):
    #     random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None,
    #                                                    name="check-noise-boolean"))
    #     add_noise_boolean = tf.equal(0, random_variable[0], name="check-add-noise-boolean")
    #     image = tf.cond(add_noise_boolean, lambda: self.add_noise_image(image), lambda: image)
    #     return image
    #
    # def shift_image_random(self, image):
    #     random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None,
    #                                                    name="shift_image_random"))
    #     shift_image_flag = tf.equal(0, random_variable[0], name="check-shift-image-boolean")
    #     image = tf.cond(shift_image_flag, lambda: self.shift_image(image), lambda: image)
    #     return image
    #
    # def shift_image_batch(self, batch_images):
    #     shapes = map(int, list(batch_images.get_shape()))
    #     print("shapes", shapes)
    #     batch_size, x, y, c = shapes
    #     with tf.name_scope('augment'):
    #         batch_images_unpacked = tf.unstack(batch_images)
    #         new_images = []
    #         for image in batch_images_unpacked:
    #             new_images.append(self.shift_image_random(image))
    #         new_images = tf.stack(new_images)
    #         new_images = tf.reshape(new_images, (batch_size, x, y, c))
    #         return new_images
    #
    # def shift_image(self, image):
    #     shapes = map(int, list(image.get_shape()))
    #     x, y, c = shapes
    #     image = image - 1.0
    #     k_h = tf.unstack(tf.random_uniform([1], minval=0, maxval=7, dtype=tf.int32, seed=None, name=None))
    #     k_w = tf.unstack(tf.random_uniform([1], minval=0, maxval=7, dtype=tf.int32, seed=None, name=None))
    #     k_hdiff = tf.cast(x-k_h[0], dtype=tf.int32)
    #     k_wdiff = tf.cast(y-k_w[0], dtype=tf.int32)
    #     image = crop_to_bounding_box(
    #         image,
    #         k_h[0],
    #         k_w[0],
    #         k_hdiff,
    #         k_wdiff
    #     )
    #     image = pad_to_bounding_box(
    #             image,
    #             0,
    #             0,
    #             x,
    #             y
    #             )
    #     image = image + 1
    #     return image
    #
    # def data_shift_batch(self, batch_images):
    #     images = tf.cond(self.augment_shift, lambda: self.shift_image_batch(batch_images), lambda: batch_images)
    #     return images
    #
    # def data_noise_batch(self, batch_images):
    #     images = tf.cond(self.augment_noise, lambda: self.add_noise_batch(batch_images), lambda: batch_images)
    #     return images

    # def data_augment_batch(self, batch_images):
    #     batch_images = self.data_rotate_batch(batch_images)
    #     batch_images = self.data_noise_batch(batch_images)
    #     batch_images = self.shift_image_batch(batch_images)
    #     return batch_images

    def loss(self, gpu_id):

        """
        Builds models, calculates losses, saves tensorboard information.
        :param gpu_id: The GPU ID to calculate losses for.
        :return: Returns the generator and discriminator losses.
        """
        with tf.name_scope("losses_{}".format(gpu_id)):

            input_A, target_A = self.data_augment_batch(self.input_A[gpu_id], self.target_A[gpu_id])
            input_B, target_B = self.data_augment_batch(self.input_B[gpu_id], self.target_B[gpu_id])

            enc_A, encoder_features_A = self.encoder(inputs=input_A, training=self.training_phase,
                                                     dropout_rate=self.dropout_rate)
            enc_B, encoder_features_B = self.encoder(inputs=input_B, training=self.training_phase,
                                                     dropout_rate=self.dropout_rate)

            dec_A_to_A, _ = self.decoder_A(inputs=enc_A, encoder_layers=encoder_features_A, training=self.training_phase,
                                        dropout_rate=self.dropout_rate)
            dec_B_to_B, _ = self.decoder_B(inputs=enc_B, encoder_layers=encoder_features_B, training=self.training_phase,
                                        dropout_rate=self.dropout_rate)
            dec_B_to_A, _ = self.decoder_A(inputs=enc_B, encoder_layers=encoder_features_B, training=self.training_phase,
                                        dropout_rate=self.dropout_rate)
            dec_A_to_B, _ = self.decoder_B(inputs=enc_A, encoder_layers=encoder_features_A, training=self.training_phase,
                                        dropout_rate=self.dropout_rate)

            autoencoder_loss_A_A = self.selected_loss(logits=dec_A_to_A, labels=target_A)
            autoencoder_loss_B_B = self.selected_loss(logits=dec_B_to_B, labels=target_B)

            tf.add_to_collection('autoencoder_loss_A_A', autoencoder_loss_A_A)
            tf.add_to_collection('autoencoder_loss_B_B', autoencoder_loss_B_B)

            tf.summary.scalar('autoencoder_loss_A_A', tf.reduce_mean(autoencoder_loss_A_A))
            tf.summary.scalar('autoencoder_loss_B_B', tf.reduce_mean(autoencoder_loss_B_B))

            tf.summary.scalar('learning rate', self.learning_rate)

            tf.summary.image('input_A', [tf.concat(tf.unstack(input_A, axis=0), axis=0)])
            tf.summary.image('input_B', [tf.concat(tf.unstack(input_B, axis=0), axis=0)])
            tf.summary.image('target_A', [tf.concat(tf.unstack(target_A, axis=0), axis=0)])
            tf.summary.image('target_B', [tf.concat(tf.unstack(target_B, axis=0), axis=0)])

            tf.summary.image('dec_A_to_A', [tf.concat(tf.unstack(dec_A_to_A, axis=0), axis=0)])
            tf.summary.image('dec_B_to_B', [tf.concat(tf.unstack(dec_B_to_B, axis=0), axis=0)])
            tf.summary.image('dec_B_to_A', [tf.concat(tf.unstack(dec_B_to_A, axis=0), axis=0)])
            tf.summary.image('dec_A_to_B', [tf.concat(tf.unstack(dec_A_to_B, axis=0), axis=0)])

        return {
            "autoencoder_loss_A_A": tf.add_n(tf.get_collection('autoencoder_loss_A_A'),
                                             name='total_autoencoder_loss_A_A'),
            "autoencoder_loss_B_B": tf.add_n(tf.get_collection('autoencoder_loss_B_B'),
                                             name='total_encoder_loss_B_B')
        }

    def train(self, opts, losses):
        """
        Returns ops for training our DAGAN system.
        :param opts: A dict with optimizers.
        :param losses: A dict with losses.
        :return: A dict with training ops for the dicriminator and the generator.
        """
        opt_ops = dict()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_ops["autoencoder_A_A_opt_op"] = opts["autoencoder_A_A_opt"].minimize(losses["autoencoder_loss_A_A"],
                                                                             var_list=self.encoder.variables+
                                                                                      self.decoder_A.variables,
                                                                             colocate_gradients_with_ops=True)

            opt_ops["autoencoder_B_B_opt_op"] = opts["autoencoder_B_B_opt"].minimize(losses["autoencoder_loss_B_B"],
                                                                             var_list=self.encoder.variables+
                                                                                      self.decoder_B.variables,
                                                                             colocate_gradients_with_ops=True)

        return opt_ops

    def crossentropy_softmax(self, outputs, targets):
        normOutputs = outputs - tf.reduce_max(outputs, axis=-1)[:, None]
        logProb = normOutputs - tf.log(tf.reduce_sum(tf.exp(normOutputs), axis=-1)[:, None])
        return -tf.reduce_mean(tf.reduce_sum(targets * logProb, axis=1))

    def init_train(self, beta1=0.5, beta2=0.99):
        """
        Initialize training by constructing the summary, loss and ops
        :param learning_rate: The learning rate for the Adam optimizer
        :param beta1: Beta1 for the Adam optimizer
        :param beta2: Beta2 for the Adam optimizer
        :return: summary op, losses and training ops.
        """
        autoencoder_loss_A_A = []
        autoencoder_loss_B_B = []

        autoencoder_A_A_opt = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2, learning_rate=self.learning_rate)
        autoencoder_B_B_opt = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2, learning_rate=self.learning_rate)

        opts = {"autoencoder_A_A_opt": autoencoder_A_A_opt, "autoencoder_B_B_opt": autoencoder_B_B_opt}

        if self.num_gpus > 0:
            device_ids = ['/gpu:{}'.format(i) for i in range(self.num_gpus)]
        else:
            device_ids = ['/cpu:0']
        for i, device_id in enumerate(device_ids):
            with tf.device(device_id):
                total_losses = self.loss(gpu_id=i)
                autoencoder_loss_A_A.append(total_losses['autoencoder_loss_A_A'])
                autoencoder_loss_B_B.append(total_losses["autoencoder_loss_B_B"])

        autoencoder_loss_A_A = tf.reduce_mean(autoencoder_loss_A_A, axis=0)
        autoencoder_loss_B_B = tf.reduce_mean(autoencoder_loss_B_B, axis=0)

        losses = {'autoencoder_loss_A_A': autoencoder_loss_A_A,
                  'autoencoder_loss_B_B': autoencoder_loss_B_B}

        summary = tf.summary.merge_all()
        apply_grads_ops = self.train(opts=opts, losses=losses)

        return summary, losses, apply_grads_ops

    def sample_images(self):

        enc_A, encoder_features_A = self.encoder(inputs=self.input_A[0], training=self.training_phase,
                                                 dropout_rate=self.dropout_rate)
        enc_B, encoder_features_B = self.encoder(inputs=self.input_B[0], training=self.training_phase,
                                                 dropout_rate=self.dropout_rate)

        dec_A_to_A, _ = self.decoder_A(inputs=enc_A, encoder_layers=encoder_features_A, training=self.training_phase,
                                    dropout_rate=self.dropout_rate)
        dec_B_to_B, _ = self.decoder_B(inputs=enc_B, encoder_layers=encoder_features_B, training=self.training_phase,
                                    dropout_rate=self.dropout_rate)
        dec_B_to_A, _ = self.decoder_A(inputs=enc_B, encoder_layers=encoder_features_B, training=self.training_phase,
                                    dropout_rate=self.dropout_rate)
        dec_A_to_B, _ = self.decoder_B(inputs=enc_A, encoder_layers=encoder_features_A, training=self.training_phase,
                                    dropout_rate=self.dropout_rate)

        return dec_A_to_A, dec_B_to_B, dec_B_to_A, dec_A_to_B

    def sample_images_A_to_B(self):

        enc_A, encoder_features_A = self.encoder(inputs=self.input_A[0], training=self.training_phase,
                                                 dropout_rate=self.dropout_rate)

        dec_A_to_B, _ = self.decoder_B(inputs=enc_A, encoder_layers=encoder_features_A, training=self.training_phase,
                                    dropout_rate=self.dropout_rate)

        return dec_A_to_B

    def sample_images_B_to_A(self):

        enc_B, encoder_features_B = self.encoder(inputs=self.input_B[0], training=self.training_phase,
                                                 dropout_rate=self.dropout_rate)

        dec_B_to_A, _ = self.decoder_A(inputs=enc_B, encoder_layers=encoder_features_B, training=self.training_phase,
                                    dropout_rate=self.dropout_rate)

        return dec_B_to_A


