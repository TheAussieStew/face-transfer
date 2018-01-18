import tensorflow as tf
import tqdm
from tensorflow.contrib import slim
import numpy as np
from utils import storage as store_util
from utils.sampling import sample_generator
from autoencoder_system_graph_builder import AutoencoderFaceTransferGraph
from utils.storage import save_statistics


def run_experiment(data, batch_size, num_gpus, continue_from_epoch, experiment_name, dropout_rate_value):
    tf.reset_default_graph()
    h, w, c = data.resize[0], data.resize[1], data.x_train_A.shape[-1]

    input_A = tf.placeholder(tf.float32, [num_gpus, batch_size, h, w, c], 'inputs_A')
    input_B = tf.placeholder(tf.float32, [num_gpus, batch_size, h, w, c], 'inputs_B')
    target_A = tf.placeholder(tf.float32, [num_gpus, batch_size, h, w, c], 'target_A')
    target_B = tf.placeholder(tf.float32, [num_gpus, batch_size, h, w, c], 'target_B')
    learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

    saved_models_filepath, logs_path, samples_filepath = store_util.build_experiment_folder(experiment_name)
    training_phase = tf.placeholder(tf.bool, name='training-flag')
    dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')
    augment_data = tf.placeholder(tf.bool, name='augment-flag')

    autoencoder_face_transfer_graph = AutoencoderFaceTransferGraph(input_A=input_A, input_B=input_B,
                                                                   target_A=target_A, target_B=target_B,
                                                                   dropout_rate=dropout_rate, batch_size=batch_size,
                                                                   is_training=training_phase, augment=augment_data,
                                                                   num_gpus=num_gpus, learning_rate=learning_rate)

    summary, losses, ops = autoencoder_face_transfer_graph.init_train()

    save_statistics(logs_path, ["epoch",  "total_train_loss_A_A",
                                                "total_train_loss_B_B",
                                                "total_val_loss_A_A",
                                                "total_val_loss_B_B"], create=True)

    sample_images = autoencoder_face_transfer_graph.sample_images()

    total_epochs = 200
    total_train_batches = int((data.x_train_A.shape[0] + data.x_train_B.shape[0]) / batch_size / num_gpus) * 50
    total_val_batches = int((data.x_val_A.shape[0]+data.x_val_B.shape[0]) / batch_size / num_gpus) * 50
    if total_val_batches == 0:
        total_val_batches = 50

    print(total_val_batches)
    print((data.x_val_A.shape[0]+data.x_val_B.shape[0]))
    init = tf.global_variables_initializer()

    autoencoder_iter = 1
    discriminator_iter = 1
    current_learning_rate = 0.001

    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True)) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=int(total_epochs/10))

        if continue_from_epoch != -1:
            if continue_from_epoch != -1:  # load checkpoint if needed
                checkpoint = "saved_models/{}_{}.ckpt".format(experiment_name, continue_from_epoch)
                variables_to_restore = []
                for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    print(var)
                    variables_to_restore.append(var)

                tf.logging.info('Fine-tuning from %s' % checkpoint)

                fine_tune = slim.assign_from_checkpoint_fn(
                    checkpoint,
                    variables_to_restore,
                    ignore_missing_vars=True)
                fine_tune(sess)


        x_train_A_stationary, _, x_train_B_stationary, _ = data.get_train_batch()
        x_val_A_stationary, _,  x_val_B_stationary, a = data.get_val_batch()
        best_val_loss = np.inf
        best_train_loss = np.inf
        with tqdm.tqdm(total=total_epochs) as pbar_e:
            for e in range(continue_from_epoch, total_epochs):

                total_train_autoencoder_loss_A_A = 0.
                total_train_autoencoder_loss_B_B = 0.


                total_val_autoencoder_loss_A_A = 0.
                total_val_autoencoder_loss_B_B = 0.


                sample_generator(phase="train", sess=sess, sample_images=sample_images, input_A=input_A, input_B=input_B,
                                 dropout_rate=dropout_rate, dropout_rate_value=dropout_rate_value,
                                 training_phase=training_phase, data=data, samples_filepath=samples_filepath,
                                 logs_path=logs_path, experiment_name=experiment_name, e=e,
                                 x_input_A=x_train_A_stationary, augment_data=augment_data,
                                 x_input_B=x_train_B_stationary, save_to_gdrive=True)

                sample_generator(phase="val", sess=sess, sample_images=sample_images, input_A=input_A, input_B=input_B,
                                 dropout_rate=dropout_rate, dropout_rate_value=dropout_rate_value,
                                 training_phase=training_phase, data=data, samples_filepath=samples_filepath,
                                 logs_path=logs_path, experiment_name=experiment_name, e=e,
                                 x_input_A=x_val_A_stationary, augment_data=augment_data,
                                 x_input_B=x_val_B_stationary, save_to_gdrive=True)

                with tqdm.tqdm(total=total_train_batches) as pbar:
                    for i in range(total_train_batches):

                        for n_g in range(autoencoder_iter):
                            x_train_A_input, x_train_A_target, x_train_B_input, x_train_B_target = \
                                data.get_train_batch(augment=True)

                            _, autoencoder_loss_A_A, _, autoencoder_loss_B_B = sess.run(
                                [ops["autoencoder_A_A_opt_op"], losses["autoencoder_loss_A_A"],
                                 ops["autoencoder_B_B_opt_op"], losses["autoencoder_loss_B_B"]],

                                feed_dict={dropout_rate: dropout_rate_value, input_A: x_train_A_input,
                                           target_A: x_train_A_target, input_B: x_train_B_input,
                                           target_B: x_train_B_target,  training_phase: True,
                                           augment_data: False,
                                           learning_rate: current_learning_rate})

                            total_train_autoencoder_loss_A_A += autoencoder_loss_A_A
                            total_train_autoencoder_loss_B_B += autoencoder_loss_B_B

                            if i%10==0:
                                _summary = sess.run(summary, feed_dict={dropout_rate: dropout_rate_value,
                                                                        input_A: x_train_A_input,
                                                                        target_A: x_train_A_target,
                                                                        input_B: x_train_B_input,
                                                                        target_B: x_train_B_target,
                                                                        training_phase: True,
                                                                        augment_data: True,
                                                                        learning_rate: current_learning_rate})
                                writer.add_summary(_summary)

                        iter_out = "autoencoder_loss_A_A_{}_autoencoder_loss_B_B_{}z" \
                            .format(total_train_autoencoder_loss_A_A / ((i + 1) * autoencoder_iter),
                                    total_train_autoencoder_loss_B_B / ((i + 1) * autoencoder_iter))

                        pbar.set_description(iter_out)
                        pbar.update(1)

                    save_path = saver.save(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, experiment_name, e))

                    total_train_autoencoder_loss_A_A /= (total_train_batches * autoencoder_iter)
                    total_train_autoencoder_loss_B_B /= (total_train_batches * autoencoder_iter)

                    print("total_train_autoencoder_loss_A_A_{}_total_train_autoencoder_loss_B_B_{}"
                          .format(total_train_autoencoder_loss_A_A,
                                  total_train_autoencoder_loss_B_B))

                    print("Model saved in", save_path)

                    total_val_autoencoder_loss_A_A /= (total_val_batches * autoencoder_iter)
                    total_val_autoencoder_loss_B_B /= (total_val_batches * autoencoder_iter)

                    with tqdm.tqdm(total=total_val_batches) as pbar:
                        for i in range(total_val_batches):

                            for n_g in range(autoencoder_iter):
                                x_val_A_input, x_val_A_target, x_val_B_input, x_val_B_target = data.get_val_batch()

                                autoencoder_loss_A_A, autoencoder_loss_B_B = sess.run(
                                    [losses["autoencoder_loss_A_A"],
                                     losses["autoencoder_loss_B_B"]],
                                    feed_dict={dropout_rate: dropout_rate_value, input_A: x_val_A_input,
                                           target_A: x_val_A_target, input_B: x_val_B_input,
                                           target_B: x_val_B_target, training_phase: False,
                                           augment_data: False, learning_rate: current_learning_rate})

                                total_val_autoencoder_loss_A_A += autoencoder_loss_A_A
                                total_val_autoencoder_loss_B_B += autoencoder_loss_B_B

                            iter_out = "autoencoder_loss_A_A_{}_autoencoder_loss_B_B_{}" \
                                .format(total_val_autoencoder_loss_A_A / ((i + 1) * autoencoder_iter),
                                        total_val_autoencoder_loss_B_B / ((i + 1) * autoencoder_iter))

                            pbar.set_description(iter_out)
                            pbar.update(1)

                        total_val_autoencoder_loss_A_A /= (total_val_batches * autoencoder_iter)
                        total_val_autoencoder_loss_B_B /= (total_val_batches * autoencoder_iter)


                    print("total_val_autoencoder_loss_A_A_{}_total_val_autoencoder_loss_B_B_{}"
                                                         .format(total_val_autoencoder_loss_A_A,
                                                                 total_val_autoencoder_loss_B_B))

                    summary_filepath = save_statistics(logs_path,
                                    [e, total_train_autoencoder_loss_A_A,
                                        total_train_autoencoder_loss_B_B,
                                        total_val_autoencoder_loss_A_A,
                                        total_val_autoencoder_loss_B_B])

                    store_util.save_item_to_gdrive_folder(
                        file_to_save_path=summary_filepath, gdrive_folder="{}_{}".format(experiment_name, "logs"))

                    if best_train_loss >= total_train_autoencoder_loss_A_A + total_train_autoencoder_loss_B_B:
                        best_train_loss = total_train_autoencoder_loss_A_A + total_train_autoencoder_loss_B_B

                    if best_val_loss >= total_val_autoencoder_loss_A_A + total_val_autoencoder_loss_B_B:
                        best_val_loss = total_val_autoencoder_loss_A_A + total_val_autoencoder_loss_B_B
                        save_path = saver.save(sess, "{}/{}_best_val_{}.ckpt".format(saved_models_filepath,
                                                                                     experiment_name, e))
                    if (e+1) % 5 == 0:
                        temp_learning_rate = current_learning_rate
                        current_learning_rate = current_learning_rate / 2.
                        if current_learning_rate <= 0.00005:
                            current_learning_rate = 0.00005
                        print("change learning rate from {} to {}".format(temp_learning_rate, current_learning_rate))

                pbar_e.update(1)
