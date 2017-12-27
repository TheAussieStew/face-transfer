import cv2
import numpy
import argparse
from pathlib import Path
from tqdm import tqdm

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data

from model import Encoder, Decoder, Autoencoder


def main(FLAGS):
    encoder = Encoder()
    decoder_A = Decoder()
    decoder_B = Decoder()
    autoencoder_A = Autoencoder(encoder, decoder_A)
    autoencoder_B = Autoencoder(encoder, decoder_B)

    # create paths
    models_dir = Path("models")
    training_data_dir = Path("data/training_data")
    person_A_training_data = training_data_dir / Path(FLAGS.person_A)
    person_B_training_data = training_data_dir / Path(FLAGS.person_B)
    encoder_fn = models_dir / Path("encoder" + ".h5")
    decoder_A_fn = models_dir / Path("decoder_" + FLAGS.person_A + ".h5")
    decoder_B_fn = models_dir / Path("decoder_" + FLAGS.person_B + ".h5")

    try:
        encoder.load_weights(str(encoder_fn))
    except IOError as err:
        user_input = input(
            "Couldn't load encoder. Do you want to create a new one? (Y/n)\n")
        if user_input is not "Y":
            exit(1)

    try:
        decoder_A.load_weights(str(decoder_A_fn))
        decoder_B.load_weights(str(decoder_B_fn))
    except:
        user_input = input(
            "Couldn't load decoder/s. Do you want to create a new one? (Y/n)\n")
        if user_input is not "Y":
            exit(1)

    def save_model_weights():
        encoder  .save_weights(str(encoder_fn))
        decoder_A.save_weights(str(decoder_A_fn))
        decoder_B.save_weights(str(decoder_B_fn))
        print("Saved model weights")

    images_A = get_image_paths(str(person_A_training_data))
    images_B = get_image_paths(str(person_B_training_data))
    images_A = load_images(images_A) / 255.0
    images_B = load_images(images_B) / 255.0

    print("Press 'q' to stop training and save model")

    for iteration in tdqm(range(1000000)):
        batch_size = 64
        warped_A, target_A = get_training_data(images_A, batch_size)
        warped_B, target_B = get_training_data(images_B, batch_size)

        loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
        print("autoencoder_A loss: {} | autoencoder_B loss: {}".format(loss_A, loss_B))

        if iteration % 5 == 0:
            save_model_weights()
            test_A = target_A[0:14]
            test_B = target_B[0:14]

            figure_A = numpy.stack([
                test_A,
                autoencoder_A.predict(test_A),
                autoencoder_B.predict(test_A),
            ], axis=1)
            figure_B = numpy.stack([
                test_B,
                autoencoder_B.predict(test_B),
                autoencoder_A.predict(test_B),
            ], axis=1)

            figure = numpy.concatenate([figure_A, figure_B], axis=0)
            figure = figure.reshape((4, 7) + figure.shape[1:])
            figure = stack_images(figure)

            figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

            cv2.imshow("", figure)
            key = cv2.waitKey(1)

        if key == ord('q'):
            save_model_weights()
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("person_A", type=str)
    parser.add_argument("person_B", type=str)
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
