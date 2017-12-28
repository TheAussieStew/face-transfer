import argparse
import cv2
import json
import numpy
from pathlib import Path
from tqdm import tqdm

from model import load_autoencoder


def adjust_avg_color(img_old, img_new):
    w, h, c = img_new.shape
    for i in range(img_new.shape[-1]):
        old_avg = img_old[:, :, i].mean()
        new_avg = img_new[:, :, i].mean()
        diff_int = (int)(old_avg - new_avg)
        for m in range(img_new.shape[0]):
            for n in range(img_new.shape[1]):
                temp = (img_new[m, n, i] + diff_int)
                if temp < 0:
                    img_new[m, n, i] = 0
                elif temp > 255:
                    img_new[m, n, i] = 255
                else:
                    img_new[m, n, i] = temp


def smooth_mask(img_old, img_new):
    w, h, c = img_new.shape
    crop = slice(0, w)
    mask = numpy.zeros_like(img_new)
    mask[h // 15:-h // 15, w // 15:-w // 15, :] = 255
    mask = cv2.GaussianBlur(mask, (15, 15), 10)
    img_new[crop, crop] = mask / 255 * img_new + (1 - mask / 255) * img_old


def convert_one_image(autoencoder, image, mat):
    size = 64
    old_face = cv2.warpAffine(image, mat * size, (size, size))
    old_face = numpy.expand_dims(old_face, 0)
    new_face = autoencoder.predict(face / 255.0)[0]
    new_face = numpy.clip(new_face * 255, 0, 255).astype(image.dtype)

    # blending
    adjust_avg_color(old_face, new_face)
    smooth_mask(old_face, new_face)

    # copy new face onto old image
    new_image = numpy.copy(image)
    image_size = image.shape[1], image.shape[0]
    cv2.warpAffine(new_face, mat * size, image_size, new_image,
                   cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT)
    return new_image


def main(args):
    autoencoder = load_autoencoder(args.encoder, args.decoder)
    input_dir = Path(args.input_dir)
    assert input_dir.is_dir()

    alignments = input_dir / args.alignments
    with alignments.open() as f:
        alignments = json.load(f)

    output_dir = input_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_file, face_file, mat in tqdm(alignments):
        image = cv2.imread(str(image_file))
        face = cv2.imread(str(face_file))

        mat = numpy.array(mat).reshape(2, 3)

        if image is None:
            print("No image found")
            continue
        if face is None:
            print("No face found")
            continue

        cv2.imshow("image", image)
        cv2.waitKey(4000)

        new_image = convert_one_image(autoencoder, image, mat)

        output_file = output_dir / Path(image_file).name
        cv2.imwrite(str(output_file), new_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("encoder", type=str)
    parser.add_argument("decoder", type=str)
    parser.add_argument("alignments", type=str, nargs='?',
                        default='alignments.json')
    parser.add_argument("output_dir", type=str, nargs='?', default='merged')
    main(parser.parse_args())
