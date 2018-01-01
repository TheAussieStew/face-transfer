import argparse
import cv2
import dlib
import json
import numpy
import skimage
from pathlib import Path
from tqdm import tqdm
from umeyama import umeyama

from face_alignment import FaceAlignment, LandmarksType


def monkey_patch_face_detector(_):
    detector = dlib.get_frontal_face_detector()

    class Rect(object):
        def __init__(self, rect):
            self.rect = rect

    def detect(*args):
        return [Rect(x) for x in detector(*args)]
    return detect


dlib.cnn_face_detection_model_v1 = monkey_patch_face_detector
FACE_ALIGNMENT = FaceAlignment(
    LandmarksType._2D, enable_cuda=False, flip_input=False)

mean_face_x = numpy.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

mean_face_y = numpy.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

landmarks_2D = numpy.stack([mean_face_x, mean_face_y], axis=1)


def transform(image, mat, size, padding=0):
    mat = mat * size
    mat[:, 2] += padding
    new_size = int(size + padding * 2)
    return cv2.warpAffine(image, mat, (new_size, new_size))


def get_cropped_faces(image):
    """Get pose invariant crop of all the faces in image"""
    cropped_faces = []
    try:
        faces = FACE_ALIGNMENT.get_landmarks(image)
    except RuntimeError as err:
        print("{} Skipping this image...".format(err))
        return None

    if faces is None:
        return None
    if len(faces) == 0:
        return None

    for points in faces:
        alignment = umeyama(points[17:], landmarks_2D, True)[0:2]
        aligned_image = transform(image, alignment, 160, 48)
        # cv2.imshow('aligned_image', aligned_image)
        # cv2.waitKey(5)
        cropped_faces.append([list(alignment.ravel()), aligned_image])

    return cropped_faces


def main(args):
    input_dir = Path(args.input_dir)
    assert input_dir.is_dir()

    output_dir = Path('data/training_data') / input_dir.stem
    print("Output dir:")
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = input_dir / args.output_file

    input_files = list(input_dir.glob("*"))
    assert len(input_files) > 0, "Can't find input files"

    def iter_face_alignments():
        for fn in tqdm(input_files):
            image = cv2.imread(str(fn))
            if image is None:
                tqdm.write("Can't read image file")
                continue

            cropped_faces = get_cropped_faces(image)

            if cropped_faces is None:
                continue

            for i, (mat, aligned_image) in enumerate(cropped_faces):
                if len(cropped_faces) == 1:
                    out_fn = "{}.jpg".format(Path(fn).stem)
                else:
                    out_fn = "{}_{}.jpg".format(Path(fn).stem, i)

                out_fn = output_dir / out_fn
                cv2.imwrite(str(out_fn), aligned_image)

                yield str(fn), str(out_fn), mat

    # Write face alignments to json file
    face_alignments = list(iter_face_alignments())
    with output_file.open('w') as f:
        results = json.dumps(face_alignments, ensure_ascii=False)
        f.write(results)

    print("Save face alignments to output file:", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str, nargs='?', default='aligned')
    parser.add_argument("output_file", type=str, nargs='?',
                        default='alignments.json')

    parser.set_defaults(only_one_face=False)
    parser.add_argument('--one-face', dest='only_one_face',
                        action='store_true')
    parser.add_argument('--all-faces', dest='only_one_face',
                        action='store_false')

    parser.add_argument("--file-type", type=str, default='jpg')

    main(parser.parse_args())
