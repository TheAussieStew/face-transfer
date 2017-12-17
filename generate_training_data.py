import cv2
import dlib
from pathlib import Path
from PIL import Image

from utils import get_image_paths

# Get current working directory
cwd = Path()

# Make sure these files are in the root directory
# You can download a trained facial shape predictor and recognition model from:
# http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
# http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
predictor_fn = "shape_predictor_5_face_landmarks.dat"
face_rec_model_path_fn = "dlib_face_recognition_resnet_model_v1.dat"

# http://dlib.net/face_recognition.py.html
def detect_faces(image):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detected_faces = face_detector(image, 1)
    print("Number of faces detected: {}".format(len(detected_faces)))
    face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]
    return face_frames

def main():
    # Load images filenames
    image_filenames = get_image_paths('raw_data/obama')

    output_dir = Path('training_data/obama')
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in image_filenames:
        print("Scanning: " + filename)
        image = cv2.imread(filename)
        detected_faces = detect_faces(image)
        for n, face_rectangle in enumerate(detected_faces):
            if len(detected_faces) == 1:
                # Crop the face
                face = image[face_rectangle[1]:face_rectangle[3], face_rectangle[0]:face_rectangle[2]]
                resized_face = cv2.resize(face, (256, 256))
                output_file = output_dir / Path(filename).name
                cv2.imwrite(str(output_file), resized_face)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_fn)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path_fn)

    main()
