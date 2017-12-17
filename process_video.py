# Use like this:
# python process_video.py [video_filename]

import os
import time
import cv2
from pathlib import Path

import test

from generate_training_data import detect_faces

output_dir = Path('test_output_videos/obama')
output_dir.mkdir(parents=True, exist_ok=True)

frame_count = 0
def main():
    video = cv2.VideoCapture(args.filename)

    start_time = time.time()
    fps = video.FPS().start()
    
    while video.isOpened():
        fps.update()
        ret, frame = video.read()
        
        face_rectangles = detect_faces(frame)
        
        # Only perform if there's one face detected
        if len(face_rectangles) == 1:
            for face in face_rectangles:
                # Crop the face
                face = frame[face_rectangle[1]:face_rectangle[3],\
                                face_rectangle[0]:face_rectangle[2]]
                resized_face = cv2.resize(face, (256, 256))
                # Use the autoencoder to process the face
                new_face = convert_one_image(autoencoder_B, resized_face)
                # Paste the new face back on the original frame
                frame[face_rectangle[1]:face_rectangle[3],\
                                face_rectangle[0]:face_rectangle[2]] = new_face
                output_file = output_dir / Path(args.filename).name
                cv2.imwrite(str(output_file) + '{}'.format(frame_count), frame)
                frame_count += 1
    
    elapsed_time = time.time() - start.time()
    print('Elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('Approx. FPS: {:.2f}'.format(fps.fps()))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Make sure these files are in the root directory
    # You can download a trained facial shape predictor and recognition model from:
    # http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
    # http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    predictor_fn = "shape_predictor_5_face_landmarks.dat"
    face_rec_model_path_fn = "dlib_face_recognition_resnet_model_v1.dat"

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_fn)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path_fn)

    main()
