# Use like this:
# python process_video.py [video_filename]

import os
import time
import cv2
import dlib
import argparse
from pathlib import Path

from test import convert_one_image

from generate_training_data import detect_faces

output_dir = Path('test_output_videos/obama')
output_dir.mkdir(parents=True, exist_ok=True)

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

encoder  .load_weights( "models/encoder.h5"   )
decoder_A.load_weights( "models/decoder_A.h5" )
decoder_B.load_weights( "models/decoder_B.h5" )

def main():
    video = cv2.VideoCapture(args.filename)

    start_time = time.time()

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    out = cv2.VideoWriter('output_obama.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        
        face_rectangles = detect_faces(frame)
        
        # Only perform if there's one face detected
        if len(face_rectangles) == 1:
            for face_rectangle in face_rectangles:
                print(face_rectangle)
                # Crop the face
                face = frame[face_rectangle[1]:face_rectangle[3],\
                                face_rectangle[0]:face_rectangle[2]]
                # Resize
                resized_face = cv2.resize(face, (256, 256))
                # Use the autoencoder to process the face
                new_face = convert_one_image(autoencoder_A, resized_face)
                # Resize to original shape
                width = face_rectangle[2] - face_rectangle[0]
                height = face_rectangle[3] - face_rectangle[1]
                resized_new_face = cv2.resize(new_face, (width, height))
                # Paste the new face back on the original frame
                frame[face_rectangle[1]:face_rectangle[3],\
                                face_rectangle[0]:face_rectangle[2]] = resized_new_face
                output_file = output_dir / Path(args.filename).name
                out.write(frame)
                #cv2.imwrite(str(output_file) + '{}.png'.format(frame_count), frame)
                frame_count += 1

                # Press Q on keyboard to stop recording
                if cv2.waitKey(1) & 0xFF == ord('q'):
                  break

    # When everything is complete, release the video capture and video write objects
    out.release()
    video.release()

    elapsed_time = time.time() - start.time()
    print('Elapsed time (total): {:.2f}'.format(elapsed_time))
    print('Approx. FPS: {:.2f}'.format(frame_count / elapsed_time))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
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
