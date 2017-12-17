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