import os
import cv2
import argparse
import time
import glob
import numpy as np
from tqdm import tqdm

from model import load_autoencoder
from align_images import get_cropped_faces
from merge_faces import convert_one_image


def process_video(video_file, FLAGS):
    video_capture = cv2.VideoCapture(video_file)
    is_opened = video_capture.isOpened()

    if not is_opened:
        print("failed to open", video_file)
        return None

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    processing_fps = fps

    base = os.path.basename(video_file)
    filename, _ = os.path.splitext(base)

    print("Testing on", base, "   FPS = ", fps)

    if FLAGS.rescale:
        height = int(height * FLAGS.rescale_ratio)
        width = int(width * FLAGS.rescale_ratio)

    if FLAGS.saveOutput and is_opened:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_video = cv2.VideoWriter(FLAGS.outputDirectory + filename +
                                   "_output.avi", fourcc, processing_fps, (width, height))

    autoencoder = load_autoencoder(FLAGS.encoder, FLAGS.decoder)

    progress_bar = tqdm(total=total_frame_count, unit="frame")
    frame_count = 0
    while is_opened:
        # process video until we reach frame limit
        if frame_count == FLAGS.frame_limit:
            break

        ret, frame = video_capture.read()

        if not ret:
            break

        # rescale frame
        if FLAGS.rescale:
            height, width, layers = frame.shape
            height = int(height * FLAGS.rescale_ratio)
            width = int(width * FLAGS.rescale_ratio)
            frame = cv2.resize(frame, (new_width, new_height))

        cropped_faces = get_cropped_faces(frame)

        if cropped_faces is None:
            out_video.write(frame)
            progress_bar.update(1)
            frame_count += 1
            continue

        for mat, aligned_image in cropped_faces:
            mat = np.array(mat).reshape(2, 3)
            new_image = convert_one_image(autoencoder, frame, mat)

        # how often to show the preview
        if FLAGS.display and frame_count % 5 == 0:
            cv2.imshow("new_image", new_image)
            cv2.waitKey(1)

        if FLAGS.saveOutput:
            out_video.write(new_image)

        progress_bar.update(1)
        frame_count += 1

    progress_bar.close()
    video_capture.release()
    cv2.destroyAllWindows()

    if FLAGS.saveOutput:
        out_video.release()


# def processImage(imFile, FLAGS):
#     img = cv2.imread(imFile)

#     base = os.path.basename(imFile)
#     filename, _ = os.path.splitext(base)

#     if img is None:
#         print("failed to open ", base)
#         return None, 0

#     print("Testing on", base)

#     face_landmarks_list = face_recognition.face_landmarks(img)
#     numFaces = len(face_landmarks_list)

#     if numFaces:
#         noseTipCenter = face_landmarks_list[0]["nose_bridge"][3]
#         leftEye = face_landmarks_list[0]["left_eye"]
#         rightEye = face_landmarks_list[0]["right_eye"]
#         faceWidth = 1.5 * max(distance.euclidean(leftEye[0], noseTipCenter),
#                               distance.euclidean(rightEye[0], noseTipCenter))

#         (left, top) = np.subtract(noseTipCenter, (faceWidth, faceWidth))
#         (right, bottom) = np.add(noseTipCenter, (faceWidth, faceWidth))
#         left, top, right, bottom = int(left), int(top), int(right), int(bottom)

#         face = img[top:bottom, left:right]
#         # resize to CROP_SIZE whixh is 256 by 256
#         face = cv2.resize(face, CROP_SIZE)
#         cv2.imwrite(FLAGS.outputDirectory + filename + "_output.png", face)


def main(FLAGS):
    t0 = time.time()

    # Image processing incomplete for now
    # if FLAGS.image:
    #     files = glob.glob(os.path.join(FLAGS.dir, "*.*"))
    #     if files == []:
    #         print("No files found in folder: " + FLAGS.folder)
    #         exit(1)
    #     for imFile in files:
    #         processImage(imFile, FLAGS)
    if FLAGS.video:
        process_video(FLAGS.dir, FLAGS)

    print("Total processing time = ", int(time.time() - t0), "secs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", type=str)
    parser.add_argument("decoder", type=str)
    parser.add_argument("--image", action="store_true", help="")
    parser.add_argument("--video", action="store_true", help="")
    parser.add_argument("--saveOutput", action="store_true", help="")
    parser.add_argument("--dir", type=str, default="", help="")
    parser.add_argument("--display", action="store_true", help="")
    parser.add_argument("--outputDirectory", type=str, default="", help="")
    parser.add_argument("--frame_limit", type=int, default="1000000", help="")
    parser.add_argument("--rescale", action="store_true", help="")
    parser.add_argument("--rescale_ratio", type=float, help="")


    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
