import os
import sys
import cv2
import argparse
import time
import glob

import numpy as np

from model import Autoencoder, Encoder, Decoder, load_autoencoder

from align_images import get_cropped_faces
from merge_faces import convert_one_image

CROP_SIZE = (256, 256)
CROP_RATIO = 0.5

facial_features = [
    "chin",
    "left_eyebrow",
    "right_eyebrow",
    "nose_bridge",
    "nose_tip",
    "left_eye",
    "right_eye",
    "top_lip",
    "bottom_lip"
]

def processVideo(videoFile, FLAGS):
    videoCapture = cv2.VideoCapture(videoFile)
    isOpened = videoCapture.isOpened()

    if not isOpened:
        print("failed to open", videoFile)
        return None

    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    processingFps = fps

    base = os.path.basename(videoFile)
    fileName, _ = os.path.splitext(base)

    print("Testing on", base, "   FPS = ", fps)

    if FLAGS.saveOutput and isOpened:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        outVideo = cv2.VideoWriter(FLAGS.outputDirectory + fileName +
                                   "_output.avi", fourcc, processingFps, (width, height))

    frameCount = 0

    autoencoder = load_autoencoder("decoder_" + FLAGS.decoder)

    while isOpened:
        ret, frame = videoCapture.read()
        if not ret:
            break
        frameCount += 1
        # skip frames based on the value of processingFps
        if frameCount % (fps / processingFps):
            continue

        # face detection
        cropped_faces = get_cropped_faces(frame)

        # skip frame if no faces are detected
        if cropped_faces is None:
            continue

        for mat, aligned_image in cropped_faces:
            mat = np.array(mat).reshape(2, 3)

            new_image = convert_one_image(autoencoder, frame, mat)

        cv2.imshow("image", new_image)
        cv2.waitKey(1)
        if frameCount % 10 == 0:
            print("{} frames processed".format(frameCount))

        if FLAGS.saveOutput:
            outVideo.write(new_image)

    videoCapture.release()
    cv2.destroyAllWindows()

    if FLAGS.saveOutput:
        outVideo.release()

    return None


# def processImage(imFile, FLAGS):
#     img = cv2.imread(imFile)

#     base = os.path.basename(imFile)
#     fileName, _ = os.path.splitext(base)

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
#         cv2.imwrite(FLAGS.outputDirectory + fileName + "_output.png", face)


def main(FLAGS):
    t0 = time.time()

    if FLAGS.image:
        files = glob.glob(os.path.join(FLAGS.dir, "*.*"))
        if files == []:
            print("No files found in folder: " + FLAGS.folder)
            exit(1)
        for imFile in files:
            processImage(imFile, FLAGS)
    if FLAGS.video:
        processVideo(FLAGS.dir, FLAGS)

    print("Total processing time = ", int(time.time() - t0), "secs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("decoder", type=str)
    parser.add_argument("--image", action="store_true", help="")
    parser.add_argument("--video", action="store_true", help="")

    parser.add_argument("--saveOutput", action="store_true", help="")
    parser.add_argument("--dir", type=str, default="", help="")
    parser.add_argument("--noDisplay", action="store_true", help="")
    parser.add_argument("--processingFps", type=int, default=1, help="")
    parser.add_argument("--outputDirectory", type=str, default="", help="")

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
