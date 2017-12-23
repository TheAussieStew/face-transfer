import os
import sys
import cv2
import argparse
import time
import glob

import face_recognition
import numpy as np
from scipy.spatial import distance

CROP_SIZE  = (256,256)
CROP_RATIO = 0.5

facial_features = [
    'chin',
    'left_eyebrow',
    'right_eyebrow',
    'nose_bridge',
    'nose_tip',
    'left_eye',
    'right_eye',
    'top_lip',
    'bottom_lip'
]

#faceLocations = face_recognition.face_locations(frame)
#numFaces      = len(faceLocations)

from test import convert_one_image

from model import Autoencoder, Encoder, Decoder

encoder  .load_weights( "models/encoder.h5"   )
decoder_A.load_weights( "models/decoder_A.h5" )
decoder_B.load_weights( "models/decoder_B.h5" )

def processVideo(videoFile,FLAGS):
	videoCapture  = cv2.VideoCapture(videoFile)
	isOpened      = videoCapture.isOpened()

	if not isOpened:
		print("failed to open",videoFile)		
		return None

	width  = int( videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH ) ) 
	height = int( videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT) ) 
	fps    = int( videoCapture.get(cv2.CAP_PROP_FPS) )
	processingFps = fps

	base        = os.path.basename(videoFile)
	fileName, _ = os.path.splitext(base)
	
	print("Testing on",base,"   FPS = ",fps)
	
	if FLAGS.saveOutput and isOpened:
		fourcc   = cv2.VideoWriter_fourcc(*'XVID')
		outVideo = cv2.VideoWriter(FLAGS.outputDirectory + fileName + 
			 '_output.avi',fourcc, processingFps, (width,height))

	frameCount = 0

	while isOpened:
		ret, frame = videoCapture.read()
		if not ret:
			break
		frameCount = frameCount + 1
		# skip frames based on the value of processingFps
		if frameCount%(fps/processingFps) :
			continue

		# face detection
		#faceLocations = face_recognition.face_locations(frame,number_of_times_to_upsample=0)
		face_landmarks_list = face_recognition.face_landmarks(frame)
		numFaces      = len(face_landmarks_list)

		# process one face if detected
		if numFaces:
			noseTipCenter = face_landmarks_list[0]["nose_bridge"][3]
			leftEye  = face_landmarks_list[0]["left_eye"]			
			rightEye = face_landmarks_list[0]["right_eye"]
			faceWidth = 1.9 * max(distance.euclidean(leftEye[0],noseTipCenter),
								  distance.euclidean(rightEye[0],noseTipCenter))
						
			(left,top)     = np.subtract(noseTipCenter , (faceWidth,faceWidth) )  
			(right,bottom) = np.add(noseTipCenter , (faceWidth,faceWidth) )
			vertical_offset = 20
			left,top,right,bottom = int(left),int(top + vertical_offset),int(right),int(bottom + vertical_offset)

			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 6)
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

			face = frame[top:bottom, left:right]
			# resize to CROP_SIZE whixh is 256 by 256
			face = cv2.resize(face, CROP_SIZE)
            # transform using autoencoder
			new_face = convert_one_image(autoencoder_A, face)
			# upscale the new face to put it back to the original frame
			new_face = cv2.resize(new_face, (bottom - top,bottom - top))
            # put new face onto the original image
			frame[top:bottom, left:right] = new_face

		#cv2.imshow('image',frame)
		if frameCount % 10 == 0:
			print("{} frames processed".format(frameCount))
		cv2.waitKey(1)

		if FLAGS.saveOutput:
			outVideo.write(frame)	
	videoCapture.release()
	cv2.destroyAllWindows()

	if FLAGS.saveOutput:
		outVideo.release()

	return None

def processImage(imFile,FLAGS):
	img = cv2.imread(imFile)

	base        = os.path.basename(imFile)
	fileName, _ = os.path.splitext(base)
	
	if img is None:
		print("failed to open ",base)
		return None, 0
	
	print("Testing on",base)

	face_landmarks_list = face_recognition.face_landmarks(img)
	numFaces = len(face_landmarks_list)

	if numFaces:
		noseTipCenter = face_landmarks_list[0]["nose_bridge"][3]
		leftEye  = face_landmarks_list[0]["left_eye"]			
		rightEye = face_landmarks_list[0]["right_eye"]
		faceWidth = 1.5 * max(distance.euclidean(leftEye[0],noseTipCenter),
							  distance.euclidean(rightEye[0],noseTipCenter))
		
		(left,top)     = np.subtract(noseTipCenter , (faceWidth,faceWidth) )  
		(right,bottom) = np.add(noseTipCenter , (faceWidth,faceWidth) )
		left,top,right,bottom = int(left),int(top),int(right),int(bottom)

		face = img[top:bottom, left:right]
		# resize to CROP_SIZE whixh is 256 by 256
		face = cv2.resize(face, CROP_SIZE)
		cv2.imwrite(FLAGS.outputDirectory + fileName + '_output.png', face)



def main(FLAGS):
	t0 = time.time()

	if FLAGS.image:
		files = glob.glob(os.path.join(FLAGS.dir, '*.*'))
		if files == []:
			print('No files found in folder: ' + FLAGS.folder)
			exit(1)
		for imFile in files:
			processImage(imFile, FLAGS) 
	if FLAGS.video:
		processVideo(FLAGS.dir, FLAGS)

	print("Total processing time = ",int(time.time()-t0),"secs")

if __name__ == '__main__':
	parser   = argparse.ArgumentParser()

	parser.add_argument('--image', action='store_true' ,help='')
	parser.add_argument('--video', action='store_true' ,help='')

	parser.add_argument('--saveOutput' ,action='store_true' ,help='')
	parser.add_argument('--dir', type=str, default='',help='')
	parser.add_argument('--noDisplay', action='store_true' ,help='')
	parser.add_argument('--processingFps', type=int ,default = 1,help='')
	parser.add_argument('--outputDirectory', type=str, default='',help='')

	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
