# Project Face Transfer

## Prerequisites:
 ```
    Anaconda
    Python 3
    OpenCV 3
    Tensorflow 1.3
    Keras 2
    boost
    boost-python
    dlib (You should install dlib using the link below, unless you want to suffer the pain of endless configuration issues) 
    https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
    CUDA Toolkit 8.0 and cuDNN v5.1 (Requires a CUDA GPU)
```
## Installation
This will install required packages and dependencies using conda.
```
conda env create -f environment_mac.yml
```
After this is complete, activate the environment.
```
source activate face-transfer
```
## Getting Started
To create the required directories, run the following in your terminal:
```
python init.py
```
### Gathering/Creating Raw Data
Create a folder in data/raw_data/ and fill it with images of the person whose face you want to transfer. This is the required directory structure:  

|--- data  
|---|-- raw_data  
|---|---|-- ryan_gosling  
|---|-- training_data  

To do this you could manually download images. But here are some faster methods.

1. Use Bulk-Bing-Image-Downloader.
https://github.com/ostrolucky/Bulk-Bing-Image-downloader
```
./bbid.py -s "ryan gosling" --filters +filterui:imagesize-wallpaper+filterui:face-portrait -o data/raw_data/ryan_gosling
```

2. Gather images from videos, by using ffmpeg to convert the video into frames with a limited framerate.
```
ffmpeg -i ryan_gosling.webm -ss 00:00:47 -t 00:00:15 -q:v 1 -vf fps=0.5 rg%d.jpeg
# -ss is the start time
# -t is the length after the start time
# -q:v controls the quality of the video encoder, lower is better
```

You should now go through all your images and delete images without a face in them. The script skips images without faces in them, but each image takes a while to scan so it's best if we take out images without a face in them. If there are multiple faces in an image, that's fine - we'll deal with that after we've created our training data.

### Creating Training Data

Run the following to crop and align the raw data. 
```
python align_images.py data/raw_data/ryan_gosling
```
The resulting training data is placed in data/training_data/firstname_lastname. Go through the images and delete any images without clear faces in them.

|--- data  
|---|-- raw_data  
|---|---|-- barack_obama  
|---|-- training_data  
|---|---|-- barack_obama  

### Training models
Run the following to train 2 encoder:decoder pairs (autoencoders):
encoder_new.h5:decoder_ryan_gosling.h5
encoder_new.h5:decoder_daisy_ridley.h5
```
python train.py new ryan_gosling daisy_ridley
```
The default batch size of 64 should take about 1 second to process on a Tesla K80.

### Processing video
Run the following to process and output a video:
```
python process.py old ryan_gosling --video --saveOutput --frame_limit 30 --dir test/input/videos/conway_video.mov --outputDirectory test/output/videos/
```

## Common Issues
Issue: I'm having issues with dlib  
Solution: Compile and install dlib from source

## Acknowledgements
Code borrows from https://github.com/deepfakes/faceswap
