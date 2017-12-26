# Project Face Transfer

### Prerequisites:
 ```
    Anaconda
    Python 3
    OpenCV 3
    Tensorflow 1.3
    Keras 2
    boost
    boost-python
    dlib (You should install using the link below, unless you want to suffer the pain of endless configuration issues) 
    https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
    CUDA Toolkit 8.0 and cuDNN v5.1 (Requires a CUDA GPU)
```
### Installation
This will install required packages and dependencies using conda.
```
conda env create -f environment_mac.yml
```
After this is complete, activate the environment.
```
source activate face-transfer
```
### Getting Started
To create the required template directories, data, models and test:
```
python init.py
```
#### Creating Raw Data
You could manually download images. A better method is to use Bulk-Bing-Image-Downloader.
https://github.com/ostrolucky/Bulk-Bing-Image-downloader

To gather images from videos, use ffmpeg to convert the video to frames with a limited fps.
```
ffmpeg -i myvideo.avi -vf fps=1 img%03d.jpg
```
#### Creating Training Data
Create a folder in data/raw_data/ and fill it with images of the person whose face you want to transfer. For example:  

|--- data  
|---|-- raw_data  
|---|---|-- barack_obama  
|---|-- training_data  
|---|---|-- barack_obama  

Run the following to crop and align the raw data. The resulting training data is placed in data/training_data/firstname_lastname
```
python align_images.py
```
### Common Issues
Issue: I'm having issues with dlib
Solution: Compile and install dlib from source
