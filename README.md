# Project Face Transfer

### Prerequisites:
 
 ```
    Anaconda
    Python 3
    OpenCV 3
    Tensorflow 1.3+
    Keras 2
    dlib    
```
Will also require a CUDA enabled GPU, plus CUDA/CUDNN drivers to be installed.
 
### Installation

This will install required packages and dependencies using conda.
```
conda env create -f environment_mac.yml
```
After this is complete, activate the environment.
```
source activate face-transfer
```
## Getting Started

To create the required template directories, data, models and test:
```
python init.py
```

## Common Issues

Issue: I'm having issues with dlib
Solution: Compile and install dlib from source
https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
