**Requirements:**
 
    Python 3
    Opencv 3
    Tensorflow 1.3+(?)
    Keras 2
    dlib - use this to install https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
 
you also need a modern GPU with CUDA support for best performance
 
**How to run:**
 
    python train.py
 
As you can see, the code is embarrassingly simple. I don't think it's worth the trouble to keep it secret from everyone.
I believe the community are smart enough to finish the rest of the owl.
 
If there is any question, welcome to discuss here.
 
**Some tips:**
 
Reuse existing models will train much faster than start from nothing.  
If there are not enough training data, start with someone looks similar, then switch the data.
