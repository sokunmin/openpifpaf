* [Why delete MaxPool2d in stem](https://github.com/vita-epfl/openpifpaf/issues/55)
```
the meaning of the convolutional weights changed with the modification, 
but they didn't become useless. 
Leaving out the input pooling layer just means that the following layers 
resolve to smaller image patches. 

In the interpretation where convolutions are regarded as pattern matchers, 
the ResNet blocks still match lines and corners etc and the higher blocks 
still match more complicated patterns, just at a smaller scale in the input image. 

But then ImageNet is trained on 224x224 images whereas we can train 
at a different input image size to compensate any mismatch in pattern scales and 
in fact the default training size for OpenPifPaf is 401x401.
```

* [Transfer learning vs. Scratch learning](https://github.com/vita-epfl/openpifpaf/issues/55)


* In [data.py](openpifpaf/data.py), why define `sigmas` manually rather than define by learning
https://zhuanlan.zhihu.com/p/89442276