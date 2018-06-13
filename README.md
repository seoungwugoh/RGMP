# Fast Video Object Segmentation by Reference-Guided Mask Propagation
#### Seoung Wug Oh, Joon-Young Lee, Kalyan Sunkavalli, Seon Joo Kim
#### CVPR 2018

This is the official demo code for the paper. [PDF](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1029.pdf)
___
## Test Environment
- Ubuntu 
- python 3.6
- Pytorch 0.3.1
  + installed with CUDA.



## How to Run
1) Download [DAVIS-2017](https://davischallenge.org/davis2017/code.html).
2) Edit path for `DAVIS_ROOT` in run.py.
``` python
DAVIS_ROOT = '<Your DAVIS path>'
```
3) Download [weights.pth](https://www.dropbox.com/s/qrtrbdn3n8qy5en/weights.pth?dl=0) and place it the same folde as run.py.
4) To run single-object video object segmentation on DAVIS-2016 validation.
``` 
python run.py
```
5) To run multi-object video object segmentation on DAVIS-2017 validation.
``` 
python run.py -MO
```
6) Results will be saved in `./results/SO` or `./results/MO`.


## Use
#### This software is for Non-commercial Research Purposes only.

If you use this code please cite:
```
@InProceedings{oh2018fast,
author = {Oh, Seoung Wug and Lee, Joon-Young and Sunkavalli, Kalyan and Kim, Seon Joo},
title = {Fast Video Object Segmentation by Reference-Guided Mask Propagation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2018}
}
```




  










