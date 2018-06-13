# Fast Video Object Segmentation by Reference-Guided Mask Propagation
#### Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
#### CVPR 2018

This is the official demo code for the paper. [PDF](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1029.pdf)
___
## Test Environment
- Ubuntu 
- python 3.6
  + Pytorch 0.3.1 (installed with CUDA capability)
  + opencv


## How to run
1) Download [DAVIS-2017](https://davischallenge.org/davis2017/code.html).
2) Edit path for `DAVIS_ROOT` in run.py.
``` python
DAVIS_ROOT = '<Your DAVIS path>'
```
3) To run single-object video object segmentation on DAVIS-2016 validation.
``` 
python run.py
```
4) To run multi-object video object segmentation on DAVIS-2017 validation.
``` 
python run.py -MO
```
5) results will be saved in `./results/SO` or `./results/SO`


## Citation
Use this code for Non-commercial Research Purposes only.

If you use this code please cite:
```
@InProceedings{oh2018fast,
author = {Oh, Seoung Wug and Lee, Joon-Young and Sunkavali, Kalyan and Kim, Seon Joo},
title = {Fast Video Object Segmentation by Reference-Guided Mask Propagation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2018}
}
```



  
  










