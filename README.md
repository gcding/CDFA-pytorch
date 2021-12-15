# CDFA-pytorch

Code for Unsupervised crowd counting via cross-domain feature adaptation.

Pre-trained models
---
[Google Drive](https://drive.google.com/drive/folders/1d5OqGtuP3rivzJnuEOOuAKFzv6cJzFnN?usp=sharing)

[Baidu Cloud](https://pan.baidu.com/s/1t_cXigANGzMC8VxbG8MV3g) : t4qc

Environment
---
We are good in the environment:

python 3.6

CUDA 9.2

Pytorch 1.2.0

numpy 1.19.2

matplotlib 3.3.4

Usage
---
We provide the test code for our model. 
The `result_gcc_qnrf.pth` model is adapted from the GCC dataset to the UCF_QNRF dataset. 
We randomly select an image from the UCF_QNRF dataset and place it in the image folder.
And you can either choose the other images for a test.

We are good to run:

```
python test.py --model CDFA --model_state ./model/result_gcc_qnrf.pth --out ./out/out.png
```

We will release more pre-trained models soon.
Please see the paper for more details.

Citation
---

```
Coming soon
```

Acknowledgement
---

Thanks to these repositories
- [C-3 Framework](https://github.com/gjy3035/C-3-Framework)
- [EDSC](https://github.com/Xianhang/EDSC-pytorch)
