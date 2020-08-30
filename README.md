# Deep Active Lesion Segmentation


Official repositoy for Deep Active Lesion Segmentaion ( DALS ). 

<img src="./Images/DALS_Framework.png" width="100%">

# Environment Setup

To install all the requirments for this project, simply run this: 

pip install -r requirments.txt 

Note that this work uses the last version of Tensorflow 1.x which is (1.15.0). Assuming that conda is installed in your system, and if you needed to install the cuda and cuDNN compatible verisons for Tensorflow, you can simply use:

```
conda create --name tf_gpu tensorflow-gpu==1.15.0 
```

# Arguments 

Training options:
```
parser.add_argument('--logdir', default='network', type=str) # Directory to save the model
parser.add_argument('--mu', default=0.2, type=float) # mu coefficient in levelset acm model 
parser.add_argument('--nu', default=5.0, type=float) # nu coefficient in levelset acm model
parser.add_argument('--batch_size', default=1, type=int) # batch size 
parser.add_argument('--train_sum_freq', default=150, type=int) # Frequency of validation during training
parser.add_argument('--train_iter', default=150000, type=int) # Number of training iterations
parser.add_argument('--acm_iter_limit', default=300, type=int) # Number of levelset acm iterations
parser.add_argument('--img_resize', default=512, type=int) # Size of input image
parser.add_argument('--f_size', default=15, type=int) # Size of pooling filter for fast intensity lookup ( in levelset acm model)
parser.add_argument('--train_status', default=1, type=int) # Status of training. 1: train from scratch 2: continue training from last checkpoint 3: inference  
parser.add_argument('--narrow_band_width', default=1, type=int) # Narrowband width in levelset acm model
parser.add_argument('--save_freq', default=1000, type=int) # Frequency of saving the moidel during training 
parser.add_argument('--lr', default=1e-3, type=float) # Training learning rate 
parser.add_argument('--gpu', default='0', type=str) # Index of gpu to be used for training

```

For instance, if you wanted to train with a batchsize of 4 and input image size of 256, you need to run:

```
python main.py --train_status=1 --img_resize=512 --batch_size=4 
```

# Paper

Official Paper: [Link](https://link.springer.com/chapter/10.1007/978-3-030-32692-0_12) <br/>

ArXiv Paper: [Link](https://arxiv.org/pdf/1908.06933.pdf) <br/>

If you use the DALS framework, its fast Tensorflow-based levelset active contours or its CNN backbone, please consider citing our paper:  

```
@inproceedings{hatamizadeh2019deep,
  title={Deep active lesion segmentation},
  author={Hatamizadeh, Ali and Hoogi, Assaf and Sengupta, Debleena and Lu, Wuyue and Wilcox, Brian and Rubin, Daniel and Terzopoulos, Demetri},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={98--105},
  year={2019},
  organization={Springer}
}
```

# Segmentation comparison of DALS with Expert Radiologist and U-Net 

<img src="./Images/DALS_outputs.png" width="95%">




