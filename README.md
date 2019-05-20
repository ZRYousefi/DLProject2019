# DLProject2019

This repository contains clean, readable and tested code for Deep learning course project. 
The aim is to reproduce few-shot learning methods for food-101, mini_imagenet and omniglot datasets.

This project is written in python 3.6 and assumes you have a GPU.


**Food-101**

WEBSITE: https://www.vision.ee.ethz.ch/datasets_extra/food-101/

DATASET: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

Download `food-101.tar.gz` and place the extracted files into `datasets/food-101` directory as follows:
```
├── ...
└── datasets 
  └── food-101
     └── meta               
     ├── license_agreement.txt 
     ├── README.txt 
     └── images
        ├── apple_pie
        ├── ...
        └── waffle
```    
Run: 
```
experiment = food101_N5_S5
python3 mainGNN.py --exp_name $experiment --dataset food-101 --test_N_way 5 --train_N_way 5 --train_N_shots 5 --test_N_shots 5 --batch_size 16  --dec_lr=10000  --iterations 80000
```
Running baseline on food 101 Dataset:
```
python food101_baseline.py --dest-root <root_folder_to expr_result> --n-epoch <n_epochs> --train-img-dir <root_to_train_images> --test-img-dir <root_to_test_images> --optimizer <name_of_the_optmize_supported_by_keras> --val-img-dir <validaiton_img_root_images>
```
Runnig Siamese network for classificaiton
```bash
python few_shots_classification.py --train-img-dir <root_to_train_images> --test-img-dir <root_to_test_images>
```
**mini_ImageNet**

Download: https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view, 

Place `images.zip` file inside `compressed/mini_imagenet` directory:
```
.
├── ...
└── datasets                    
   └── compressed                
      └── mini_imagenet
         └── images.zip
```

The `images.zip` file must contain the splits and images in the following format:
```
── images.zip
   ├── test.csv                
   ├── train.csv 
   ├── val.csv 
   └── images
      ├── n0153282900000006.jpg
      ├── ...
      └── n1313361300001299.jpg
```
The splits {test.csv, train.csv, val.csv} are inside `datasets/mini_imagenet` directory. 

Run: 
```
experiment = mini_imagenet_N5_S1
python3 mainGNN.py --exp_name $experiment --dataset mini_imagenet --test_N_way 5 --train_N_way 5 --train_N_shots 1 --test_N_shots 1 --batch_size 16  --dec_lr=10000  --iterations 80000
```

**Omniglot**

Download : https://github.com/brendenlake/omniglot/tree/master/python, 

Download `images_background.zip` and `images_evaluation.zip` files and copy it inside the `compressed/omniglot` directory:
```
.
├── ...
└── datasets                    
   └── compressed                
      └── omniglot
         ├── images_background.zip
         └── images_evaluation.zip
```

Run: 
```
experiment = omniglot_N20_S5
python3 mainGNN.py --exp_name $experiment --dataset omniglot --test_N_way 20 --train_N_way 20 --train_N_shots 5 --test_N_shots 5 --batch_size 16  --dec_lr=10000  --iterations 80000
```
