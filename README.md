# DLProject2019
Deep Learning Project_E!=m^2z


**Food-101**
WEBSITE: https://www.vision.ee.ethz.ch/datasets_extra/food-101/
DATASET: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
Download food-101.tar.gz and unzip it and copy it inside food-101 directory as follows:
```
├── ...
└── datasets 
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
python3 main.py --exp_name $experiment --dataset food-101 --test_N_way 5 --train_N_way 5 --train_N_shots 5 --test_N_shots 5 --batch_size 16  --dec_lr=10000  --iterations 80000
```

**mini_ImageNet**
Create images.zip file and copy it inside mini_imagenet directory:
```
.
├── ...
└── datasets                    
   └── compressed                
      └── mini_imagenet
         └── images.zip
```

The images.zip file must contain the splits and images in the following format:
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
The splits {test.csv, train.csv, val.csv} are in mini_imagenet folder. 

Run: 
```
experiment = mini_imagenet_N5_S1
python3 main.py --exp_name $experiment --dataset mini_imagenet --test_N_way 5 --train_N_way 5 --train_N_shots 1 --test_N_shots 1 --batch_size 16  --dec_lr=10000  --iterations 80000
```

**Omniglot**

Download images_background.zip and images_evaluation.zip files from  and copy it inside the omniglot directory:
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
python3 main.py --exp_name $experiment --dataset omniglot --test_N_way 20 --train_N_way 20 --train_N_shots 5 --test_N_shots 5 --batch_size 16  --dec_lr=10000  --iterations 80000
```
