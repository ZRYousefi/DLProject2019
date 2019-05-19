from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import fnmatch
import numpy as np
from PIL import Image as pil_image
import random
import pickle
from . import parserFood
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import shutil
 
class Food101(data.Dataset):
    def __init__(self, root, dataset='food-101'):
        self.root = root
        self.dataset = dataset
        self.seed = 10
        if not self._check_exists_():
            self._init_folders_()
#            if self.check_decompress():
#                self._decompress_()
#            self._split_images_()
            self._preprocess_()

    def _init_folders_(self):
        decompress = False
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root, 'food-101')):
            os.makedirs(os.path.join(self.root, 'food-101'))
            decompress = True
        if not os.path.exists(os.path.join(self.root, 'compacted_datasets')):
            os.makedirs(os.path.join(self.root, 'compacted_datasets'))
            decompress = True
        return decompress

#    def check_decompress(self):
#        return os.listdir('%s/food_101' % self.root) == []
#
#    def _decompress_(self):
#        print("\nDecompressing Images...")
#        compressed_file = '%s/compressed/food_101/images.zip' % self.root        # TODO : check inside the folder
#        if os.path.isfile(compressed_file):
#            os.system('unzip %s -d %s/food_101/' % (compressed_file, self.root))
#        else:
#            raise Exception('Missing %s' % compressed_file)
#        print("Decompressed")

    def _check_exists_(self):
        return os.path.exists(os.path.join(self.root, 'compacted_datasets', 'food101_train.pickle')) and \
               os.path.exists(os.path.join(self.root, 'compacted_datasets', 'food101_test.pickle'))


    def _preprocess_(self):
         print('\nPreprocessing food-101 images...')
 
         source = os.path.join(self.root, 'food-101', 'images')
         extension = 'jpg'
         images_path, class_names = [], []
         for root, dirnames, filenames in os.walk(source):            
             filenames = [filename for filename in filenames]
             n_samples = 0
             for filename in fnmatch.filter(filenames, '*.'+extension):
                 images_path.append(os.path.join(root, filename))
                 class_name = root.split('/')
                 class_name = class_name[len(class_name)-1:]
                 class_name = ''.join(class_name)
                 class_names.append(class_name)
                 n_samples+=1
                 if n_samples == 300:
                     break
         
         keys_all = sorted(list(set(class_names)))
         label_encoder = {}
         label_decoder = {}
         for i in range(len(keys_all)):
             label_encoder[keys_all[i]] = i
             label_decoder[i] = keys_all[i]
         
         
         
         dest_root = os.path.join(self.root,'asghar_kuchik')
         dest_path = Path(dest_root)
         dest_path.mkdir(exist_ok=True,parents=True)
         for class_, path in zip(class_names, images_path):
             img_path = Path(path)
             img = pil_image.open(path)
             
             img = img.convert('RGB')
             img = img.resize((224, 224), pil_image.ANTIALIAS)
 #            img = np.array(img, dtype='float32')
             class_name = img_path.parent.stem
             dest_file = dest_path.joinpath(class_name)
             dest_file.mkdir(exist_ok=True, parents=True)
             dest_file = dest_file.joinpath(img_path.stem+'.JPEG')
             img.save(str(dest_file)) 
             
         res_dir = {}
         for indx, dir_name in enumerate(dest_path.iterdir()):
             res_dir[dir_name.stem] = []
             for img_file in dir_name.iterdir():
                 if img_file.suffix.lower()=='.jpeg':
                     json_rec = os.path.join(dir_name.stem,img_file.stem+img_file.suffix)  
                     res_dir[dir_name.stem].append(json_rec)
                     
                     
         json.dump(res_dir,open(os.path.join(self.root,'asghar_kuchik.json'),'w'),indent=True)


         all_set = {}
         fh = open(os.path.join(self.root,'asghar_kuchik.json'))
         all_set = json.load(fh)
         # Now we save the 80 training - 21 testing partition
         keys = sorted(list(all_set.keys()))
         random.seed(self.seed)
         random.shuffle(keys)
         
         self.sanity_check(all_set)
         print('all_set is read from asghar json')
         

         train_path = os.path.join(self.root, 'asghar_split/train')
         train_root = Path(train_path)
         train_root.mkdir(exist_ok=True,parents=True)
         test_path = os.path.join(self.root,'asghar_split/test')
         test_root = Path(test_path)
         test_root.mkdir(exist_ok=True,parents=True)
         
         tr_set = {}
         ts_set = {}
         for i in range(80):
             tr_set[keys[i]] = all_set[keys[i]]
             for i in range(80, len(keys)):
                 ts_set[keys[i]] = all_set[keys[i]]
                         
         for filename in os.listdir(dest_root):
             if filename in tr_set:
                 shutil.move(os.path.join(dest_root, filename), train_path)
             else:
                 shutil.move(os.path.join(dest_root, filename), test_path)
         shutil.rmtree(dest_root) 
         
         train_dir = {}
         train_set = {}
         class_to_id_mapper = {}
         for indx, dir_name in enumerate(train_root.iterdir()):
             train_dir[dir_name.stem] = []
             train_set[indx] = []
             class_to_id_mapper[dir_name.stem] = indx
             for img_file in dir_name.iterdir():
                 if img_file.suffix.lower()=='.jpeg':      			
                     json_rec = os.path.join(dir_name.stem,img_file.stem+img_file.suffix)  
                     train_dir[dir_name.stem].append(json_rec)
                     img_np_array = plt.imread(img_file.resolve())
                     train_set[indx].append(img_np_array)
                     
         with open(os.path.join(self.root, 'compacted_datasets', 'food_101_train.pickle'), 'wb') as handle:
             pickle.dump(train_set, handle, protocol=2)
         json.dump(train_dir,open(os.path.join(self.root,'train_asghar.json'),'w'),indent=True)
         #json.dump(class_to_id_mapper, open(dest_path.joinpath('train_label_mapper.json'),'w'), indent=True)
         
         test_dir = {}
         test_set = {}
         class_to_id_mapper = {}
         for indx, dir_name in enumerate(test_root.iterdir()):
             test_dir[dir_name.stem] = []
             test_set[indx] = []
             class_to_id_mapper[dir_name.stem] = indx
             for img_file in dir_name.iterdir():
                 if img_file.suffix.lower()=='.jpeg':
                     json_rec = os.path.join(dir_name.stem,img_file.stem+img_file.suffix)  
                     test_dir[dir_name.stem].append(json_rec)        			
                     img_np_array = plt.imread(img_file.resolve())        			
                     test_set[indx].append(img_np_array)
                     
         with open(os.path.join(self.root, 'compacted_datasets', 'food_101_test.pickle'), 'wb') as handle:                    
             pickle.dump(test_set, handle, protocol=2)					
         json.dump(test_dir,open(os.path.join(self.root,'test_asghar.json'),'w'),indent=True)
         #json.dump(class_to_id_mapper, open(dest_path.joinpath('label_mapper.json'),'w'), indent=True)


         label_encoder = {}
         keys = list(train_set.keys()) + list(test_set.keys())
         for id_key, key in enumerate(keys):
             label_encoder[key] = id_key
         with open(os.path.join(self.root, 'compacted_datasets', 'food_101_label_encoder.pickle'), 'wb') as handle:
             pickle.dump(label_encoder, handle, protocol=2)
             
         print('Images preprocessed')

    def sanity_check(self, all_set):
        all_good = True
        for class_ in all_set:
            if len(all_set[class_]) != 300:
                all_good = False
        if all_good:
            print("All classes have 300 samples")

    def load_dataset(self, train, size=(224, 224)):
        print("Loading dataset")
        if train:
            with open(os.path.join(self.root, 'compacted_datasets', 'food_101_train.pickle'), 'rb') as handle:
                data = pickle.load(handle)
        else:
            with open(os.path.join(self.root, 'compacted_datasets', 'food_101_test.pickle'), 'rb') as handle:
                data = pickle.load(handle)
        print("Num classes before rotations: "+str(len(data)))
        
        with open(os.path.join(self.root, 'compacted_datasets', 'food_101_label_encoder.pickle'),
                  'rb') as handle:
            label_encoder = pickle.load(handle)
            
        # Resize images and normalize
        for class_ in data:
            for i in range(len(data[class_])):
                data[class_][i] = np.transpose(data[class_][i], (2, 0, 1))

        print("Num classes " + str(len(data)))
        num_images = 0
        for class_ in data:
            num_images += len(data[class_])
        print("Num images " + str(num_images))
        
        return data, label_encoder
