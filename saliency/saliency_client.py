import os

import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow
import numpy as np

import random
import csv
import matplotlib.pyplot as plt

import requests
import urllib.request
import tempfile
import zipfile
import hashlib
import h5py
import json
import imageio

# import pydicom
# import magic


## Implementation of the client goes here for now to simplify working on it
class SaliencyClient:
    labels = []
    files = []
    keys = []
    studies = []
    
    # Initialize with login and password -- we need these to download the CSV
    def __init__(self, login, password):
        self.login = login
        self.password = password
        
    # Downloads images from a csv
    def download_images(self, csv_reader, held_out_study = None):
        os.makedirs("tmp/images/",exist_ok=True)
    
        self.n = 0
        for row in csv_reader:
            
#             if (row["study"]==held_out_study):
#                 continue

            self.n += 1
            
            _, filename = os.path.split(row["file"])
            path = "tmp/images/%s" % filename
            
            if not os.path.isfile(path):
                urllib.request.urlretrieve(row["file"], path)
                
            self.labels.append(row["label"])
            self.files.append(path)
            self.studies.append(row["study"])
            
#            if self.n > MAX_IMAGES:
#                break
                  
    # Downloads a list of images (csv) and calls download_images with that csv
    def get_dataset(self, held_out_study = None, index_csv = "tmp/monsterx.csv"):
        self.dataset_descriptor = index_csv
        
        if not os.path.isfile(self.dataset_descriptor):
            response = requests.get("https://api.saliency.ai/projections/export/", auth=(self.login, self.password))

            # Throw an error for bad status codes
            response.raise_for_status()

            with open(self.dataset_descriptor, 'wb') as handle:
                for block in response.iter_content(1024):
                    handle.write(block)
        
        with open(self.dataset_descriptor, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            self.download_images(csv_reader, held_out_study = held_out_study)
    
        
    # Splits ids into train, validation, test
    def prepare_dataset_held_out(self, split = (80,10,10), held_out_study = None, class_weight_power = 1, print_labels = False):
#         self.labels = self.studies
        self.labels=[self.studies[i]+"_"+self.labels[i] for i in np.arange(len(self.labels))]
        
#         held_out_index = np.array(self.studies) == held_out_study
        held_out_index = np.isin(np.array(self.studies), held_out_study)
        n_held_out = np.sum(held_out_index)
#         remaining_index = np.array(self.studies) != held_out_study
        remaining_index = np.invert(held_out_index)
#         n_remaining = self.n - n_held_out

        self.seq = {}
        self.seq["held_out"] = np.argwhere(held_out_index)
        self.seq["held_out"] = np.array([i[0] for i in self.seq["held_out"]])
        self.seq["train"] = []
        self.seq["val"] = []
        self.seq["test"] = []
        
        # Distribute each class evenly between datasets
        self.held_out_labels = np.array(self.labels)[held_out_index]
        self.held_out_files = np.array(self.files)[held_out_index]
        
        self.remaining_labels = np.array(self.labels)[remaining_index]
        self.remaining_files = np.array(self.files)[remaining_index]
        
        if print_labels == True:
            print("remaining labels:")
            print(np.unique(self.remaining_labels))
            print()
            print()
            print("heldout labels:")
            print(np.unique(self.held_out_labels))
        
        for lab in np.unique(self.remaining_labels):
#             print(lab)
            indices = np.argwhere(np.array(self.remaining_labels) == lab)
            indices = np.array([a[0] for a in indices])                
            
            lab_ntrain = int(len(indices) * (split[0] / 100.0))
            lab_nval = int(len(indices) * (split[1] / 100.0))
            lab_ntest = len(indices) - lab_nval - lab_ntrain          
            
            self.seq["train"].append(np.array(indices[0:lab_ntrain]))
            self.seq["val"].append(np.array(indices[lab_ntrain:(lab_ntrain+lab_nval)]))
            self.seq["test"].append(np.array(indices[(lab_ntrain+lab_nval):(lab_ntrain + lab_nval + lab_ntest)]))
        
        self.seq["train"] = np.array(np.concatenate(self.seq["train"], axis = 0))
        self.seq["val"] = np.concatenate(self.seq["val"], axis = 0) 
        self.seq["test"] = np.concatenate(self.seq["test"], axis = 0) 

        
        ntrain = len(self.seq["train"])
        nval = len(self.seq["val"])
        ntest = len(self.seq["test"])
        
        # Process labels
        self.keys = dict([(b,a) for a,b in enumerate(list(set(self.remaining_labels)))])
        self.label_dict = dict([(a,b) for a,b in enumerate(list(set(self.remaining_labels)))])
        self.Y = np.array([self.keys[lab] for lab in self.remaining_labels])
        
        self.keys_held_out = dict([(b,a) for a,b in enumerate(list(set(self.held_out_labels)))])
        self.Y_held_out = np.array([self.keys_held_out[lab] for lab in self.held_out_labels])
        
        # Calculate class weights for weighted loss functions
        self.calculate_class_weights(power = class_weight_power)
        
    def load_image(self, filename):
#         return misc.imread(filename)
        return imageio.imread(filename)
        
#     def get_batch(self, index, split, batch_size, augmentations = []):
#         sq = self.seq[split]
    
#         frm = index*batch_size
#         to = min((index+1)*batch_size,len(sq))
#         ids = sq[frm:to]

#         Xtmp = None
#         dims = None

#         for i,id in enumerate(ids):
#             augmented = self.load_image(self.files[id])
#             for augmentation in augmentations:
#                 augmented = augmentation(augmented)

#             if not dims:
#                 dims = augmented.shape
#                 Xtmp = np.zeros((to - frm, ) + dims)

#             Xtmp[i,:,:,:] = augmented
            
#         X = Xtmp
#         if split == 'held_out':
#             Y = keras.utils.to_categorical(self.Y_held_out[ids], num_classes = len(np.unique(self.Y_held_out)))
#         else:
#             Y = keras.utils.to_categorical(self.Y[ids], num_classes = len(np.unique(self.Y)))


#         return X, Y, np.array([1]*len(ids))
    
    def get_batch_held_out(self, index, split, batch_size, channels_first, augmentations=[]):
        'helper function only to be used with the keras data generator...NOT the pytoch generator'
        sq = self.seq[split]
    
        frm = index*batch_size
        to = min((index+1)*batch_size,len(sq))
        ids_list = sq[frm:to]

        Xtmp = None
        dims = None

        for i,ids in enumerate(ids_list):
            augmented = self.load_image(self.remaining_files[ids])
            for augmentation in augmentations:
                augmented = augmentation(augmented, 
                                         channels_first = channels_first, 
                                         output_width = self.output_width,
                                         output_height = self.output_height,
                                         output_channels = self.output_channels)

            
            if not dims:
                dims = augmented.shape
                Xtmp = np.zeros((to - frm, ) + dims)

            Xtmp[i,:,:,:] = augmented
            
        X = Xtmp
        if split == 'held_out':
            Y = tensorflow.keras.utils.to_categorical(self.Y_held_out[ids_list], num_classes = len(np.unique(self.Y_held_out)))
        else:
            Y = tensorflow.keras.utils.to_categorical(self.Y[ids_list], num_classes = len(np.unique(self.Y)))
        
        sample_weights = np.array([self.class_weight_dict[i] for i in self.Y[ids_list]])
        return X, Y, sample_weights
        
        
    def get_num_steps(self, split, batch_size):
        sq = self.seq[split]
        
        return 1+int((len(sq)-1)/float(batch_size))
    
    def get_generator(self, 
                      split, 
                      task_type, 
                      output_height,
                      output_width,
                      output_channels,
                      batch_size = 8, 
                      augmentations = [], 
                      class_weight_power = 1, 
                      framework = 'keras'):
        
        self.framework = framework
        self.output_height = output_height
        self.output_width = output_width
        self.output_channels = output_channels
        self.task_type = task_type
        
        if self.framework == 'pytorch':
            self.channels_first = True
        if self.framework == 'keras':
            self.channels_first = False
                        
        if self.framework == 'keras':
            if self.task_type != 'single_label_classification':
                print('We do not support '+ self.task_type + 'in keras currently')
                return
            
            class SaliencyDataGenerator(tensorflow.keras.utils.Sequence):
                'Generates data for Keras'
                def __init__(self, sapi, split, batch_size):
                    'Initialization'
                    self.sapi = sapi
                    self.split = split
                    self.batch_size = batch_size
                    self.on_epoch_end()
                    

                def __len__(self):
                    'Denotes the number of batches per epoch'
                    return self.sapi.get_num_steps(self.split, self.batch_size)

                def __getitem__(self, index):
                    return self.sapi.get_batch_held_out(index, self.split, self.batch_size, augmentations = augmentations, channels_first = self.sapi.channels_first)
    #                 return self.sapi.get_batch(index, self.split, self.batch_size, augmentations = augmentations)

                def on_epoch_end(self):
                    'Updates indexes after each epoch'
                    random.shuffle(self.sapi.seq[self.split])
                    
                def see_one_batch(self):
                    example = self[0]
                    X = example[0]
                    Y = example[1]
                    Weights = example[2]

                    print("Shape of X for one batch:", X.shape)
                    print("Shape of Y for one batch:", Y.shape)
                    print("Shape of W for one batch:", Weights.shape)

                    for j in np.arange(len(X)):
                        plt.imshow(X[j,:,:,0], cmap = 'bone')
                        plt.colorbar()
                        plt.title(self.sapi.label_dict[np.where(Y[j]==1)[0][0]])
                        plt.show()
                        print("Y for one image:")
                        print(Y[j])
                        print()
                        print("Weight for one image:")
                        print(Weights[j])

            return SaliencyDataGenerator(self, split, batch_size)
        
        if self.framework == 'pytorch':
            
            class SaliencyDataset(Dataset):
                'Generates data for Pytorch'
                def __init__(self, sapi, split):
                    'Initialization'
                    self.sapi = sapi
                    self.split = split
                
                def __len__(self):
                    'Denotes the number of SAMPLES per epoch (differs from keras which denotes BATCHES/epoch)'
                    return len(self.sapi.seq[split])
                
                def __getitem__(self, index):
                    # Select sample
                    ID = self.sapi.seq[self.split][index] # ID is the location of the item in sapi.remaining_files or sapi.held_out_files

                    'Load data and get label'
                
                    'Note: the target for pytorch should be expressed as an int denoting the index of the true class'
                    '(This differs from Keras, which expects the target to be expressed as a one-hot vector)'
                    if split == 'held_out':
                        X = self.sapi.load_image(self.sapi.held_out_files[ID])
                        y = self.sapi.Y_held_out[ID]
                        
                    else:
                        X = self.sapi.load_image(self.sapi.remaining_files[ID])
                        y = self.sapi.Y[ID]
    
                    for augmentation in augmentations:
                        X = augmentation(X, 
                                         channels_first=self.sapi.channels_first, 
                                         output_width = self.sapi.output_width,
                                         output_height = self.sapi.output_height,
                                         output_channels = self.sapi.output_channels)
                        
                    return X,y
                
            class SaliencyDataLoader(DataLoader):
                def __init__(self, dataset, batch_size=1, 
                             shuffle=False, sampler=None,batch_sampler=None, 
                             num_workers=0, collate_fn=None,pin_memory=False, 
                             drop_last=False, timeout=0, worker_init_fn=None):
                    
                    super().__init__(dataset, batch_size=batch_size, 
                                     shuffle=shuffle, sampler=sampler,batch_sampler=batch_sampler, 
                                     num_workers=num_workers, collate_fn=collate_fn,pin_memory=pin_memory, 
                                     drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)
                    self.dataset = dataset
                    
                def see_one_batch(self):
                    example = iter(self).next()
                    X = example[0]
                    Y = example[1]

                    print("Shape of X:", X.shape)
                    print("Shape of Y:", Y.shape)

                    for j in np.arange(len(X)):
                        plt.imshow(X[j,0,:,:], cmap = 'bone')
                        plt.colorbar()
                        plt.title(self.dataset.sapi.label_dict[Y[j].item()])
                        plt.show()
                        print("Y for one image:", Y[j].item())
                        print()
    
        dataset =  SaliencyDataset(self, split)
        dataloader = SaliencyDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        return dataloader
    
    def get_num_classes(self):
        return len(self.keys)
    
    
    def calculate_class_weights(self, power=1):
        labels, counts = np.unique(self.Y, return_counts = True)
        label_percent = counts/np.sum(counts)
        self.class_weight_dict = {}
        
        for i in np.arange(len(labels)):
            if power == None:
                self.class_weight_dict[labels[i]] = 1.0
            else:
                self.class_weight_dict[labels[i]] = 1.0/(label_percent[i]**power)
    
        class_weight_list = [self.class_weight_dict[i] for i in np.arange(len(self.class_weight_dict))]
        self.class_weight_tensor = torch.FloatTensor(class_weight_list)
            
            
    def get_class_weights(self, framework = 'keras'):
        'Keras expects weights to be given as a dictionary'
        'Pytorch expects weights to be given as a tensor'

        if self.framework == 'keras':
            return self.class_weight_dict
        
        if self.framework == 'pytorch':
            return self.class_weight_tensor
