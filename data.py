# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:08:56 2025

@author: duminil
"""
from tensorflow.keras.utils import Sequence
import numpy as np 
import os
import pandas as pd
import tensforflow as tf 
from tf.keras.preprocessing.image import load_img
from tf.keras.preprocessing.image import img_to_array

class ImageGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_path,
                 batch_size, 
                 model,
                 shape_img, 
                 transformation=False,
                 shuffle = True
                 ):
        'Initialization'
        self.list_path = list_path
        # self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.shape_img = shape_img
        self.model = model
        self.transformation = transformation
        self.shuffle = shuffle
        
        self.on_epoch_end()
        
        self.dataA = []
        self.dataB = []
        self.dataC = []
        
        self.__list_all_files()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataA) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_a = self.dataA[index*self.batch_size:(index+1)*self.batch_size]
        indexes_b = self.dataB[index*self.batch_size:(index+1)*self.batch_size]
    
          
        A = np.array(indexes_a)
        B = np.array(indexes_b)  

        
        return A, B
        
    def __list_all_files(self):
        

        tmp_df = pd.read_csv(self.list_path)
        liste_de_frames = tmp_df.values.tolist()
        # counter = 0
       
        for i in liste_de_frames :
            print('i', i)
            # i_0 = i[0].replace("E:", "D:")
            # i_1 = i[1].replace("E:", "D:")
            
            i_0 = i[0]
            i_1 = i[1]
            # i_2 = i[2]
            
            img_base_list = os.listdir(i_0)
            sketch_base_list = os.listdir(i_1)

            
            for j, k in zip(img_base_list, sketch_base_list):
                
                a = load_img(i_0 + j, target_size=(self.shape_img[1], self.shape_img[2], 3))
                b = load_img(i_1 + k, color_mode='grayscale', target_size=(self.shape_img[1], self.shape_img[2], 1))

                a = img_to_array(a)
                b = img_to_array(b)

                #normalization
                an = tf.cast(a, tf.float32) / 127.5 - 1
                
                bn = tf.cast(b, tf.float32) / 127.5 - 1
                
                self.dataA.append(an)
                self.dataB.append(bn)

                        
                        