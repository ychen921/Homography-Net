import tensorflow as tf
import cv2
import sys
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import *
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Misc.DataGenerator import DataGenerator
import numpy as np
import time
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

def config_ds(ds, MiniBatchSize):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.batch(MiniBatchSize)
    return ds

def main():
    BasePath = "/home/ychen921/733/Data"
    im_crop_shape = (128, 128, 3)
    mode = "unsupervised_with_h4pt"
    im_ori_shape = (240, 320, 3)
    output_signature=(  #input
                (tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32),
                tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32),
                tf.TensorSpec(shape=im_ori_shape,dtype=tf.float32),
                tf.TensorSpec(shape=(2,),dtype=tf.float32),
                ),
                    # output
                (tf.TensorSpec(shape=im_ori_shape,dtype=tf.float32), 
                    tf.TensorSpec(shape=(8,),dtype=tf.float32))
                )
    model = UnsupHomographyNet(BatchSize=8)
    
    
    
    
    # mode = "supervised"
    # output_signature = ((tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32), 
    #                     tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32)),
    #                 tf.TensorSpec(shape = (8,),dtype=tf.float32))
    # model = HomographyNet()
    
    
    
    
    checkpoint_filepath = './chkpt_weight/Unsupervised/'
    checkpoint_path = os.path.join(checkpoint_filepath, "cp_{epoch:04d}.ckpt")

    TrainDataLoader = DataGenerator(BasePath+'/Train', mode=mode)
    ValidDataLoader = DataGenerator(BasePath+'/Val', mode=mode)

    # Data Loader
    train_ds = tf.data.Dataset.from_generator(TrainDataLoader, output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(ValidDataLoader, output_signature=output_signature)

    train_ds = config_ds(train_ds, 8)
    val_ds = config_ds(val_ds, 8)

    sample_input, sample_output = next(iter(train_ds))
    print("input shapes:")
    for i in sample_input:
        print(i.shape)
    print("output shape")
    for i in sample_output:
        print(i.shape)

    sample_model_output = model(sample_input)

    print(sample_output[0])
    print(tf.round(sample_model_output[0]))

    plt.imshow(np.hstack(((sample_output[0][2,:,:,:]).numpy(), (sample_model_output[0][2,:,:,:]).numpy())))
    plt.show()

    model.summary()
    

if __name__ == '__main__':
    main()
 
