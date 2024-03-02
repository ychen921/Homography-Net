
"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

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

# Don't generate pyc codes
sys.dont_write_bytecode = True

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              
  
            
def config_ds(ds, MiniBatchSize):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.batch(MiniBatchSize)
    return ds

# L2 Loss
def custom_loss(y_true, y_pred):
    return 0.5*tf.reduce_mean(tf.square(y_true-y_pred))

def photometric_loss(y_true, y_pred):
    # y_true and y_pred are now batch of images
    return tf.reduce_mean(tf.abs(y_true-y_pred)) 
    # notice 1-norm is used

def metric_abs(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true-y_pred))



def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/ychen921/733/Data', help='Base path of images, Default:/home/ychen921/733/Project1/Data/Train')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=8, help='Size of the MiniBatch to use, Default:128')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType
    
    
    if not os.path.exists(CheckPointPath):
        os.mkdir(CheckPointPath)

    if not os.path.exists("../Results"):
        os.mkdir("../Results")

    im_crop_shape = (128, 128, 3)

    # Select model and model configuration
    if ModelType == "Sup":
        mode = "supervised"
        output_signature = ((tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32), 
                         tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32)),
                        tf.TensorSpec(shape = (8,),dtype=tf.float32))
        model = HomographyNet()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=custom_loss,)
        checkpoint_filepath = './chkpt_weight/Supervised/'
        checkpoint_path = os.path.join(checkpoint_filepath, "cp_{epoch:04d}.ckpt")

    else:
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
        model = UnsupHomographyNet(BatchSize=MiniBatchSize)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipvalue=0.01),run_eagerly=True)
        checkpoint_filepath = './chkpt_weight/Unsupervised/'
        checkpoint_path = os.path.join(checkpoint_filepath, "cp_{epoch:04d}.ckpt")

    TrainDataLoader = DataGenerator(BasePath+'/Train', mode=mode)
    ValidDataLoader = DataGenerator(BasePath+'/Val', mode=mode)
    
    # Data Loader
    train_ds = tf.data.Dataset.from_generator(TrainDataLoader, output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(ValidDataLoader, output_signature=output_signature)

    train_ds = config_ds(train_ds, MiniBatchSize)
    val_ds = config_ds(val_ds, MiniBatchSize)

    steps_per_epoch = int(np.floor(5000/MiniBatchSize))

    # reduce learning rate when performance plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                    factor=0.2,
                                                    patience=3,
                                                    min_lr=1e-6,
                                                    verbose=1,
                                                    cooldown=3)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    monitor='loss',
                                                    mode='min',
                                                    save_freq='epoch',
                                                    #save_best_only=True,
                                                    verbose=True)
    
    history = model.fit(train_ds,
                        epochs=NumEpochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_ds,
                        validation_steps=int(np.floor(1000/MiniBatchSize)),
                        validation_freq=1,
                        verbose=True,
                        callbacks=[reduce_lr, checkpoint_callback])
       
        


    plt.plot(history.history['loss'])
    plt.plot(history.history["val_loss"])
    plt.legend(["train","validation"])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False)
    plt.savefig('../Results/Loss_Epochs.png' , bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
 
