
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
import keras
import sys
import os
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import get_model, metric_dist
from Misc.DataGenerator import DataGenerator
from Misc.tf_dataset import get_tf_dataset
import numpy as np
import argparse
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def PrettyPrint(NumEpochs, MiniBatchSize, NumTrainSamples=5000):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
                 
  
            
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
    Parser.add_argument('--BasePath', default='/home/ychen921/733/Data', help='Base path of images, Default:/home/ychen921/733/Project1/Data')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--MiniBatchSize', type=int, default=8, help='Size of the MiniBatch to use, Default:8')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    CheckPointPath = Args.CheckPointPath
    ModelType = Args.ModelType
    
    PrettyPrint(NumEpochs, MiniBatchSize)

    if not os.path.exists("../Results"):
        os.mkdir("../Results")

    im_crop_shape = (128, 128, 3)

    # Select model and model configuration
    if ModelType == "Sup":
        mode = "supervised"
        monitor_name = "mse_loss"
        checkpoint_filepath = './chkpt_weight/Supervised/'
        checkpoint_path = os.path.join(checkpoint_filepath, "cp_{epoch:04d}.ckpt")
        y_min, y_max = 100, 290

    else:
        mode = "unsupervised"
        monitor_name = "mae_loss"
        checkpoint_filepath = './chkpt_weight/Unsupervised/'
        checkpoint_path = os.path.join(checkpoint_filepath, "cp_{epoch:04d}.ckpt")
        y_min, y_max = 0.1, 0.25

    train_path = BasePath+'/Train'
    val_path = BasePath+'/Val'
    
    # Data Loader
    train_ds = get_tf_dataset(path=train_path, batch_size=MiniBatchSize, mode=mode)
    val_ds = get_tf_dataset(path=val_path, batch_size=MiniBatchSize, mode=mode)

    model = get_model(mode=mode)

    if mode == "supervised":
        train_loss_name = 'loss'
        val_loss_name = "val_loss"
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss=keras.losses.MeanSquaredError(name="mse_loss"),
                      metrics=[keras.losses.MeanAbsoluteError(name="mae"),
                               metric_dist])
    else:
        train_loss_name = 'mae_loss'
        val_loss_name = "val_val_mae_loss"
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3,
                                                      clipvalue=0.01),
                                                      run_eagerly=True)


    steps_per_epoch = int(np.floor(5000/MiniBatchSize))

    # reduce learning rate when performance plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_name,
                                                    factor=0.2,
                                                    patience=3,
                                                    min_lr=1e-6,
                                                    verbose=1,
                                                    cooldown=3)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    monitor=monitor_name,
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
       

    plt.plot(history.history[train_loss_name])
    plt.plot(history.history[val_loss_name])
    plt.ylim(y_min, y_max) 
    plt.legend(["train","validation"])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False)
    plt.savefig('../Results/Loss_Epochs.png' , bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
 
