#!/usr/bin/env python

"""
CMSC733 Spring 2023: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Yi-Chung Chen (ychen921@umd.edu)
Master in Robotics,
University of Maryland, College Park

Author(s):
Ji Liu (liuji@umd.edu)
Master in Robotics,
University of Maryland, College Park

"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import keras
import sys
import os
import matplotlib.pyplot as plt
from Network.Network import get_model, metric_dist
from Misc.tf_dataset import get_tf_dataset
import numpy as np
import argparse
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

def Mean_Corner_error(corners1, corners2):
    assert corners1.shape == corners2.shape
    distances = np.sum(np.abs(corners1 - corners2), axis=1)
    mean_error = np.mean(distances)
    return mean_error


def calculate_metric(ds, nimg, model, mode, BatchSize=8):
    corner_dist = []
    # go through all test images

    for i in tqdm(range(int(nimg/BatchSize))):
        # retrieve a sample batch
        sample_input, sample_output = next(iter(ds))

        if mode == "supervised":
            im_crop1, im_crop2 = sample_input
            h4pt = sample_output
            h4pt = h4pt.numpy().reshape((-1,4,2))

        elif mode == "unsupervised":        
            # im_crop1, im_crop2, _, _ = sample_input
            _, h4pt = sample_output
            h4pt = h4pt.numpy().reshape((-1,4,2))

        if mode == "supervised":
            h4pt_pred = model([im_crop1,im_crop2])
            h4pt_pred = (np.round(h4pt_pred.numpy())).reshape((-1,4,2))

        elif mode == "unsupervised":
            model_out_us = model(sample_input)
            im_warp_pred_us, h4pt_pred = model_out_us
            im_warp_pred_us = np.round(im_warp_pred_us.numpy()*255)
            h4pt_pred = np.round(h4pt_pred.numpy()).reshape((-1,4,2))

        for b in range(BatchSize):
            corner_dist.append(Mean_Corner_error(np.squeeze(h4pt[b,:,:]), np.squeeze(h4pt_pred[b,:,:])))
        
    corner_dist = np.array(corner_dist)

    return corner_dist


def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/ychen921/733/MyAutoPano/Phase2/Code/chkpt_weight/Supervised/cp_0050.ckpt', help='Path to load all check points from, Default:/home/ychen921/733/MyAutoPano/Phase2/Code/chkpt_weight/Supervised/cp_0100.ckpt')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/ychen921/733/Data/Test', help='Path to load images from, Default:/home/ychen921/733/Data/Test')
    Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup')
    Parser.add_argument('--MiniBatchSize', type=int, default=8, help='Size of the MiniBatch to use, Default:8')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    ModelType = Args.ModelType
    MiniBatchSize = Args.MiniBatchSize
    NumEpochs = Args.NumEpochs

    test_path = BasePath
    
    # Select model and model configuration
    if ModelType == "Sup":
        mode = "supervised"
    else:
        mode = "unsupervised"


    test_ds = get_tf_dataset(path=test_path, batch_size=MiniBatchSize, mode=mode)
    model = get_model(mode=mode)
    model.load_weights(ModelPath).expect_partial()

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


    corners_err = calculate_metric(test_ds, nimg=1000, model=model, mode=mode)
 

    print("#-------------------------------------#")
    print(f"Model error: mean {np.mean(corners_err):.3f}, "f"std {np.std(corners_err):.3f}")
    print("#-------------------------------------#")

if __name__ == '__main__':
    main()
 
