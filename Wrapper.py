#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import tensorflow as tf
import keras
import sys
import os
import matplotlib.pyplot as plt
from Network.Network import get_model, metric_dist
from Misc.tf_dataset import get_tf_dataset
import numpy as np
import argparse
import cv2
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
# Add any python libraries here



def main():
    
	# Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/ychen921/733/MyAutoPano/Phase2/Code/chkpt_weight/Supervised/cp_0050.ckpt', help='Path to load all check points from, Default:/home/ychen921/733/MyAutoPano/Phase2/Code/chkpt_weight/Supervised/cp_0100.ckpt')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/ychen921/733/MyAutoPano/Phase1/Data/Train/CustomSet2', help='Path to load images from, Default:/home/ychen921/733/MyAutoPano/Phase1/Data/Train/CustomSet2')
    Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Sup')
    Parser.add_argument('--MiniBatchSize', type=int, default=8, help='Size of the MiniBatch to use, Default:8')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    ModelType = Args.ModelType
    MiniBatchSize = Args.MiniBatchSize

    test_path = BasePath
    
    # Select model and model configuration
    if ModelType == "Sup":
        mode = "supervised"
    else:
        mode = "unsupervised"


    # test_ds = get_tf_dataset(path=test_path, batch_size=MiniBatchSize, mode=mode)
    model = get_model(mode=mode)
    model.load_weights(ModelPath).expect_partial()
    
    img1 = cv2.imread(BasePath+"/1.jpg")
    img2 = cv2.imread(BasePath+"/2.jpg")
    
    img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    img2 = tf.convert_to_tensor(img2, dtype=tf.float32)

    # if mode == "supervised":
    #     model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    #                     loss=keras.losses.MeanSquaredError(name="mse_loss"),
    #                     metrics=[keras.losses.MeanAbsoluteError(name="mae"),
    #                             metric_dist])
    # else:
    #     model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3,
    #                                                   clipvalue=0.01),
    #                                                   run_eagerly=True)
        
	
    

    
if __name__ == '__main__':
    main()
 
