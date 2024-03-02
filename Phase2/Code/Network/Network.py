"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


import sys
import numpy as np
from Misc.TFSpatialTransformer import transformer
import tensorflow as tf
from keras.applications.vgg19 import VGG19, preprocess_input
from Misc.tensor_dlt import TensorDLT
from Misc.spatial_transformer import spatial_transformer_network

# Don't generate pyc codes
sys.dont_write_bytecode = True

class InceptionBlock(tf.keras.Model):
    def __init__(self, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32):
        super(InceptionBlock, self).__init__()
        
        self.conv_1x1 = tf.keras.layers.Conv2D(filters=filters_1x1, kernel_size=(1,1), padding='same', activation='relu')
        
        self.conv3x3_reduce = tf.keras.layers.Conv2D(filters=filters_3x3_reduce, kernel_size=(1,1), padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=filters_3x3, kernel_size=(3,3), padding='same', activation='relu')
        
        self.conv5x5_reduce = tf.keras.layers.Conv2D(filters=filters_5x5_reduce, kernel_size=(1,1), padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.Conv2D(filters=filters_5x5, kernel_size=(5,5), padding='same', activation='relu')
        
        self.MaxPool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.MaxPool_reduce = tf.keras.layers.Conv2D(filters=filters_pool_proj, kernel_size=(1,1), padding='same', activation='relu')
        
    def call(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv3x3(self.conv3x3_reduce(x))
        x3 = self.conv5x5(self.conv5x5_reduce(x))
        x4 = self.MaxPool_reduce(self.MaxPool(x))
        
        output = tf.keras.layers.Concatenate(axis=3)([x1,x2,x3,x4])
        return output
        

class HomographyNet(tf.keras.Model):
    def __init__(self, num_blocks=3):
        super(HomographyNet, self).__init__()
        
        self.base_model = tf.keras.applications.vgg19.VGG19(include_top=False, 
                                                       weights='imagenet', 
                                                       input_shape=(128, 128, 3),
                                                       pooling='max')
        self.base_model.trainable = False
        input = self.base_model.input
        output = self.base_model.get_layer('block3_conv4').output
        
        self.base_model2 = tf.keras.Model(inputs=input, outputs=output)
        self.base_model2.trainable = False
        
        self.Inception_model = self._make_blocks(3)
        
        self.fc = tf.keras.Sequential([
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=512, activation='relu',  kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=512, activation='relu', kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=8, activation=None,  kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01))
                                      ])
        
    def _make_blocks(self, num_blocks):
        blocks = tf.keras.Sequential()
        for _ in range(num_blocks):
            blocks.add(InceptionBlock())
            blocks.add(tf.keras.layers.BatchNormalization())
        return blocks
        
    def call(self, inputs):
       
        input1, input2 = inputs
        x1 = preprocess_input(input1)
        x2 = preprocess_input(input2)
        x1 = self.base_model2(x1)
        x2 = self.base_model2(x2)
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])
        x = self.Inception_model(x)
        x = self.fc(x)
        
        return x
    

class UnsupHomographyNet(tf.keras.Model):
    def __init__(self, BatchSize):
        super(UnsupHomographyNet, self).__init__()
        self.rho = 32
        self.BatchSize = BatchSize

        self.homography_net = HomographyNet(num_blocks=3)

        self.loss_tracker = tf.keras.metrics.MeanAbsoluteError(name='loss')
        self.metric_h4pt = tf.keras.metrics.MeanAbsoluteError(name='mae_h4pt')

        self.loss_tracker_val = tf.keras.metrics.MeanAbsoluteError(name='val_loss')
        self.metric_h4pt_val = tf.keras.metrics.MeanAbsoluteError(name='val_mae_h4pt')


    def call(self, inputs):
        p1, p2, im_ori, upper_left_corner = inputs
        h4pt_batch = self.homography_net([p1,p2])
   
        h4pt_batch = tf.clip_by_value(h4pt_batch, 
                                      clip_value_min=-self.rho,
                                      clip_value_max=self.rho)
        
        homography = TensorDLT(h4pt_batch=h4pt_batch, 
                               upper_left_corner=upper_left_corner, 
                               batch_size=self.BatchSize)
        # img_pred, _ = transformer(im_ori, homography, (240, 320))
        img_pred = spatial_transformer_network(im_ori, homography, img_height=240, img_width=320)

        return img_pred, h4pt_batch
    
    def train_step(self, data):
        inputs, targets = data

        with tf.GradientTape() as tape:
            pred = self(inputs, )
            loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(targets[0], pred[0]))

        gradients = tape.gradient(loss, self.trainable_variables)
        
        skip = False
        for g in gradients:
            try:
                tf.debugging.check_numerics(g, message='Checking grad')
            except Exception as e:
                tf.print("==================== nan found ====================",)
                skip = True
                break

        if skip is False:
            # do gradient update
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.loss_tracker.update_state(targets[0], pred[0])
            self.metric_h4pt.update_state(targets[-1], pred[-1])

        return {"loss": self.loss_tracker.result(), "mae_h4pt": self.metric_h4pt.result()}
    
    def test_step(self, data):
        data_in, data_out = data
        model_out = self(data_in, training=False)

        self.loss_tracker_val.update_state(data_out[0], model_out[0])
        self.metric_h4pt_val.update_state(data_out[-1], model_out[-1])

        return {"val_loss": self.loss_tracker_val.result(), "val_mae_h4pt": self.metric_h4pt_val.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.metric_h4pt]
