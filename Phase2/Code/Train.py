import tensorflow as tf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from Network.Network import get_model, metric_dist
from Misc.tf_dataset import get_tf_dataset
import argparse

# Don't generate pyc codes
sys.dont_write_bytecode = True
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# L2 Loss
def custom_loss(y_true, y_pred):
    return 0.5*tf.reduce_mean(tf.square(y_true-y_pred))

def photometric_loss(y_true, y_pred):
    # y_true and y_pred are now batch of images
    return tf.reduce_mean(tf.abs(y_true-y_pred)) 
    # notice 1-norm is used

def metric_abs(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true-y_pred))


def main(NumEpochs,
         BasePath,
         DivTrain,
         MiniBatchSize,
         LoadCheckPoint,
         CheckPointPath,
         LogsPath,
         ModelType):

    if not os.path.exists(CheckPointPath):
        os.mkdir(CheckPointPath)

    if not os.path.exists("../Results"):
        os.mkdir("../Results")

    # Select model and model configuration
    if ModelType == "Sup":
        mode = "supervised"
        batch_size = 32
        monitor_name = "mse_loss"
    else:
        mode = "unsupervised"
        batch_size = 8
        monitor_name = "mae_loss"
    checkpoint_path = f"./chkpt/mdl_{mode}"

    # set up tensorflow database
    train_path = BasePath+'/Train'
    val_path = BasePath+'/Val'

    train_ds = get_tf_dataset(path=train_path, batch_size=batch_size,mode=mode)
    val_ds = get_tf_dataset(path=val_path, batch_size=batch_size,mode=mode)

    # set up model
    model = get_model(mode=mode)
    if mode == "supervised":
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss=keras.losses.MeanSquaredError(name="mse_loss"),
                      metrics=[keras.losses.MeanAbsoluteError(name="mae"),
                               metric_dist])
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3,
                                                      clipvalue=0.01),
                    run_eagerly=True)

    steps_per_epoch = int(np.floor(5000/batch_size))

    # reduce learning rate when performance plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor_name,
                                                  factor=0.2,
                                                  patience=3,
                                                  min_lr=1e-6,
                                                  verbose=1,
                                                  cooldown=3)
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                                    filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    monitor=monitor_name,
                                                    mode='min',
                                                    # save_freq='epoch',
                                                    save_best_only=True,
                                                    verbose=True)
    
    history = model.fit(train_ds,
                        epochs=NumEpochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_ds,
                        validation_steps=int(np.floor(1000/MiniBatchSize)),
                        validation_freq=1,
                        verbose=True,
                        callbacks=[reduce_lr, checkpoint_callback])
       
    # plt.plot(history.history['loss'])
    # plt.plot(history.history[loss_type])
    # plt.legend(["train","validation"])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # for pos in ['right', 'top']: 
    #     plt.gca().spines[pos].set_visible(False)
    # plt.savefig('../Results/Loss_Epochs.png' , bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/ji/Dropbox/Robotics/CMSC733/Project1/Phase2/Data',
                        help='Base path of images, Default:/home/ychen921/733/Project1/Data')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=8, help='Size of the MiniBatch to use, Default:8')
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

    main(NumEpochs,
         BasePath,
         DivTrain,
         MiniBatchSize,
         LoadCheckPoint,
         CheckPointPath,
         LogsPath,
         ModelType)
 
