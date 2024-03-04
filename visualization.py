# imports
import numpy as np
import tensorflow as tf
import cv2
import argparse
import matplotlib.pyplot as plt
from Misc.tf_dataset import get_tf_dataset
from Network.Network import get_model
from Test import Mean_Corner_error


def plot_box(im, pts, color=1):
    if color == 1:
        c = (0,0,255)
    elif color == 2:
        c = (255,0,0)
    elif color == 3:
        c = (255,166,0)
    for i in range(4):
        cv2.line(im,
                 (np.flip(pts[i,:].astype(int))),
                 (np.flip(pts[int((i+1)%4)].astype(int))),
                 color=c, thickness=2)
        
def plot_result(h4pt_us,
                h4pt_s,
                im_ori,
                h4pt,
                upper_left,
                crop_sz=128,):
    ch = crop_sz
    cw = crop_sz
    h,w = im_ori.shape[1:3]
    B = im_ori.shape[0]

    plt.figure(figsize=(8,14))
    for b in range(B):
        upper_left_coord = upper_left[[b],:]
        corners = upper_left_coord + np.array([[0,0],
                                              [ch-1,0],
                                              [ch-1,cw-1],
                                              [0,cw-1]])
        
        corners_shift = corners - np.reshape(h4pt[[b],:],(4,2))
        corners_shift_pred_s = corners - np.reshape(h4pt_s[[b]],(4,2))
        corners_shift_pred_us = corners - np.reshape(h4pt_us[[b]],(4,2))

        error_s = Mean_Corner_error(corners_shift, corners_shift_pred_s)
        error_us = Mean_Corner_error(corners_shift, corners_shift_pred_us)

        im_ori_this = np.squeeze(im_ori[b,:,:,:])

        # plot the corners on the original image
        im_ori0 = im_ori_this.copy()
        plot_box(im_ori0, corners, color=1)

        # supervised performance
        im_ori1 = im_ori_this.copy()
        fs = 0.7
        c = (255,0,0)
        org = (185,30)
        cv2.putText(im_ori1, text=f"Error: {error_s:.2f}", org=org,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fs,
                    color=c,
                    thickness=2)
        plot_box(im_ori1, corners_shift, color=1)
        plot_box(im_ori1, corners_shift_pred_s, color=2)

        # unsupervised performance
        im_ori2 = im_ori_this.copy()
        cv2.putText(im_ori2, text=f"Error: {error_us:.2f}", org=org,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fs,
                    color=c,
                    thickness=2)
        plot_box(im_ori2, corners_shift, color=1)
        plot_box(im_ori2, corners_shift_pred_us, color=2)

        plt.subplot(B,1,b+1)
        plt.imshow(np.hstack((im_ori0, im_ori1, im_ori2)))
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"visualization.",dpi=300) 
    plt.show()

def plot_transform(h4pt_s,
                im_crop1,
                im_crop2,
                im_ori,
                h4pt,
                h4pt_us,
                upper_left,
                crop_sz=128,):
    ch = crop_sz
    cw = crop_sz
    h,w = im_ori.shape[1:3]
    B = im_ori.shape[0]

    plt.figure(figsize=(10,6))
    b = 0
    upper_left_coord = upper_left[[b],:]
    corners = upper_left_coord + np.array([[0,0],
                                            [ch-1,0],
                                            [ch-1,cw-1],
                                            [0,cw-1]])
    
    corners_shift = corners - np.reshape(h4pt[[b],:],(4,2))
    corners_shift_pred_s = corners - np.reshape(h4pt_s[[b]],(4,2))
    corners_shift_pred_us = corners - np.reshape(h4pt_us[[b]],(4,2))

    error_s = Mean_Corner_error(corners_shift, corners_shift_pred_s)
    error_us = Mean_Corner_error(corners_shift, corners_shift_pred_us)

    im_ori_this = np.squeeze(im_ori[b,:,:,:])

    # plot the corners on the original image
    im_ori0 = im_ori_this.copy()
    plot_box(im_ori0, corners, color=3) # plot in orange
    plot_box(im_ori0, corners_shift, color=1)

    plt.subplot(1,3,1)
    plt.imshow(im_ori0)
    plt.axis("off")

    plt.subplot(1,3,2)
    imc1,imc2 = np.squeeze(im_crop1[b,:,:,:]/255),\
                np.squeeze(im_crop2[b,:,:,:]/255)
    plt.imshow(np.hstack((imc1,np.ones((128,10,3)),imc2)))
    plt.axis("off")

    # supervised performance
    im_ori1 = im_ori_this.copy()
    fs = 0.7
    c = (255,0,0)
    org = (185,30)
    cv2.putText(im_ori1, text=f"Error: {error_s:.2f}", org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fs,
                color=c,
                thickness=2)
    plot_box(im_ori1, corners_shift, color=1)
    plot_box(im_ori1, corners_shift_pred_s, color=2)
    plt.subplot(1,3,3)
    plt.imshow(im_ori1)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"transform_illustration.",dpi=300) 
    plt.show()



def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--TestPath', default='/home/ychen921/733/project1/Phase2_Data/Test', help='Base path of images, Default:/home/ychen921/733/project1/Phase2_Data/Test')
    Parser.add_argument('--SupCheckPointPath', default='./chkpt_weight/cp_0100.ckpt', help='Path to save Checkpoints, Default: ../chkpt_weight/cp_0100.ckpt')
    Parser.add_argument('--UnsupCheckPointPath', default='./chkpt_weight/cp_0050.ckpt', help='Path to save Checkpoints, Default: ../chkpt_weight/cp_0050.ckpt')

    Args = Parser.parse_args()
    TestPath = Args.TestPath
    SupCheckPointPath = Args.SupCheckPointPath
    UnsupCheckPointPath = Args.UnsupCheckPointPath

    # Test Dataset
    ds = get_tf_dataset(path=TestPath, batch_size=8, mode="unsupervised")

    sample_input, sample_output = next(iter(ds))
    im_crop1, im_crop2, im_ori, upper_left_coord = sample_input
    im_warp, h4pt = sample_output

    # convert to numpy arrays
    im_crop1 = im_crop1.numpy()
    im_crop2 = im_crop2.numpy()
    im_ori = im_ori.numpy()
    upper_left_coord = upper_left_coord.numpy().astype(int)
    im_warp = im_warp.numpy()
    h4pt = h4pt.numpy().astype(int)

    # Load supervised model
    model_s = get_model(mode="supervised")
    model_s.load_weights(SupCheckPointPath).expect_partial()

    # Load unsupervised model
    model_us = get_model(mode="unsupervised")
    model_us.load_weights(UnsupCheckPointPath).expect_partial()

    # call model
    h4pt_s = model_s([im_crop1,im_crop2])
    h4pt_s = np.round(h4pt_s.numpy())

    model_out_us = model_us(sample_input)
    im_warp_pred_us, h4pt_us = model_out_us
    im_warp_pred_us = np.round(im_warp_pred_us.numpy()*255)
    h4pt_us = np.round(h4pt_us.numpy())

    plot_result(h4pt_s=h4pt_s, h4pt_us=h4pt_us,
                im_ori=im_ori, h4pt=h4pt,
                upper_left=upper_left_coord)
    
    plot_transform(im_crop1=im_crop1, im_crop2=im_crop2,
                   h4pt_s=h4pt_s, im_ori=im_ori,
                   h4pt=h4pt, h4pt_us=h4pt_us,
                   upper_left=upper_left_coord)

if __name__ == '__main__':
    main()