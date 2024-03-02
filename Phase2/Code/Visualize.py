import os
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw
from Network.Network import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

def Mean_Corner_error(corners1, corners2):
    assert corners1.shape == corners2.shape
    distances = np.sqrt(np.sum(np.square(corners1 - corners2), axis=1))
    mean_error = np.mean(distances)
    return mean_error

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Args = Parser.parse_args()
    ModelType = Args.ModelType
    
    if ModelType == "supervised":
        model = HomographyNet()
        model.load_weights('./chkpt_weight/checkpoint_mdl_v2_rho32')

    else:
        model = UnsupHomographyNet()



    resize_shape = (320,240)
    rho = 32

    test_path = "/home/ychen921/733/Data/Test"
    errors = []
    for i in os.listdir(test_path)[0:3]:
        print(test_path+'/'+i)

        im = Image.open(test_path+'/'+i)
        im = im.resize(resize_shape)
        im_ori = np.array(im)

        if im_ori.ndim < 3:
            continue
        
        h,w = im_ori.shape[:2]

        ch, cw = 128, 128

        upper_left_h = np.random.randint(low=rho, high=h-rho-ch)
        upper_left_w = np.random.randint(low=rho, high=w-rho-cw)
        # print(upper_left_h, upper_left_w)

        corner_pts = np.array([[0,0],[ch-1,0],[ch-1,cw-1],[0,cw-1]])+np.array([upper_left_h, upper_left_w])[np.newaxis,:]
        
        corner_pts_new = np.copy(corner_pts)
        for i in range(4):
                corner_pts_new[i,:] += np.random.randint(-rho, rho+1,size=(2,))
        
        # difference of two coordinates
        H4pt = (corner_pts - corner_pts_new).flatten()
        
        # Compute th ground truth homography
        H = cv2.getPerspectiveTransform(
                src=np.fliplr(corner_pts_new.astype(np.float32)),
                dst=np.fliplr(corner_pts.astype(np.float32)))
        
        im_warp = cv2.warpPerspective(im_ori, H, (w,h))

        p1 = im_ori[upper_left_h:upper_left_h+ch,upper_left_w:upper_left_w+cw]
        p2 = im_warp[upper_left_h:upper_left_h+ch,upper_left_w:upper_left_w+cw]

        Pred_H4pt = model.predict([p1[np.newaxis,:],p2[np.newaxis,:]])
        
        pred_corner_pts = np.round(corner_pts.flatten() - Pred_H4pt.flatten()).reshape(4,2).astype(int)
        
        error = Mean_Corner_error(corner_pts_new, pred_corner_pts)
        print("L2: {}".format(error))
        errors.append(error)
        
        im_ori_ = im_ori.copy()
        im_warp_ = im_warp.copy()

        corner_pts_ = corner_pts_new.copy()
        corner_pts_[:,[1,0]] = corner_pts_[:,[0,1]]
        pred_corner_pts_ = pred_corner_pts.copy()
        pred_corner_pts_[:,[1,0]] = pred_corner_pts_[:,[0,1]]

        cv2.line(im_ori_, (corner_pts_[0]), (corner_pts_[1]), color=(0, 0, 255), thickness=2)
        cv2.line(im_ori_, (corner_pts_[1]), (corner_pts_[2]), color=(0, 0, 255), thickness=2)
        cv2.line(im_ori_, (corner_pts_[2]), (corner_pts_[3]), color=(0, 0, 255), thickness=2)
        cv2.line(im_ori_, (corner_pts_[3]), (corner_pts_[0]), color=(0, 0, 255), thickness=2)

        cv2.line(im_ori_, (pred_corner_pts_[0]), (pred_corner_pts_[1]), color=(255, 0, 0), thickness=2)
        cv2.line(im_ori_, (pred_corner_pts_[1]), (pred_corner_pts_[2]), color=(255, 0, 0), thickness=2)
        cv2.line(im_ori_, (pred_corner_pts_[2]), (pred_corner_pts_[3]), color=(255, 0, 0), thickness=2)
        cv2.line(im_ori_, (pred_corner_pts_[3]), (pred_corner_pts_[0]), color=(255, 0, 0), thickness=2)

        plt.imshow(im_ori_)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    print(f"Average L2:{sum(errors)/len(errors)}")
if __name__ == '__main__':
    main()