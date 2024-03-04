"""
CMSC733 Spring 2024: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Yi-Chung Chen(ychen921@umd.edu) 
Master Student in Robotics,
University of Maryland, College Park
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math
import argparse
import cv2
import os


def DrawCorners(img, corners, SaveName):
    image = img.copy()
    for c in corners:
        x, y = c.ravel()
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(SaveName, bbox_inches='tight')


def CornerDetect(img, NumFeatures):
    """Detect corners in the image and draw the corners

    Args:
        image (numpy.ndarray): an input image
        NumFeatures (int): Maximum number of feature for corner detection
        filename (String): file name of the image

    Returns:
        BestN (numpy.ndarray): Best N corner coordinates of the image
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gray = np.float32(image_gray)

    corners_ = cv2.goodFeaturesToTrack(image_gray, NumFeatures, 0.001, 10)
    corners = np.int0(corners_)

    # for c in corners:
    #     x, y = c.ravel()
    #     cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    
    # plt.imshow(image)
    # plt.axis('off')
    # plt.savefig(SavePath+'/Corners_'+filename, bbox_inches='tight')
    return corners


def ANMS(image, corners):
    """Adaptive Non-Maximal Suppression

    Args:
        image (numpy.ndarray): Original Input image
        corners (numpy.ndarray): Detected coners coordinates of the image
    Return:    
        (numpy.ndarray): corner coordinates of the image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)    
    row, _, col = corners.shape
    corners = corners.reshape(row, col)
    NumOfCorners = row
    NumCornerBest = int(NumOfCorners*0.4)
    
    r = float('inf')*np.ones((NumOfCorners, 3))
    ED = float('inf')
    for i in range(NumOfCorners):
        for j in range(NumOfCorners):
            x_i = int(corners[i,0])
            y_i = int(corners[i,1])
            x_j = int(corners[j,0])
            y_j = int(corners[j,1]) 
            if (image_gray[y_j, x_j] > image_gray[y_i, x_i]):
                ED = np.square(x_j - x_i) + np.square(y_j - y_i)
            else:
                ED = float('inf')
                
            if (ED < r[i,0]):
                r[i,0] = ED
                r[i,1] = corners[j,0]
                r[i,2] = corners[j,1]
         
    sorted_indices = np.argsort(r[:,0])[::-1]
    sorted_r = r[sorted_indices].astype('uint8')
    BestN = sorted_r[:NumCornerBest, 1:3]
    
    
    return corners[:NumCornerBest, :] 

def FindDescriptor(image, corners):
    """Feature Descriptor

    Args:
        image (numpy.ndarray): An original RGB image
        corners (numpy.ndarray): A set of ANMS corners coordinates

    Returns:
        GoodCorners (list): corners that can find 40*40 patch
        Descriptors (list):
    """
    PatchSize = 40
    NumCorners = corners.shape[0]
    image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    GoodCorners = []
    Descriptors = []
    for i in range(NumCorners):
        x, y = corners[i,0], corners[i,1]
        # Crap a 40*40 region around the keypoint
        Patch = image_gray[y-int(PatchSize/2):y+int(PatchSize/2), x-int(PatchSize/2):x+int(PatchSize/2)]
        if Patch.shape == (40, 40):
            # Implement gaussian blur and down sample the patch from 40*40 to 8*8
            Patch = cv2.GaussianBlur(Patch, (3,3), 0)
            SubSample = cv2.resize(Patch, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
            
            # Reshape the down sample patch from 8*8 to 64*1
            SubSample = SubSample.reshape(64,1)
            
            # Standarlize
            desc = (SubSample - np.mean(SubSample)) / np.std(SubSample)
            GoodCorners.append([x,y])
            Descriptors.append(desc)
            
    return GoodCorners, Descriptors

def PlotDescriptors(desc, SavePath):
    num_rows=2
    num_cols=10
    figsize=(10, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax.imshow(desc[i].reshape(8,8), cmap='gray')
        ax.axis('off')
    plt.savefig(SavePath, bbox_inches='tight')


def FeatureMatch(corners1, corners2, des1, des2):
    """Feature Matching

    Args:
        corners1 (list): A set of corners of image 1
        corners2 (list): A set of corners of image 2
        des1 (list): A set of descriptors of image 1
        des2 (list): A set of descriptors of image 2

    Returns:
        list: 2 set of matching keypoints
    """
    kps1 = []
    kps2 = []
    for i in range(len(des1)):
        SSD = []
        for j in range(len(des2)):
           ssd = np.sum(np.square(des1[i] - des2[j]))
           SSD.append(ssd)
        
        idx1 = np.argsort(SSD)[0]
        idx2 = np.argsort(SSD)[1]
        
        if (SSD[idx1]/SSD[idx2]) < 0.7:
            kps1.append(corners1[i])
            kps2.append(corners2[idx1])
            
    return (kps1, kps2)

def DrawMatches(img1, img2, kps1, kps2, SaveName):
    """Draw Matching point between 2 images

    Args:
        img1 (numpy array): RGB image 1
        img2 (numpy array): RGB image 2
        kps1 (list): keypoints coordinates of image 1
        kps2 (list): keypoints coordinates of image 2
    """
    NumKps = len(kps1)
    MatchImage = np.concatenate((img1,img2), axis=1)
    for i in range(NumKps):
        x1, y1 = kps1[i][0], kps1[i][1]
        x2, y2 = kps2[i][0]+int(img1.shape[1]), kps2[i][1]
        cv2.line(MatchImage,(x1,y1),(x2,y2),(255,255,153),2)
        cv2.circle(MatchImage,(x1,y1),3,255,-1)
        cv2.circle(MatchImage,(x2,y2),3,255,-1)
    
    plt.imshow(MatchImage)
    plt.axis('off')
    plt.savefig(SaveName, bbox_inches='tight')
    # plt.show()


def compute_homography(pts):
    """Compute homography

    Args:
        pts (np arrau): two sets of points of two image

    Returns:
        H (numpy array): Computed homography by four random points
    """
    np.random.shuffle(pts)
    random_points = pts[:4, :]
    
    A = []
    match_num = random_points.shape[0]
        
    for i in range(match_num):
        pt_1 = random_points[i, 0:2]
        pt_2 = random_points[i, 2:4]
        
        sub_A = [0, 0, 0, pt_1[0], pt_1[1], 1, -pt_2[1]*pt_1[0], -pt_2[1]*pt_1[1], -pt_2[1]]
        sub_B = [pt_1[0], pt_1[1], 1, 0, 0, 0, -pt_2[0]*pt_1[0], -pt_2[0]*pt_1[1], -pt_2[0]]
        
        A.append(sub_A)
        A.append(sub_B)
    
    # Solve by SVD
    U, s, V = np.linalg.svd(np.array(A))
    H = V[-1].reshape(3, 3)
    H = (1 / H[-1, -1]) * H
    
    return H
    
def point_err(pts, H):
    """Compute the error of homography of each point

    Args:
        pts (np array): 2 sets of keypoints coordinates of two images.
        H (np array): Homography matrix.

    Returns:
        errs (np array): A list of errors of each estimated point.
    """
    
    points_num = pts.shape[0]
    add_z = np.ones((points_num, 1))
    
    pts_1 = np.column_stack((pts[:, 0:2], add_z))
    pts_2 = pts[:, 2:4]
    pt2_estimate = np.zeros((points_num, 2))
    
    for i in range(points_num):
        p2_dot = H @ pts_1[i]
        pt2_estimate[i] = ((1 / p2_dot[2])*p2_dot)[0:2]
    
    errs = pts_2 - pt2_estimate
    errs = np.linalg.norm(errs, axis=1)**2

    return errs
        
def inlier_num(err, threshold):
    count = 0
    indices = []
    for i in range(len(err)):
        if err[i] < threshold:
            count+=1
            indices.append(i)
    return count, indices

def RANSAC_Homography(pts, threshold):
    """Compute the Best homgraphy between 2 images
       This function is refered by Yi-Chung's previous project in ENPM673
    Args:
        pts (np array): 2 sets of keypoints coordinates of two images
        threshold (float): the threshold that determine the error points

    Returns:
         H (np array): Best homography
         Best Kps1 (np array): Best keypoints coordinates in image1
         Best Kps2(np array): Best keypoints coordinates in image2
    """
    
    max_inlier_count = 0
    iter = 10000
    iter_count = 0
    sample_num = 4
    row = pts.shape[0]
    while iter > iter_count:
        
        np.random.shuffle(pts)
        random_points = pts[:sample_num, :]
        H = compute_homography(random_points)
        errs = point_err(pts, H)
        inlier_count, indices_ = inlier_num(errs, threshold)
        
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_H = H
            indices = indices_
        iter_count += 1
    
    BestKps = np.zeros((len(indices), 4))
    for i ,idx in enumerate(indices):
        BestKps[i,:] = pts[idx,:]
    print("Best Inliers Number: ", max_inlier_count)
    return best_H , BestKps[:, 0:2].astype('int'), BestKps[:,2:4].astype('int')

def match_point_sets(kp_a, kp_b):
    pts_1 = np.array(kp_a)
    pts_2 = np.array(kp_b)
    pts = np.column_stack((pts_1, pts_2))
    return pts

def stitch(img1, img2, H):
    
    h1 ,w1 ,_ = img1.shape
    h2 ,w2 ,_ = img2.shape
    
    # Four corners coordinates of the first image after transformation
    CornerImg1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1,1,2)
    CornerImg1_T = cv2.perspectiveTransform(CornerImg1, H)
    
    CornerImg2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1,1,2)
    # print(CornerImg1_T.shape, CornerImg2.shape)
    NewImgCorners = np.concatenate((CornerImg1_T, CornerImg2), axis = 0)
    
    # Find the shape of the New image
    NewImgCorners = NewImgCorners.squeeze()
    xMax, yMax = np.round(np.amax(NewImgCorners, axis=0)).astype(int)
    xMin, yMin = np.round(np.amin(NewImgCorners, axis=0)).astype(int)
    
    # Add the translation info to the homography
    H_translate = np.array([[1, 0, -1*xMin], [0, 1, -1*yMin], [0, 0, 1]])
    H_ = H_translate @ H
    
    N_width = int(round(xMax - xMin))
    N_height = int(round(yMax - yMin))
    size = (N_width, N_height)
    Wraped = cv2.warpPerspective(img1, H_ , size)

    images_stitched = Wraped.copy()
    images_stitched[abs(yMin):abs(yMin)+h2, abs(xMin):abs(xMin)+w2] = img2
    
    return images_stitched

def CropImage(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    _, binary_mask = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[len(contours)-1])
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image

def Stitch1(images, NumFeatures, SavePath, saveflag=1):
    prime_image = images[0]
    count=0
    for i in range(len(images)):
        if np.array_equal(images[i], prime_image) is not True:
            img1 = prime_image
            img2 = images[i]
            # Implement Corner detection
            corners1 = CornerDetect(img1, NumFeatures)
            corners2 = CornerDetect(img2, NumFeatures)
            
            if saveflag == 1:
                DrawCorners(img1, corners1, SaveName=SavePath+'/Corners_'+str(count)+'_.png')
                DrawCorners(img2, corners2, SaveName=SavePath+'/Corners_'+str(count+1)+'_.png')
                
            # Implement ANMS to find the best corners
            NBest1 = ANMS(img1, corners1)
            NBest2 = ANMS(img2, corners2)
            if saveflag == 1:
                DrawCorners(img1, NBest1, SaveName=SavePath+'/ANMS_'+str(count)+'_.png')
                DrawCorners(img2, NBest2, SaveName=SavePath+'/ANMS_'+str(count+1)+'_.png')
                
            # Find descriptors from the keypoints
            corners1, des1 = FindDescriptor(img1, NBest1)
            corners2, des2 = FindDescriptor(img2, NBest2)
                
            # Feature matching and draw matching
            kps1, kps2 = FeatureMatch(corners1, corners2, des1, des2)
            if i == 1 and saveflag == 1:
                DrawMatches(img1, img2, kps1, kps2, SaveName=SavePath+'/NaiveMatch_'+str(count)+'.png')
                PlotDescriptors(desc=des1, SavePath=SavePath+'/Desc_'+str(count)+'.png')

            print(len(kps1))
            if len(kps1) < 20:
                print('Passing.... Two images may not overlap')
                continue
            
            # Compute homography by RANSAC and reject the outliers
            matches_set = match_point_sets(kps1, kps2)
            BestH, kps1, kps2 = RANSAC_Homography(matches_set, threshold=0.4)
            if i == 1 and saveflag == 1:
                DrawMatches(img1, img2, kps1, kps2, SaveName=SavePath+'/InliersMatch'+str(count)+'.png')
                
            # Stitch and blend two images
            StitchImg = stitch(img1, img2, BestH)
            StitchImg = CropImage(StitchImg)
            
            plt.imshow(StitchImg)
            plt.axis('off')
            plt.savefig(SavePath+'/Stitch_'+str(i)+'.png' , bbox_inches='tight')
            prime_image = StitchImg
            count+=2
    return prime_image
                
    
def Stitch2(images, NumFeatures, layer, SavePath):
    NumImage = len(images)
        
    stitch_img_A = []
    stitch_img_B = []
    count = 0
    for i in range(0, int(NumImage/2), 2):
        img1 = images[i]
        img2 = images[i+1]
            
            # Implement Corner detection
        corners1 = CornerDetect(img1, NumFeatures)
        corners2 = CornerDetect(img2, NumFeatures)
        if layer == 0:
            DrawCorners(img1, corners1, SaveName=SavePath+'/Corners_'+str(count)+'_.png')
            DrawCorners(img2, corners2, SaveName=SavePath+'/Corners_'+str(count+1)+'_.png')
        
        # Implement ANMS to find the best corners
        NBest1 = ANMS(img1, corners1)
        NBest2 = ANMS(img2, corners2)
        if layer == 0:
            DrawCorners(img1, NBest1, SaveName=SavePath+'/ANMS_'+str(count)+'_.png')
            DrawCorners(img2, NBest2, SaveName=SavePath+'/ANMS_'+str(count+1)+'_.png')
                    
        # Find descriptors from the keypoints
        corners1, des1 = FindDescriptor(img1, NBest1)
        corners2, des2 = FindDescriptor(img2, NBest2)

        kps1, kps2 = FeatureMatch(corners1, corners2, des1, des2)
        if layer == 0:
            DrawMatches(img1, img2, kps1, kps2, SaveName=SavePath+'/NaiveMatch_'+str(count)+'.png')
            # PlotDescriptors(desc=des1, SavePath=SavePath+'/Desc_'+str(count)+'.png')
        
        
        if len(kps1) < 20:
            print('Passing.... Two images may not overlap')
            continue
            
        matches_set = match_point_sets(kps1, kps2)
        BestH, kps1, kps2 = RANSAC_Homography(matches_set, threshold=0.4)
        if layer == 0:
            DrawMatches(img1, img2, kps1, kps2, SaveName=SavePath+'/InliersMatch'+str(count)+'.png')
            
        StitchImg = stitch(img1, img2, BestH)
        StitchImg = CropImage(StitchImg)
        stitch_img_A.append(StitchImg)
        plt.imshow(StitchImg)
        plt.axis('off')
        plt.savefig(SavePath+'/Stitch_tmp.png' , bbox_inches='tight')
        count += 2
        
        
    for i in range(NumImage-1, int(NumImage/2)-1, -2):
        img1 = images[i]
        img2 = images[i-1]
            
        # Implement Corner detection
        corners1 = CornerDetect(img1, NumFeatures)
        corners2 = CornerDetect(img2, NumFeatures)
        if layer == 0:
            DrawCorners(img1, corners1, SaveName=SavePath+'/Corners_'+str(count)+'_.png')
            DrawCorners(img2, corners2, SaveName=SavePath+'/Corners_'+str(count+1)+'_.png')
                
        # Implement ANMS to find the best corners
        NBest1 = ANMS(img1, corners1)
        NBest2 = ANMS(img2, corners2)
        if layer == 0:
            DrawCorners(img1, NBest1, SaveName=SavePath+'/ANMS_'+str(count)+'_.png')
            DrawCorners(img2, NBest2, SaveName=SavePath+'/ANMS_'+str(count+1)+'_.png')
                
        # Find descriptors from the keypoints
        corners1, des1 = FindDescriptor(img1, NBest1)
        corners2, des2 = FindDescriptor(img2, NBest2)

        kps1, kps2 = FeatureMatch(corners1, corners2, des1, des2)
        
        if layer == 0:
            DrawMatches(img1, img2, kps1, kps2, SaveName=SavePath+'/NaiveMatch_'+str(count)+'.png')
            # PlotDescriptors(desc=des1, SavePath=SavePath+'/Desc_'+str(count)+'.png')
        
        if len(kps1) < 20:
            print('Passing.... Two images may not overlap')
            continue
            
        matches_set = match_point_sets(kps1, kps2)
        BestH, kps1, kps2 = RANSAC_Homography(matches_set, threshold=0.4)
        if layer == 0:
            DrawMatches(img1, img2, kps1, kps2, SaveName=SavePath+'/InliersMatch'+str(count)+'.png')
            
        StitchImg = stitch(img1, img2, BestH)
        StitchImg = CropImage(StitchImg)
        stitch_img_B.append(StitchImg)
        count += 2
        plt.imshow(StitchImg)
        plt.axis('off')
        plt.savefig(SavePath+'/Stitch_tmp.png' , bbox_inches='tight')
        
    return stitch_img_A+stitch_img_B

 
def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/ychen921/733/MyAutoPano/Phase1', help='Path for saving the set of images, Default:/home/ychen921/733/MyAutoPano/Phase1')
    Parser.add_argument('--Dataset', default='Train', help='Path for saving the set of images, Default:Train')
    Parser.add_argument('--Set', default='Set1', help='Path for saving the set of images, Default:Set1')
    Parser.add_argument('--NumFeatures', default=1500, help='Number of best features to extract from each image, Default:100')
    
    Args = Parser.parse_args()
    BasePath = Args.BasePath
    DataSet = Args.Dataset
    Set = Args.Set
    NumFeatures = Args.NumFeatures

    # BasePath = 'C:/Users/steve/Desktop/733/Project1/MyAutoPano/Phase1'
    DataPath = BasePath + '/Data/'+str(DataSet)+'/'+str(Set)+'/'
    ResultPath = BasePath + '/Results'
    SavePath = ResultPath+'/'+str(Set)
    
    if not os.path.exists(ResultPath):
        os.mkdir(ResultPath)
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)

    images = []
    file_names = []
    
    for file_name in tqdm(os.listdir(DataPath)):
        # Read a set of images for image stitching        
        img = cv2.imread(DataPath+file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        file_names.append(file_name)
    
    # Stitch images if number of image < 3
    if (len(images) <= 3) or ((np.sqrt(len(images)).is_integer()) is not True) or (np.sqrt(len(images))%2!=0):
        print('Stitching <= 3 images')
        # Set the first image as the base image
        StitchImg = Stitch1(images, NumFeatures, SavePath)
    
    # Stitch images if number of image > 3
    if len(images) > 3 and np.sqrt(len(images)).is_integer() and (np.sqrt(len(images))%2==0):
        print('Stitching > 3 images')
        for i in range(int(np.sqrt(len(images)))):
            print("Layer: {}, Image Num: {}".format(i, len(images)))
            layer = i
            images = Stitch2(images, NumFeatures, layer, SavePath)
        print("Last layer, Image Num: {}".format(len(images)))
        images = Stitch1(images, NumFeatures, SavePath, saveflag=0)
        
    else:
        StitchImg = Stitch1(images, NumFeatures, SavePath)

if __name__ == '__main__':
    main()