'''
This script can be used to extract object from an image. The segmentation is performed by considering the HSV value of the image.
In the project, this script is used to extract the organ of interest.
'''


import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
import open3d as o3d
from sksurgerytorch.models import high_res_stereo
import skimage
import csv

#function to extract matrices (calibration) from text file (file.dat):
def extract_matrices(file_path):

    # Initilization of matrices:
    R = []
    T = []
    intrinsic = []
    distortion = []

    # Controll variables
    reading_R = False
    reading_T = False
    reading_distortion= False
    reading_intrinsic = False

    # Read file row by row
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # delete no used spaces

            # matrix selection
            if line.startswith('R:'): # Matrix R
                reading_R = True
                reading_T = False
                reading_distortion = False
                reading_intrinsic = False
                continue  # go ahead, skippng R line
            elif line.startswith('T:'): # Matrix T
                reading_T = True
                reading_R = False
                reading_intrinsic = False
                reading_distortion = False
                continue  # go ahead, skipping T line
            elif line.startswith('intrinsic:'): # Matrix 'intrinsic'
                reading_T = False
                reading_R = False
                reading_intrinsic = True
                reading_distortion = False
                continue
            elif line.startswith('distortion:'): # Matrix distortion
                reading_T = False
                reading_R = False
                reading_intrinsic = False
                reading_distortion = True
                continue

            # If R
            if reading_R:
                R.append(list(map(float, line.split())))

            # If T
            if reading_T:
                T.append(float(line))

            # If intrinsic
            if reading_intrinsic:
                intrinsic.append(list(map(float, line.split())))

            # If distortion
            if reading_distortion:
                distortion.extend(map(float, line.split()))

    # Convert matrices
    R = np.array(R)
    T = np.array(T)
    intrinsic = np.array(intrinsic)
    distortion = np.array(distortion)

    return R, T, intrinsic, distortion


def remove_outliers(disparity_map, colors):
    # Get rid of points with value 0 (no depth)
    mask_map = disparity_map > disparity_map.min()
    filtered_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    # Remove outlier
    z_values = filtered_points[:, 2]
    z_mean = np.mean(z_values)
    z_std = np.std(z_values)

    # Remove if > or < of 2 std
    lower_bound = z_mean - 2 * z_std
    upper_bound = z_mean + 2 * z_std
    final_mask = (filtered_points[:, 2] >= lower_bound) & (filtered_points[:, 2] <= upper_bound)
    output_points = filtered_points[final_mask]
    output_colors = output_colors[final_mask]
    return output_points, output_colors

# This function is used to select a ROI in the image. HSV parameters of the ROI are extracted
def select_roi(event, x, y, flags, param):
    global ix, iy, drawing,img, img_hsv, roi_hsv
    global min_h, max_h, min_s, max_s, min_v, max_v

    # Start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()  # image copy to update the drawing
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)  # draw the rectangle
            cv2.imshow("Select ROI", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:  # Finish the drawing
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", img)

        # Select ROI (HSV)
        roi_hsv = img_hsv[iy:y, ix:x]

        h, s, v = cv2.split(roi_hsv)
        min_h_roi = np.min(h)
        max_h_roi = np.max(h)
        min_s_roi = np.min(s)
        max_s_roi = np.max(s)
        min_v_roi = np.min(v)
        max_v_roi = np.max(v)

        min_h = min(min_h, min_h_roi)
        max_h = max(max_h, max_h_roi)
        min_s = min(min_s, min_s_roi)
        max_s = max(max_s, max_s_roi)
        min_v = min(min_v, min_v_roi)
        max_v = max(max_v, max_v_roi)

# From HSV extracted with function before, the image is processed and the segmnatation is obtained
def process_hsv_image(img_hsv, min_h, max_h, min_s, max_s, min_v, max_v, hole_size=30000, object_size=500,
                      filter_size=(20, 20), output_csv="mappa_filtrata_L.csv"):

    # Create the binary mask based on HSV thresholds
    low_L = (int(min_h), int(min_s), int(min_v))
    high_L = (int(max_h), int(max_s), int(max_v))
    mappa_L = cv2.inRange(img_hsv, low_L, high_L)

    # Process the mask: invert binary values (1 where mask was 0, and 0 otherwise)
    for i in range(len(mappa_L)):
        for j in range(len(mappa_L[i])):
            if int(mappa_L[i][j]) > 0:
                mappa_L[i][j] = 0
            else:
                mappa_L[i][j] = 1

    # Convert to boolean mask
    mappa_tot_L = mappa_L > 0

    # Apply morphological operations
    kernel = np.ones(filter_size)
    img_filtered_L = skimage.morphology.binary_closing(mappa_tot_L, kernel)
    img_filtered_L = skimage.morphology.binary_dilation(img_filtered_L, kernel)
    img_filtered_L = skimage.morphology.binary_closing(img_filtered_L, kernel)

    # Remove small holes and objects
    img_filtered_L = skimage.morphology.remove_small_holes(img_filtered_L, hole_size)
    img_filtered_L = skimage.morphology.remove_small_objects(img_filtered_L, object_size)

    # Save the filtered image as CSV
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(img_filtered_L.astype(int))  # Convert to int for CSV saving

    # Plot the original mask and the filtered result
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Binary Mask (mappa_L)")
    plt.imshow(mappa_L, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Filtered Image (img_filtered_L)")
    plt.imshow(img_filtered_L, cmap='gray')

    plt.show()

    return img_filtered_L


# Import paramters:
path_camera1_rot_trans='.../camera1_rot_trans.dat'
path_intrinsic_left='.../camera_parameters_intrinsics_left.dat'
path_intrinsic_right='.../camera_parameters_intrinsics_right.dat'

R, T, _, _ = extract_matrices(path_camera1_rot_trans)
_, _, Kl, dist_l = extract_matrices(path_intrinsic_left)
_, _, Kr, dist_r = extract_matrices(path_intrinsic_right)
h=1080
w=1920

# Elaborate camera parameters (compute Q for depth computation and maps for rectification)

# Import Images:
path_img_left='.../IMG_LEFT.png'
path_img_right='.../IMG_RIGHT.png'
img_left = cv2.imread(path_img_left)
img_right = cv2.imread(path_img_right)
image_size = (img_left.shape[1], img_left.shape[0])

# Image filtering:
kernel_size = (5, 5)
sigma = 0  # 0 means the sigma will be calculated based on the kernel size
img_left = cv2.GaussianBlur(img_left, kernel_size, sigma)
img_right = cv2.GaussianBlur(img_right, kernel_size, sigma)

# Rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1=Kl, cameraMatrix2=Kr,
                                                            distCoeffs1=dist_l, distCoeffs2=dist_r,
                                                            imageSize=(h, w),
                                                            R=R, T=T,
                                                            flags=cv2.CALIB_ZERO_DISPARITY,
                                                            alpha=0.5
                                                            )

# Map creation
map1_left, map2_left = cv2.initUndistortRectifyMap(Kl, dist_l, R1, P1, image_size, cv2.CV_32FC1)
map1_right, map2_right = cv2.initUndistortRectifyMap(Kr, dist_r, R2, P2, image_size, cv2.CV_32FC1)

rectified_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

# Save of rectified images
cv2.imwrite('rectified_left.jpg', rectified_left)
cv2.imwrite('rectified_right.jpg', rectified_right)

path_img_rect_left='.../rectified_left.jpg'
path_img_rect_right='.../rectified_right.jpg'

#Segmentation:

# HSV max and min value:
min_h, max_h = 179, 0
min_s, max_s = 255, 0
min_v, max_v = 255, 0

# Extract parameters
img = cv2.imread(path_img_rect_left)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
ix, iy = -1, -1
drawing = False
cv2.imshow("Select ROI", img)
cv2.setMouseCallback("Select ROI", select_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Segmentation from HSV parameters
img_filtered_L = process_hsv_image(img_hsv, min_h=min_h, max_h=max_h, min_s=min_s, max_s=max_s, min_v=min_v, max_v=max_v,
    hole_size=30000,
    object_size=500,
    filter_size=(20, 20),
    output_csv="mappa_filtrata_L.csv"
)


#Disparity map

# Using https://github.com/gengshan-y/high-res-stereo
HSMNet = high_res_stereo.HSMNet(max_disp=288,
                                entropy_threshold=-1,
                                level=1,
                                scale_factor=1,
                                weights='.../finetune_27999.tar')

# disparity in levels
disparity_hsm, _ = HSMNet.predict(rectified_left, rectified_right)

plt.imshow(disparity_hsm,'gray')
plt.show()

## 3D reconstruction:

baseline=1/abs(Q[3,2])
f=Q[2,3]

# Pass to cm
disparity_hsm_cm=(baseline*f)/disparity_hsm
disparity_map_segmented=disparity_hsm_cm*(img_filtered_L)

plt.show()
plt.imshow(disparity_map_segmented,'gray')
plt.show()

# Camera intrinsic parameters
fx = Kl[0, 0]  # Focal length in x
fy = Kl[1, 1]  # Focal length in y
cx = Kl[0, 2]   # Principal point x-coordinate
cy = Kl[1, 2]   # Principal point y-coordinate

# Depth map (already computed)
depth_map = disparity_map_segmented  # Depth map in real-world units

# Create 3D points from depth map
height, width = depth_map.shape
u, v = np.meshgrid(np.arange(width), np.arange(height))
X = (u - cx) * depth_map / fx
Y = (v - cy) * depth_map / fy
Z = depth_map

# visualization of points

# Data: x, y, z
x = X[X!=0]
y = Y[Y!=0]
z = Z[Z!=0]
xmean=np.mean(X[X!=0])
ymean=np.mean(Y[Y!=0])
zmean=np.mean(Z[Z!=0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add points:
ax.scatter(x, y, z, c='blue', marker='o')
ax.scatter(xmean, ymean, zmean, c='red', marker='o')

ax.set_xlabel('Asse X')
ax.set_ylabel('Asse Y')
ax.set_zlabel('Asse Z')
plt.show()

# Stack into points array
points_3D = np.stack((X, Y, Z), axis=-1)

# Save
np.savetxt('region_of_interest.txt', points_3D.reshape(-1, 3), delimiter=' ', fmt='%f')


