import sys
import cv2
import numpy as np

#PASTE THE DIM, K, AND D FROM CALIBRATE.PY HERE
DIM=(1280, 720)
K=np.array([[532.1431453101939, 0.0, 640.7764709543267], [0.0, 529.8409227573983, 360.7090734387717], [0.0, 0.0, 1.0]])
D=np.array([[-0.06054925466176916], [0.04371585904153406], [-0.03257994726108615], [0.006837798919737519]])


def undistort(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(1280,720))
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    print(dim1,DIM)
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

undistort('test.png',balance=1)
