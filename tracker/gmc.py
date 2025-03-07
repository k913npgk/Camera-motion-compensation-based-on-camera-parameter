import cmath
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class GMC:
    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        super(GMC, self).__init__()
        self.count = 0
        self.blur_score_total = 0
        self.blur_score_not_blur = 0
        self.prev_H_skip = []
        self.last_call = 0
        self.method = method
        self.check = False
        self.downscale = max(1, int(downscale))
        self.prev_diff = []
        self. x_diff_list = []
        self. y_diff_list = []
        self. z_diff_list = []
        self.angular_list = []
        self.frame_list = []

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)
            # self.gmc_file = open('GMC_results.txt', 'w')
        elif self.method == 'new':
            self.point = []
            self.H = [[1, 0, 0], [0, 1, 0]]
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)
            print("NEW")

        elif self.method == 'file' or self.method == 'files':
            seqName = verbose[0]
            ablation = verbose[1]
            if ablation:
                filePath = r'tracker/GMC_files/MOT17_ablation'
            else:
                filePath = r'tracker/GMC_files/MOTChallenge'

            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]

            self.gmcFile = open(filePath + "/GMC-" + seqName + ".txt", 'r')

            if self.gmcFile is None:
                raise ValueError("Error: Unable to open GMC file in directory:" + filePath)
        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'
        else:
            raise ValueError("Error: Unknown CMC method:" + method)

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    def apply(self, raw_frame, camera_parameter, prev_camera_parameter, next_camera_parameter, detections=None):
        if self.method == 'orb' or self.method == 'sift':
            return self.applyFeaures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)
        elif self.method == 'new':
            height, width, _ = raw_frame.shape
            if self.point == []:
                # for i in detections:
                #     self.point.append([i[0], i[1]])
                #     self.point.append([i[0], (i[3] - i[1])/2])
                #     self.point.append([i[2], (i[3] - i[1])/2])
                #     self.point.append([i[0], i[3]])
                #     self.point.append([i[2], i[1]])
                #     self.point.append([(i[2] - i[0])/2, i[1]])
                #     self.point.append([(i[2] - i[0])/2, i[3]])
                #     self.point.append([i[2], i[3]])
                for i in range(height):
                    if i != 0 and i % 100 == 0:
                        for j in range(width):
                            if j != 0 and j % 100 == 0:
                                self.point.append([j, i])
            return self.applyNew(raw_frame, camera_parameter, prev_camera_parameter, next_camera_parameter, detections)
        elif self.method == 'none':
            return np.eye(2, 3), [0, 0], False
        else:
            return np.eye(2, 3)

    def applyEcc(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Run the ECC algorithm. The results are stored in warp_matrix.
        # (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria)
        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except:
            print('Warning: find transform failed. Set warp as identity')

        return H

    def applyFeaures(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # find the keypoints
        mask = np.zeros_like(frame)
        # mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                   prevKeyPointLocation[1] - currKeyPointLocation[1])

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                        (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Draw the keypoint matches on the output image
        if 0:
            matches_img = np.hstack((self.prevFrame, frame))
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.prevFrame, 1)
            for m in goodMatches:
                prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))

                matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
                matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)

            plt.figure()
            plt.imshow(matches_img)
            plt.show()

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySparseOptFlow(self, raw_frame, detections=None):

        t0 = time.time()
        self.count += 1
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H, [[], []], False

        # find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            # print(self.count)
            # print(prevPoints[0])
            # print(currPoints[0])
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
            # print(H[0][0:2])
            # print(H[1][0:2])
            # print([H[0][2], H[1][2]])
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        t1 = time.time()

        # gmc_line = str(1000 * (t1 - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)
        # print(H)
        # p = input()
        return H, [prevPoints, currPoints], False

    def applyFile(self, raw_frame, detections=None):
        line = self.gmcFile.readline()
        tokens = line.split("\t")
        H = np.eye(2, 3, dtype=np.float_)
        H[0, 0] = float(tokens[1])
        H[0, 1] = float(tokens[2])
        H[0, 2] = float(tokens[3])
        H[1, 0] = float(tokens[4])
        H[1, 1] = float(tokens[5])
        H[1, 2] = float(tokens[6])

        return H
    
    def applyNew(self, raw_frame, camera_parameter, prev_camera_parameter, next_camera_parameter, detections=None):
        t0 = time.time()
        self.count += 1
        skip_check = False
        # print(self.count)
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        self.fix_point = [935, 504]
        # Handle first frame
        # Downscale image
        # if self.downscale > 1.0:
        #     frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            # frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            # frame = cv2.resize(frame, (width, height))
        
        H = np.eye(2, 3)
        # blur_score = cv2.Laplacian(frame, cv2.CV_64F).var()
        # print(self.count)
        # print(blur_score)
        # if self.count <= 100:
        #     if self.count not in [12, 21, 29, 45, 46, 47, 54, 63, 66, 69, 70, 72, 79, 88, 95, 96, 97, 98]:
        #         self.blur_score_not_blur += blur_score
        #     self.blur_score_total += blur_score
        # if self.count == 100:
        #     print(self.blur_score_total/100)
        #     print(self.blur_score_not_blur/82)
        # if blur_score < 0:
        #     skip_check = True
        #     self.check = True
        #     curr = [[[0,0]],[[0,0]]]
        #     return H, curr, skip_check
        # if self.check == True:
        #     # prev_frame = cv2.resize(self.frame_keep, (width // self.downscale, height // self.downscale))
        #     # curr_frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        #     # prev_feature = cv2.goodFeaturesToTrack(self.frame_keep, mask=None, **self.feature_params)
        #     # curr_feature = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        #     sift = cv2.SIFT_create()
        #     kp1, des1 = sift.detectAndCompute(self.frame_keep, None)
        #     kp2, des2 = sift.detectAndCompute(frame, None)
        #     FLANN_INDEX_KDTREE = 1
        #     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        #     search_params = dict(checks=50)
        #     flann = cv2.FlannBasedMatcher(index_params, search_params)
        #     matches = flann.knnMatch(des1, des2, k=2)
        #     matches_mask = [[0, 0] for _ in range(len(matches))]
        #     good_match = []
        #     for i, (m, n) in enumerate(matches):
        #         if m.distance < 0.5 * n.distance:
        #             good_match.append(m)
        #             matches_mask[i] = [1, 0]
        #     src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

        #     H, inliesrs = cv2.estimateAffinePartial2D(src_pts.astype(np.float32), dst_pts.astype(np.float32), cv2.RANSAC)
        #     # Handle downscale
        #     # if self.downscale > 1.0:
        #     #     H[0, 2] *= self.downscale
        #     #     H[1, 2] *= self.downscale
        #     skip_check = False
        #     self.check = False
        #     curr = [[[0,0]],[[0,0]]]
        #     self.frame_keep = frame
        #     return H, curr, skip_check
        self.frame_keep = frame
        # feature_params_check = dict(maxCorners=100000, qualityLevel=0.25, minDistance=1, blockSize=3,
        #                                useHarrisDetector=False, k=0.04)
        # prev_keypoints_check = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params_check)
        # print(len(prev_keypoints_check))
        if not self.initializedFirstFrame:
            # self.prev_keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
            
            # for i in detections:
            #     point.append([i[0], i[1]])
            #     point.append([i[0], (i[3] - i[1])/2])
            #     point.append([i[2], (i[3] - i[1])/2])
            #     point.append([i[0], i[3]])
            #     point.append([i[2], i[1]])
            #     point.append([(i[2] - i[0])/2, i[1]])
            #     point.append([(i[2] - i[0])/2, i[3]])
            #     point.append([i[2], i[3]])
            
                    
            curr = [[[0,0]],[[0,0]]]
            self.fix_list = []
            
            self.fix_list.append(self.fix_point)
            self.prev_H = [[0, 0, 0], [0, 0, 0]]
            # for i in self.prev_keypoints:
            #     temp = i[0]
            #     # if temp[0] <= 1920:
            #     #     if temp[1] <= 1080:
            #     point.append(temp)
            self.prev_keypoints = self.point
            # self.prev_currpoint = np.array(point).astype(np.float32)
            self.prevFrame = frame.copy()
            # Initialize data
            self.prev_detections = detections
            # Initialization done
            self.initializedFirstFrame = True
            rotation = [0, 0, 0]
            return H, rotation, skip_check
        # 2
        # point = []
        # for i in self.prev_keypoints:
        #     if 0 <= i[0] <= 1920:
        #         if 0 <= i[1] <= 1080:
        #             point.append(i)
        # self.prev_keypoints = point
        # if len(self.prev_keypoints) <= 8:
        #     self.prev_keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        #     point = []
        #     for i in self.prev_keypoints:
        #         temp = i[0]
        #         if temp[0] <= 1920:
        #             if temp[1] <= 1080:
        #                 point.append(temp)
        #     self.prev_keypoints = point
        # 2
        # camera_parameter = [[1 - 2 * (camera_parameter[1] * camera_parameter[1] + camera_parameter[2] * camera_parameter[2]), 2 * (camera_parameter[0] * camera_parameter[1] - camera_parameter[3] * camera_parameter[2]), 2 * (camera_parameter[3] * camera_parameter[1] + camera_parameter[0] * camera_parameter[2]), camera_parameter[4]],
        # [2 * (camera_parameter[0] * camera_parameter[1] + camera_parameter[3] * camera_parameter[2]), 1 - 2 * (camera_parameter[0] * camera_parameter[0] + camera_parameter[2] * camera_parameter[2]), 2 * (camera_parameter[1] * camera_parameter[2] - camera_parameter[3] * camera_parameter[0]), camera_parameter[5]],
        # [2 * (camera_parameter[0] * camera_parameter[2] - camera_parameter[3] * camera_parameter[1]), 2 * (camera_parameter[1] * camera_parameter[2] + camera_parameter[3] * camera_parameter[0]), 1 - 2 * (camera_parameter[0] * camera_parameter[0] + camera_parameter[1] * camera_parameter[1]), camera_parameter[6]]]
        # prev_camera_parameter = [[1 - 2 * (prev_camera_parameter[1] * prev_camera_parameter[1] + prev_camera_parameter[2] * prev_camera_parameter[2]), 2 * (prev_camera_parameter[0] * prev_camera_parameter[1] - prev_camera_parameter[3] * prev_camera_parameter[2]), 2 * (prev_camera_parameter[3] * prev_camera_parameter[1] + prev_camera_parameter[0] * prev_camera_parameter[2]), prev_camera_parameter[4]],
        # [2 * (prev_camera_parameter[0] * prev_camera_parameter[1] + prev_camera_parameter[3] * prev_camera_parameter[2]), 1 - 2 * (prev_camera_parameter[0] * prev_camera_parameter[0] + prev_camera_parameter[2] * prev_camera_parameter[2]), 2 * (prev_camera_parameter[1] * prev_camera_parameter[2] - prev_camera_parameter[3] * prev_camera_parameter[0]), prev_camera_parameter[5]],
        # [2 * (prev_camera_parameter[0] * prev_camera_parameter[2] - prev_camera_parameter[3] * prev_camera_parameter[1]), 2 * (prev_camera_parameter[1] * prev_camera_parameter[2] + prev_camera_parameter[3] * prev_camera_parameter[0]), 1 - 2 * (prev_camera_parameter[0] * prev_camera_parameter[0] + prev_camera_parameter[1] * prev_camera_parameter[1]), prev_camera_parameter[6]]]
        focal_matrix, camera_parameter = camera_parameter[0], camera_parameter[1]
        prev_focal_matrix, prev_camera_parameter = prev_camera_parameter[0], prev_camera_parameter[1]
        prev_angle_z = math.atan2(prev_camera_parameter[1][0], prev_camera_parameter[0][0])
        prev_angle_y = math.atan2(-prev_camera_parameter[2][0], math.sqrt(math.pow(prev_camera_parameter[2][0], 2)+math.pow(prev_camera_parameter[2][2], 2)))
        prev_angle_x = math.atan2(prev_camera_parameter[2][1], prev_camera_parameter[2][2])
        angle_z = math.atan2(camera_parameter[1][0], camera_parameter[0][0])
        angle_y = math.atan2(-camera_parameter[2][0], math.sqrt(math.pow(camera_parameter[2][0], 2)+math.pow(camera_parameter[2][2], 2)))
        angle_x = math.atan2(camera_parameter[2][1], camera_parameter[2][2])
        x_diff = angle_x - prev_angle_x
        y_diff = angle_y - prev_angle_y
        z_diff = angle_z - prev_angle_z
        # print([prev_angle_x, prev_angle_y, prev_angle_z])
        # print([angle_x, angle_y, angle_z])
        height, width, _ = raw_frame.shape
        # frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)
        prevFrameMatrix_r = [[prev_camera_parameter[0][0], prev_camera_parameter[0][1], prev_camera_parameter[0][2]],[prev_camera_parameter[1][0], prev_camera_parameter[1][1], prev_camera_parameter[1][2]],[prev_camera_parameter[2][0], prev_camera_parameter[2][1], prev_camera_parameter[2][2]]]
        # prevFrameMatrix_r = [[math.cos(prev_angle_y), 0, math.sin(prev_angle_y)],[0, 1, 0],[-math.sin(prev_angle_y), 0, math.cos(prev_angle_y)]]
        # prevFrameMatrix_r = [[prev_camera_parameter[0][0], prev_camera_parameter[0][1]],[prev_camera_parameter[1][0], prev_camera_parameter[1][1]],[prev_camera_parameter[2][0], prev_camera_parameter[2][1]]]
        # prevFrameMatrix_r = [[math.cos(prev_angle_z), -math.sin(prev_angle_z)], [math.sin(prev_angle_z), math.cos(prev_angle_z)]]
        # prevFrameMatrix_t = [prev_camera_parameter[0][3], prev_camera_parameter[1][3]]
        prevFrameMatrix_t = [prev_camera_parameter[0][3], prev_camera_parameter[1][3], prev_camera_parameter[2][3]]
        frameMatrix_r = [[camera_parameter[0][0], camera_parameter[0][1], camera_parameter[0][2]],[camera_parameter[1][0], camera_parameter[1][1], camera_parameter[1][2]],[camera_parameter[2][0], camera_parameter[2][1], camera_parameter[2][2]]]
        # frameMatrix_r = [[math.cos(angle_y), 0, math.sin(angle_y)],[0, 1, 0],[-math.sin(angle_y), 0, math.cos(angle_y)]]
        # frameMatrix_r = [[camera_parameter[0][0], camera_parameter[0][1]],[camera_parameter[1][0], camera_parameter[1][1]],[camera_parameter[2][0], camera_parameter[2][1]]]
        # frameMatrix_r = [[math.cos(angle_z), -math.sin(angle_z)], [math.sin(angle_z), math.cos(angle_z)]]
        # frameMatrix_t = [camera_parameter[0][3], camera_parameter[1][3]]
        frameMatrix_t = [camera_parameter[0][3], camera_parameter[1][3], camera_parameter[2][3]]
        # focal_matrix = [[487.20352, 0], [0, 487.1574]]
        # focal_matrix_x = [[1670, 0, 960],[0, 1670, 540],[0, 0, 1]]
        # focal_matrix_3 = [[1670, 0, 960],[0, 1670, 540]]
        # [488.25812, 488.65976, 325.7842, 236.43318] #20230720_150914
        # [488.25812, 488.65976, 325.7842, 236.43318] #20230721_131448、img11、img14
        # prev_focal_matrix = [[488.25812, 0, 325.7842], [0, 488.65976, 236.43318], [0, 0, 1]]
        # focal_matrix = [[488.25812, 0, 325.7842], [0, 488.65976, 236.43318], [0, 0, 1]]
        
        # compensation_matrix = [math.sin(angle_y - prev_angle_y)/math.cos(angle_y - prev_angle_y), math.sin(angle_x - prev_angle_x)/math.cos(angle_x - prev_angle_x), 0]
        # compensation_matrix = np.dot(focal_matrix, compensation_matrix)
        # prevFrameMatrix_r = np.dot(focal_matrix_x, prevFrameMatrix_r)
        # frameMatrix_r = np.dot(focal_matrix_x, frameMatrix_r)
        # prevFrameMatrix_t = np.dot(focal_matrix_x, prevFrameMatrix_t)
        # prevFrameMatrix_t = [prevFrameMatrix_t[0]/prevFrameMatrix_t[2], prevFrameMatrix_t[1]/prevFrameMatrix_t[2]]
        # frameMatrix_t = np.dot(focal_matrix_x, frameMatrix_t)
        # frameMatrix_t = [frameMatrix_t[0]/frameMatrix_t[2], frameMatrix_t[1]/frameMatrix_t[2]]
        # prevFrameMatrix_t = [(focal_matrix[0][0]*prevFrameMatrix_t[0]/prevFrameMatrix_t[2]), (focal_matrix[1][1]*prevFrameMatrix_t[1]/prevFrameMatrix_t[2])]
        # frameMatrix_t = [(focal_matrix[0][0]*frameMatrix_t[0]/frameMatrix_t[2]), (focal_matrix[1][1]*frameMatrix_t[1]/frameMatrix_t[2])]
        # Downscale image
        # if self.downscale > 1.0:
        #     frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
        #     frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        # 前後幀矩陣變化
        
        # rotate_c = [[rotate_c[0][0], rotate_c[0][1]],[rotate_c[1][0], rotate_c[1][1]],[rotate_c[2][0], rotate_c[2][1]]]
        
        # 1
        compensation_matrix = [2*focal_matrix[0][0]*math.sin(((angle_y - prev_angle_y)/2))+(frameMatrix_t[0] - prevFrameMatrix_t[0]), 2*focal_matrix[1][1]*math.sin(((angle_x - prev_angle_x)/2))+(frameMatrix_t[1] - prevFrameMatrix_t[1]), 0]
        compensation_r = [[math.cos(angle_z - prev_angle_z), -math.sin(angle_z - prev_angle_z)], [math.sin(angle_z - prev_angle_z), math.cos(angle_z - prev_angle_z)]]
        # rotate_c = np.dot(np.linalg.inv(prevFrameMatrix_r), frameMatrix_r)
        # prevFrameMatrix_t = np.dot(focal_matrix, prevFrameMatrix_t)
        # frameMatrix_t = np.dot(focal_matrix, frameMatrix_t)
        # translation_c = [[frameMatrix_t[0] - prevFrameMatrix_t[0]], [frameMatrix_t[1] - prevFrameMatrix_t[1]]]
        # translation_cx = [[(translation_c[0][0] + compensation_matrix[0])], [(translation_c[1][0] + compensation_matrix[1])]]
        # fuse_matrix = np.concatenate((rotate_c, translation_cx), axis=1)
        # H = fuse_matrix
        # print(compensation_matrix)
        # print(translation_c)
        # 1
        
        # 3
        # prevFrameMatrix_r = np.dot(prev_focal_matrix, prevFrameMatrix_r)
        # frameMatrix_r = np.dot(focal_matrix, frameMatrix_r)
        # prevFrameMatrix_t = np.dot(prev_focal_matrix, prevFrameMatrix_t)
        # frameMatrix_t = np.dot(focal_matrix, frameMatrix_t)
        # rotate_c = np.dot(np.linalg.inv(frameMatrix_r), prevFrameMatrix_r)
        # rotate_c = [[rotate_c[0][0], rotate_c[0][1]], [rotate_c[1][0], rotate_c[1][1]]]
        # translation_c = [[prevFrameMatrix_t[0] - frameMatrix_t[0]], [prevFrameMatrix_t[1] - frameMatrix_t[1]]]
        # fuse_matrix = np.concatenate((rotate_c, translation_c), axis=1)
        # H = fuse_matrix
        # 3
        
        # print(translation_c)
        # print(compensation_matrix)
        # translation_c = np.dot(focal_matrix, translation_c)
        # rotate_c = np.dot(focal_matrix, rotate_c)
        # frameMatrix_t = np.dot(focal_matrix, frameMatrix_t)
        # det_box = [1, 1, 1, 1, 1, 1]
        # def tlbr_to_tlwh(tlbr):
        #     ret = np.asarray(tlbr).copy()
        #     ret[2:] -= ret[:2]
        #     return ret
        # det_box = [tlbr_to_tlwh(tlbr) for tlbr in self.prev_detections][0]
        # print(det_box)
        # det_box = [det_box[0], det_box[1], 1, det_box[2], det_box[3], 1]
        # x = 640/495
        # y = 480/495
        # cx = 330
        # cy = 237
        # x = range(0, 30)
        # y = range(0, 30)
        # z = range(0, 1)
        # X, Y, Z= np.meshgrid(x, y, z)
        # X1=X.flatten()
        # Y1=Y.flatten()
        # Z1=Z.flatten()
        # V=(Z1,X1,Y1)
        # V = np.transpose(V)
        # bn_point = [cx, cy, 495, 1]
        # bw_point = [cx-x, cy-y, 495, 1]
        # bs_point = [cx-x, cy, 495, 1]
        # be_point = [cx, cy-y, 495, 1]
        # tn_point = [400, 250, 495, 1]
        # tw_point = [500, 350, 495, 1]
        # ts_point = [550, 400, 495, 1]
        # te_point = [650, 500, 495, 1]
        # 5
        # focal_matrix = [[focal_matrix[0][0], focal_matrix[0][1], focal_matrix[0][2], 0], [focal_matrix[1][0], focal_matrix[1][1], focal_matrix[1][2], 0] ,[focal_matrix[2][0], focal_matrix[2][1], focal_matrix[2][2], 0]]
        # currFrameMatrix = [[frameMatrix_r[0][0], frameMatrix_r[0][1], frameMatrix_r[0][2], frameMatrix_t[0]], [frameMatrix_r[1][0], frameMatrix_r[1][1], frameMatrix_r[1][2], frameMatrix_t[1]], [frameMatrix_r[2][0], frameMatrix_r[2][1], frameMatrix_r[2][2], frameMatrix_t[2]], [0, 0, 0, 1]]
        # currTrans = np.dot(focal_matrix, currFrameMatrix)
        # world_point = []
        # for i in self.prev_keypoints:
        #     temp = np.dot(np.linalg.pinv(focal_matrix), [i[0], i[1], 1])
        #     temp = np.dot(np.linalg.pinv(currFrameMatrix), temp)
        #     # temp = [temp[0]/temp[3], temp[1]/temp[3], temp[2]/temp[3], temp[3]/temp[3]]
        #     world_point.append(temp)
        # 5
        # 2
        prevFrameMatrix = [[prevFrameMatrix_r[0][0], prevFrameMatrix_r[0][1], prevFrameMatrix_r[0][2], prevFrameMatrix_t[0]], [prevFrameMatrix_r[1][0], prevFrameMatrix_r[1][1], prevFrameMatrix_r[1][2], prevFrameMatrix_t[1]], [prevFrameMatrix_r[2][0], prevFrameMatrix_r[2][1], prevFrameMatrix_r[2][2], prevFrameMatrix_t[2]], [0, 0, 0, 1]]
        prev_focal_matrix = [[prev_focal_matrix[0][0], prev_focal_matrix[0][1], prev_focal_matrix[0][2], 0], [prev_focal_matrix[1][0], prev_focal_matrix[1][1], prev_focal_matrix[1][2], 0], [prev_focal_matrix[2][0], prev_focal_matrix[2][1], prev_focal_matrix[2][2], 0]]
        prevTrans = np.dot(prev_focal_matrix, prevFrameMatrix)
        prevTrans_cut = [[prevTrans[0][0], prevTrans[0][1], prevTrans[0][3]], [prevTrans[1][0], prevTrans[1][1], prevTrans[1][3]], [prevTrans[2][0], prevTrans[2][1], prevTrans[2][3]]]
        prevPoints = []
        world_point = []
        # print(self.count)
        # print(prevFrameMatrix)
        for i in self.prev_keypoints:
            temp = np.dot(np.linalg.pinv(prev_focal_matrix), [i[0], i[1], 1])
            temp = np.dot(np.linalg.inv(prevFrameMatrix), temp)
            # temp = np.dot(np.linalg.inv(prevTrans_cut), [i[0], i[1], 1])
            # gamma = 1 / temp[2]
            # temp = temp[:2] * gamma
            world_point.append(temp)
        # 2
        # 4
        # x_y_rate = 5
        # z_rate = 1
        # virturl_point = [[x_y_rate, x_y_rate, z_rate], [x_y_rate, x_y_rate, z_rate], [x_y_rate, -x_y_rate, z_rate], [-x_y_rate, x_y_rate, z_rate], [-x_y_rate, -x_y_rate, z_rate], [-x_y_rate, x_y_rate, z_rate], [x_y_rate, -x_y_rate, z_rate], [-x_y_rate, -x_y_rate, z_rate]]
        # prevFrameMatrix = [[prevFrameMatrix_r[0][0], prevFrameMatrix_r[0][1], prevFrameMatrix_r[0][2], prevFrameMatrix_t[0]], [prevFrameMatrix_r[1][0], prevFrameMatrix_r[1][1], prevFrameMatrix_r[1][2], prevFrameMatrix_t[1]], [prevFrameMatrix_r[2][0], prevFrameMatrix_r[2][1], prevFrameMatrix_r[2][2], prevFrameMatrix_t[2]], [0, 0, 0, 1]]
        # prev_focal_matrix = [[prev_focal_matrix[0][0], prev_focal_matrix[0][1], prev_focal_matrix[0][2], 0], [prev_focal_matrix[1][0], prev_focal_matrix[1][1], prev_focal_matrix[1][2], 0], [prev_focal_matrix[2][0], prev_focal_matrix[2][1], prev_focal_matrix[2][2], 0]]
        # prevTrans = np.dot(prev_focal_matrix, prevFrameMatrix)
        # prevPoints = []
        # for i in virturl_point:
        #     temp = np.dot(prevTrans, [i[0], i[1], i[2], 1])
        #     prevPoints.append([temp[0]/temp[2], temp[1]/temp[2]])
        # 4
        # for i in V:
        #     temp = np.dot(prevTrans, [i[0], i[1], i[2], 1])
        #     prevPoints.append([temp[0]/temp[2], temp[1]/temp[2]])
        # prevPoints = np.array(prevPoints)
        # bn_prev = np.dot(prevTrans, bn_point)
        # bn_prev = [bn_prev[0]/bn_prev[2], bn_prev[1]/bn_prev[2]]
        # bw_prev = np.dot(prevTrans, bw_point)
        # bw_prev = [bw_prev[0]/bw_prev[2], bw_prev[1]/bw_prev[2]]
        # bs_prev = np.dot(prevTrans, bs_point)
        # bs_prev = [bs_prev[0]/bs_prev[2], bs_prev[1]/bs_prev[2]]
        # be_prev = np.dot(prevTrans, be_point)
        # be_prev = [be_prev[0]/be_prev[2], be_prev[1]/be_prev[2]]
        # tn_prev = np.dot(prevTrans, tn_point)
        # tn_prev = [tn_prev[0]/tn_prev[2], tn_prev[1]/tn_prev[2]]
        # tw_prev = np.dot(prevTrans, tw_point)
        # tw_prev = [tw_prev[0]/tw_prev[2], tw_prev[1]/tw_prev[2]]
        # ts_prev = np.dot(prevTrans, ts_point)
        # ts_prev = [ts_prev[0]/ts_prev[2], ts_prev[1]/ts_prev[2]]
        # te_prev = np.dot(prevTrans, te_point)
        # te_prev = [te_prev[0]/te_prev[2], te_prev[1]/te_prev[2]]
        # print('prev')
        # print(prevFrameMatrix)
        # print(prevTrans)
        # print(te_point)
        # print(te_prev)
        # prevPoints = np.array([bn_prev, bw_prev, bs_prev, be_prev])
        # print(prevPoints)
        # det_box = [tlbr_to_tlwh(tlbr) for tlbr in detections][0]
        # print(det_box)
        # det_box = [det_box[0], det_box[1], 1, det_box[2], det_box[3], 1]
        # 5
        # prevFrameMatrix = [[prevFrameMatrix_r[0][0], prevFrameMatrix_r[0][1], prevFrameMatrix_r[0][2], prevFrameMatrix_t[0]], [prevFrameMatrix_r[1][0], prevFrameMatrix_r[1][1], prevFrameMatrix_r[1][2], prevFrameMatrix_t[1]], [prevFrameMatrix_r[2][0], prevFrameMatrix_r[2][1], prevFrameMatrix_r[2][2], prevFrameMatrix_t[2]], [0, 0, 0, 1]]
        # prev_focal_matrix = [[prev_focal_matrix[0][0], prev_focal_matrix[0][1], prev_focal_matrix[0][2], 0], [prev_focal_matrix[1][0], prev_focal_matrix[1][1], prev_focal_matrix[1][2], 0], [prev_focal_matrix[2][0], prev_focal_matrix[2][1], prev_focal_matrix[2][2], 0]]
        # prevTrans = np.dot(prev_focal_matrix, prevFrameMatrix)
        # prevPoints = []
        # point_index = 0
        # for i in world_point:
        #     zc_prev = i[0] * prevFrameMatrix_r[2][0] + i[1] * prevFrameMatrix_r[2][1] + i[2] * prevFrameMatrix_r[2][2] + prevFrameMatrix_t[2]
        #     zc_now = i[0] * frameMatrix_r[2][0] + i[1] * frameMatrix_r[2][1] + i[2] * frameMatrix_r[2][2] + frameMatrix_t[2]
        #     temp = np.dot(prevTrans, i)
        #     temp[0] = temp[0] * (zc_prev/zc_now)
        #     temp[1] = temp[1] * (zc_prev/zc_now)
        #     temp[2] = temp[2] * (zc_prev/zc_now)
        #     temp2 = [temp[0]/temp[2], temp[1]/temp[2]]
        #     diff = x_diff * temp2[1]
        #     if diff > 0:
        #         temp2 = [(temp[0]/temp[2]), self.prev_keypoints[point_index][1] - ((temp[1]/temp[2]) - self.prev_keypoints[point_index][1])]
        #     prevPoints.append([temp2[0], temp2[1]])
        #     point_index += 1
        # 5
        # 2
        focal_matrix = [[focal_matrix[0][0], focal_matrix[0][1], focal_matrix[0][2], 0], [focal_matrix[1][0], focal_matrix[1][1], focal_matrix[1][2], 0] ,[focal_matrix[2][0], focal_matrix[2][1], focal_matrix[2][2], 0]]
        currFrameMatrix = [[frameMatrix_r[0][0], frameMatrix_r[0][1], frameMatrix_r[0][2], frameMatrix_t[0]], [frameMatrix_r[1][0], frameMatrix_r[1][1], frameMatrix_r[1][2], frameMatrix_t[1]], [frameMatrix_r[2][0], frameMatrix_r[2][1], frameMatrix_r[2][2], frameMatrix_t[2]], [0, 0, 0, 1]]
        currTrans = np.dot(focal_matrix, currFrameMatrix)
        currTrans_cut = [[currTrans[0][0], currTrans[0][1], currTrans[0][3]], [currTrans[1][0], currTrans[1][1], currTrans[1][3]], [currTrans[2][0], currTrans[2][1], currTrans[2][3]]]
        # FUTURE SOLUTION
        # temp_f = np.dot(np.linalg.inv(focal_matrix), [962, 540, 1])
        # temp_f = np.dot(np.linalg.inv(currFrameMatrix), temp_f)
        # FUTURE SOLUTION
        currPoints = []
        point_index = 0
        opt = 0
        # if self.prev_diff != [] and self.count > 20:
            # threshold = 0.1
            # angular_delta = [abs(x_diff) - abs(self.prev_diff[0]), abs(y_diff) - abs(self.prev_diff[1]), abs(z_diff) - abs(self.prev_diff[2])]
            # angular_delta = [abs(x_diff), abs(y_diff), abs(z_diff)]
            # self.x_diff_list.append(abs(angular_delta[0]))
            # self.y_diff_list.append(abs(angular_delta[1]))
            # self.z_diff_list.append(abs(angular_delta[2]))
            # self.angular_list.append(f"[{abs(angular_delta[0])},{abs(angular_delta[1])},{abs(angular_delta[2])}]\n")
            # self.frame_list.append(self.count)
            # # if abs(angular_delta[0]) > threshold:
            #     print(self.count, "x")
            # if abs(angular_delta[1]) > threshold:
            #     print(self.count, "y")
            # if abs(angular_delta[2]) > threshold:
            #     print(self.count, "z")
        # print("===========================")
        # print(currFrameMatrix)
        for i in world_point:
            # zc_prev = i[0] * prevFrameMatrix_r[2][0] + i[1] * prevFrameMatrix_r[2][1] + i[2] * prevFrameMatrix_r[2][2] + prevFrameMatrix_t[2]
            # zc_now = i[0] * frameMatrix_r[2][0] + i[1] * frameMatrix_r[2][1] + i[2] * frameMatrix_r[2][2] + frameMatrix_t[2]
            # temp = np.dot(currTrans_cut, [i[0], i[1], 1])
            temp = np.dot(currTrans, i)
            # temp[0] = temp[0] * (zc_now/zc_prev)
            # temp[1] = temp[1] * (zc_now/zc_prev)
            # temp[2] = temp[2] * (zc_now/zc_prev)
            # temp2 = [(temp[0]/temp[2]), self.prev_keypoints[point_index][1] - ((temp[1]/temp[2]) - self.prev_keypoints[point_index][1])]
            temp2 = [temp[0]/temp[2], temp[1]/temp[2]]
            diff_x = x_diff * (temp2[1] - self.prev_keypoints[point_index][1])
            diff_y = y_diff * (temp2[0] - self.prev_keypoints[point_index][0])
            if diff_x < 0:
                # print(self.prev_keypoints[point_index][1])
                # print(temp2[1])
                # print(x_diff)
                # print(point_index)
                temp2[1] = self.prev_keypoints[point_index][1] - ((temp[1]/temp[2]) - self.prev_keypoints[point_index][1])
                # opt = 1
                # break
            if diff_y < 0:
                temp2[0] = self.prev_keypoints[point_index][0] - ((temp[0]/temp[2]) - self.prev_keypoints[point_index][0])
            # currPoints.append([temp2[0]/self.downscale, temp2[1]/self.downscale])
            currPoints.append([temp2[0], temp2[1]])
            point_index += 1
        # FUTURE SOLUTION
        # next_focal_matrix = next_camera_parameter[0]
        # n_focal_matrix = [[next_focal_matrix[0][0], next_focal_matrix[0][1], next_focal_matrix[0][2], 0], [next_focal_matrix[1][0], next_focal_matrix[1][1], next_focal_matrix[1][2], 0] ,[next_focal_matrix[2][0], next_focal_matrix[2][1], next_focal_matrix[2][2], 0]]
        # nextTrans = np.dot(n_focal_matrix, [next_camera_parameter[1][0], next_camera_parameter[1][1], next_camera_parameter[1][2], [0, 0, 0, 1]])
        # temp_f = np.dot(nextTrans, temp_f)
        # n_gap = temp_f[0] - 962
        # FUTURE SOLUTION
        # warping
        # grid_x, grid_y = array_ops.meshgrid(
        #     math_ops.range(width), math_ops.range(height))
        # stacked_grid = math_ops.cast(
        #     array_ops.stack([grid_y, grid_x], axis=2), float)
        # currPoints = _interpolate_bilinear(stacked_grid, np.transpose(currPoints), width, height)
        # currPoints = np.transpose(currPoints)
        
        # change opt
        # Downscale image
        # if self.downscale > 1.0:
        #     # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
        #     frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        
        
        # 校準
        # curr_count = 0
        # for index, i in enumerate(currPoints):
        #     if curr_count == 0:
        #         prev_curr = i
        #         curr_count += 1
        #     elif curr_count%10 == 0:
        #         currPoints[index][0] = currPoints[index][0] + (100 - abs(i[0] - currPoints[index-10][0]))
        #         currPoints[index][1] = currPoints[index][1] - (i[1] - currPoints[index-10][1])
        #         prev_curr = currPoints[index]
        #         curr_count += 1
        #     else:
        #         currPoints[index][0] = currPoints[index][0] - (i[0] - prev_curr[0])
        #         currPoints[index][1] = currPoints[index][1] + (100 - abs(i[1] - prev_curr[1]))
        #         prev_curr = currPoints[index]
        #         curr_count += 1
        # 2
        # 4
        # focal_matrix = [[focal_matrix[0][0], focal_matrix[0][1], focal_matrix[0][2], 0], [focal_matrix[1][0], focal_matrix[1][1], focal_matrix[1][2], 0] ,[focal_matrix[2][0], focal_matrix[2][1], focal_matrix[2][2], 0]]
        # currFrameMatrix = [[frameMatrix_r[0][0], frameMatrix_r[0][1], frameMatrix_r[0][2], frameMatrix_t[0]], [frameMatrix_r[1][0], frameMatrix_r[1][1], frameMatrix_r[1][2], frameMatrix_t[1]], [frameMatrix_r[2][0], frameMatrix_r[2][1], frameMatrix_r[2][2], frameMatrix_t[2]], [0, 0, 0, 1]]
        # currTrans = np.dot(focal_matrix, currFrameMatrix)
        # currPoints = []
        # for i in virturl_point:
        #     temp = np.dot(currTrans, [i[0], i[1], i[2], 1])
        #     currPoints.append([temp[0]/temp[2], temp[1]/temp[2]])
        # 4
        # for i in V:
        #     temp = np.dot(currTrans, [i[0], i[1], i[2], 1])
        #     currPoints.append([temp[0]/temp[2], temp[1]/temp[2]])
        # currPoints = np.array(currPoints)
        # bn_curr = np.dot(currTrans, bn_point)
        # bn_curr = [bn_curr[0]/bn_curr[2], bn_curr[1]/bn_curr[2]]
        # bw_curr = np.dot(currTrans, bw_point)
        # bw_curr = [bw_curr[0]/bw_curr[2], bw_curr[1]/bw_curr[2]]
        # bs_curr = np.dot(currTrans, bs_point)
        # bs_curr = [bs_curr[0]/bs_curr[2], bs_curr[1]/bs_curr[2]]
        # be_curr = np.dot(currTrans, be_point)
        # be_curr = [be_curr[0]/be_curr[2], be_curr[1]/be_curr[2]]
        # tn_curr = np.dot(currTrans, tn_point)
        # tn_curr = [tn_curr[0]/tn_curr[2], tn_curr[1]/tn_curr[2]]
        # tw_curr = np.dot(currTrans, tw_point)
        # tw_curr = [tw_curr[0]/tw_curr[2], tw_curr[1]/tw_curr[2]]
        # ts_curr = np.dot(currTrans, ts_point)
        # ts_curr = [ts_curr[0]/ts_curr[2], ts_curr[1]/ts_curr[2]]
        # te_curr = np.dot(currTrans, te_point)
        # te_curr = [te_curr[0]/te_curr[2], te_curr[1]/te_curr[2]]
        # print('curr')
        # print(currFrameMatrix)
        # print(currTrans)
        # print(te_point)
        # print(te_curr)
        # currPoints = np.array([bn_curr, bw_curr, bs_curr, be_curr])
        # print(currPoints)
        # Find rigid matrix
        # H = np.dot(focal_matrix, fuse_matrix)
        # 1
        # H = fuse_matrix
        # 1
        # 2
        self.prev_keypoints = np.array(self.prev_keypoints)
        # prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)
        # print(self.prev_keypoints[0])
        # print(prevPoints[0])
        # print(currPoints[0])
        
        # test mute max and min 10 point
        # gap = []
        # for i in range(len(self.prev_keypoints)):
        #     temp_gap = cmath.sqrt(math.pow(currPoints[i][0] - self.prev_keypoints[i][0], 2) + math.pow(currPoints[i][1] - self.prev_keypoints[i][1], 2)).real
        #     gap.append(temp_gap)
        # for i in range(10):
        #     gap_index =  gap.index(max(gap))
        #     np.delete(self.prev_keypoints, gap_index)
        #     np.delete(currPoints, gap_index)
        #     del gap[gap_index]
        # for i in range(10):
        #     gap_index =  gap.index(min(gap))
        #     np.delete(self.prev_keypoints, gap_index)
        #     np.delete(currPoints, gap_index)
        #     del gap[gap_index]
        
        # downscale
        # ds_prev_keypoint = []
        # for i in self.prev_keypoints:
        #     ds_prev_keypoint.append([i[0]//self.downscale, i[1]//self.downscale])
        # ds_prev_keypoint = np.array(ds_prev_keypoint)
        
        # ds_currPoints = []
        # for i in currPoints:
        #     ds_currPoints.append([i[0]//self.downscale, i[1]//self.downscale])
        # ds_currPoints = np.array(ds_currPoints)

        # print(self.prev_keypoints)
        # print("--------------------------")
        # print(currPoints[0])
        # print("--------------------------")
        # H, inliesrs = cv2.estimateAffinePartial2D(ds_prev_keypoint.astype(np.float32), ds_currPoints.astype(np.float32), cv2.RANSAC)
        # print(H)
        # print("--------------------------")
        # print(H)
        # print("POINTS")
        # for i, _ in enumerate(self.prev_keypoints):
        #     print([self.prev_keypoints[i], currPoints[i]])
        
        H, inliesrs = cv2.estimateAffinePartial2D(self.prev_keypoints.astype(np.float32), currPoints.astype(np.float32), cv2.RANSAC)
        # 2
        
        # H[1][2] = -(H[1][2])
        # diff = x_diff * H[1][2]
        # if diff > 0:
        #     H[1][2] = -(H[1][2])
        # print([focal_matrix[0][0] * (frameMatrix_t[0] - prevFrameMatrix_t[0]), focal_matrix[1][1] * (frameMatrix_t[1] - prevFrameMatrix_t[1])])
        # print('#!')
        # print([H[0][2], H[1][2]])
        # print([compensation_matrix[0], compensation_matrix[1]])
        # print((compensation_matrix[0] + focal_matrix[0][0] * (frameMatrix_t[0] - prevFrameMatrix_t[0])), (compensation_matrix[1] + focal_matrix[1][1] * (frameMatrix_t[1] - prevFrameMatrix_t[1])))
        # print('##')
        
        # SKIP SOLUTION
        # if H[0][2]*self.prev_H[0][2] < 0 and abs(n_gap) > 100 and abs(H[0][2]) < abs(n_gap):
        #     skip_check = True
            # H = np.array([[1, 0, 0], [0, 1, 0]])
        # SKIP SOLUTION
        
        # FUTURE SOLUTION
        # if self.last_call == 1:
        #      H[0][2] *= 1.3
        #      self.last_call = 0
        # if H[0][2]*self.prev_H[0][2] < 0 and abs(n_gap) > 100 and abs(H[0][2]) < abs(n_gap):
        #     print(self.count)
        #     print(n_gap)
        #     print(H[0][2])
        #     print(self.prev_H[0][2])
        #     H[0][2] *= (abs(n_gap)*0.6)/abs(H[0][2])
        #     self.last_call = 1
        #     print(H[0][2])
        # FUTURE SOLUTION
        
        # OPT SOLUTION
        # if  H[0][2]*self.prev_H[0][2] < 0 and abs(H[0][2] - self.prev_H[0][2]) > 10:
        #     print(self.count)
        #     opt = 1
            
        # if opt == 1:
        #     prev_key = cv2.goodFeaturesToTrack(self.prevFrame, mask=None, **self.feature_params)
        #     matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, prev_key, None)
        #     prevPoints = []
        #     currPoints = []
        #     for i in range(len(status)):
        #         if status[i]:
        #             prevPoints.append(prev_key[i])
        #             currPoints.append(matchedKeypoints[i])

        #     prevPoints = np.array(prevPoints)
        #     currPoints = np.array(currPoints)
        # if opt == 1:
        #     H, inliesrs = cv2.estimateAffinePartial2D(prevPoints.astype(np.float32), currPoints.astype(np.float32), cv2.RANSAC)
        # OPT SOLUTION
        # print(H[0][0:2])
        # print(H[1][0:2])
        # print([H[0][2], H[1][2]])
        
        # Handle downscale
        # if self.downscale > 1.0:
        #     H[0, 2] *= self.downscale
        #     H[1, 2] *= self.downscale
        
        # mix compesation
        # print("MATRIX")
        # print([H, compensation_matrix])
        H[0][2] = H[0][2]*0.5 + compensation_matrix[0]*0.5
        H[1][2] = H[1][2]*0.5 + compensation_matrix[1]*0.5
        # H[:, 0:2] = (H[:, 0:2] + compensation_r)/2
        # theta = math.atan(H[1][0]/H[0][0])
        # sx = math.sqrt(math.pow(H[0][0], 2) + math.pow(H[1][0], 2))
        # # sx = H[0][0]/math.cos(theta)
        # msy = H[0][1] * math.cos(theta) + H[1][1] * math.sin(theta)
        # if math.sin(theta) != 0:
        #     sy = (msy * math.cos(theta) - H[0][1])/math.sin(theta)
        # else:
        #     sy = (H[1][1] - msy * math.sin(theta))/math.cos(theta)
        # m = msy / sy
        # rotate_avg = (theta + (angle_z - prev_angle_z))/2
        # A11 = sx * math.cos(rotate_avg)
        # A12 = sy * m * math.cos(rotate_avg) - sy * math.sin(rotate_avg)
        # A21 = sx * math.sin(rotate_avg)
        # A22 = sy * m * math.sin(rotate_avg) + sy * math.cos(rotate_avg)
        # H[:, :2] = [[A11, A12], [A21, A22]]
        self.prev_H = H
        self.prev_diff = [x_diff, y_diff, z_diff]
        # Store to next iteration
        self.prevFrame = frame.copy()
        #前後幀矩陣變化
        t1 = time.time()
        self.prev_detections = detections
        # 2
        curr = [self.prev_keypoints, currPoints]
        # self.prev_keypoints = currPoints
        # self.prev_fix_point = self.fix_point
        # if len(self.fix_list) > 20:
        #     self.prev_fix_point = [935, 540]
        #     self.fix_point = [935, 540]
        #     self.fix_list = []
        # if self.prev_fix_point[0] > 1920:
        #     if self.prev_fix_point[1] > 1280:
        #         self.prev_fix_point = [935, 540]
        #         self.fix_point = [935, 540]
        #         self.fix_list = []
        self.fix_point = [H[0][0] * self.fix_point[0] + H[0][1] * self.fix_point[1] + H[0][2], H[1][0] * self.fix_point[0] + H[1][1] * self.fix_point[1] + H[1][2]]
        # self.fix_list.append(self.fix_point)
        # self.prev_keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        # point = []
        # for i in detections:
        #     point.append([i[0], i[1]])
        #     point.append([i[0], (i[3] - i[1])/2])
        #     point.append([i[2], (i[3] - i[1])/2])
        #     point.append([i[0], i[3]])
        #     point.append([i[2], i[1]])
        #     point.append([(i[2] - i[0])/2, i[1]])
        #     point.append([(i[2] - i[0])/2, i[3]])
        #     point.append([i[2], i[3]])
        # self.prev_keypoints = point
        self.prev_currpoint = self.prev_keypoints.astype(np.float32)
        # 2
        # gmc_line = str(1000 * (t1 - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)
        
        # test camera parameter
        # if len(prev_camera_parameter) < 4:
        #     prev_camera_parameter.append([0, 0, 0, 1])
        # if len(camera_parameter) < 4:
        #     camera_parameter.append([0, 0, 0, 1])
        # point_assume = [100, 100, 100, 1]
        # prev_assume = np.dot(np.dot(prev_focal_matrix, prev_camera_parameter), point_assume)
        # curr_assume = np.dot(np.dot(focal_matrix, camera_parameter), point_assume)
        # prev_assume = [prev_assume[0]/prev_assume[2], prev_assume[1]/prev_assume[2]]
        # curr_assume = [curr_assume[0]/curr_assume[2], curr_assume[1]/curr_assume[2]]
        # print(prev_assume)
        # print(curr_assume)
        # print(self.fix_point)
        # print(self.count)
        # print(H)
        
        # if self.count == 120:
        #     plt.plot(self.frame_list, self.x_diff_list, color='b')
        #     plt.plot(self.frame_list, self.y_diff_list, color='r')
        #     plt.plot(self.frame_list, self.z_diff_list, color='g')
        #     plt.xlabel('frame') # 設定x軸標題
        #     plt.xticks(self.frame_list, rotation='vertical') # 設定x軸label以及垂直顯示
        #     plt.title('angular_change') # 設定圖表標題
        #     plt.ylim(0.00,0.200)
        #     # with open("angular/imu/imu_0000.txt", 'w') as f:
        #     #     f.writelines(self.angular_list)
        #     plt.show()
        
        # change Iterative compensation
        # H = np.array(H)
        # self.H = np.array(self.H)
        # R = np.dot(self.H[:, :2], H[:, :2])
        # self.H = [[R[0][0], R[0][1], H[0][2] + self.H[0][2]], [R[1][0], R[1][1], H[1][2] + self.H[1][2]]]
        # H =  self.H
        # change
        
        #change rotation matching
        rotation = [x_diff, y_diff, z_diff]
        #change 
        
        # print(H)
        # print(angle_z - prev_angle_z)
        # p = input()
        return H, rotation, skip_check

def _interpolate_bilinear(grid,
                          query_points, width, height, 
                          name='interpolate_bilinear',
                          indexing='xy'):
    """Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
      name: a name for the operation (optional).
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the inputs
        invalid.
    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

    with ops.name_scope(name):
        grid = ops.convert_to_tensor(grid)
        query_points = ops.convert_to_tensor(query_points)
        shape = array_ops.unstack(array_ops.shape(query_points))
        # if len(shape) != 4:
        #     msg = 'Grid must be 4 dimensional. Received: '
        #     raise ValueError(msg + str(shape))
        query_type = query_points.dtype
        query_shape = array_ops.unstack(array_ops.shape(query_points))
        grid_type = grid.dtype
        im_shape = [width, height]
        s_height, s_width = shape

        # if len(query_shape) != 3:
        #     msg = ('Query points must be 3 dimensional. Received: ')
        #     raise ValueError(msg + str(query_shape))
        
        alphas = []
        floors = []
        ceils = []

        index_order = [0, 1] if indexing == 'ij' else [1, 0]
        # unstacked_query_points = array_ops.unstack(query_points, axis=2)

        for dim in index_order:
            with ops.name_scope('dim-' + str(dim)):
                queries = query_points[dim]

                size_in_indexing_dimension = im_shape[dim]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = constant_op.constant(0.0, dtype=query_type)
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                int_floor = math_ops.cast(floor, dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = math_ops.cast(queries - floor, grid_type)
                min_alpha = constant_op.constant(0.0, dtype=grid_type)
                max_alpha = constant_op.constant(1.0, dtype=grid_type)
                alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                # alpha = array_ops.expand_dims(alpha, 2)
                alphas.append(alpha)

        flattened_grid = array_ops.reshape(grid, [width, height]) # !
        batch_offsets = array_ops.reshape(
            math_ops.range(1) * s_height * s_width, [1, 1])

        # This wraps array_ops.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using array_ops.gather_nd.
        def gather(y_coords, x_coords, name):
            with ops.name_scope('gather-' + name):
                # print(batch_offsets)
                linear_coordinates = (batch_offsets + y_coords * width + x_coords)
                print(linear_coordinates)
                gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                return array_ops.reshape(gathered_values,
                                         [s_height, s_width])

        # print(floors[1])
        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], 'top_left')
        top_right = gather(floors[0], ceils[1], 'top_right')
        bottom_left = gather(ceils[0], floors[1], 'bottom_left')
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right')
        # print(top_left)
        # print(top_right)
        

        # now, do the actual interpolation
        with ops.name_scope('interpolate'):
            # print(alphas[1])
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top
        # print(interp)
        return interp