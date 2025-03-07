import cmath
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import math

class GMC:
    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        super(GMC, self).__init__()
        self.count = 0
        self.method = method
        self.downscale = max(1, int(downscale))

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

    def apply(self, raw_frame, camera_parameter, prev_camera_parameter, detections=None):
        if self.method == 'orb' or self.method == 'sift':
            return self.applyFeaures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)
        elif self.method == 'new':
            return self.applyNew(raw_frame, camera_parameter, prev_camera_parameter, detections)
        elif self.method == 'none':
            return np.eye(2, 3), [0, 0]
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

            return H, [[], []]

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
            print(self.count)
            print(prevPoints[0])
            print(currPoints[0])
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
            print(H[0][0:2])
            print(H[1][0:2])
            print([H[0][2], H[1][2]])
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        t1 = time.time()

        # gmc_line = str(1000 * (t1 - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)

        return H, [prevPoints, currPoints]

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
    
    def applyNew(self, raw_frame, camera_parameter, prev_camera_parameter, detections=None):
        t0 = time.time()
        self.count += 1
        print(self.count)
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        # Handle first frame
        # Downscale image
        # if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            # frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        
        H = np.eye(2, 3)
        if not self.initializedFirstFrame:
            # self.prev_keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
            point = []
            for i in detections:
                point.append([i[0], i[1]])
                point.append([i[0], (i[3] - i[1])/2])
                point.append([i[2], (i[3] - i[1])/2])
                point.append([i[0], i[3]])
                point.append([i[2], i[1]])
                point.append([(i[2] - i[0])/2, i[1]])
                point.append([(i[2] - i[0])/2, i[3]])
                point.append([i[2], i[3]])
            # for i in range(width):
            #     if i != 0:
            #         if i % 100 == 0:
            #             for j in range(height):
            #                 if j != 0:
            #                     if j % 100 == 0:
            #                         point.append([i, height - j])
                    
            curr = [[[0,0]],[[0,0]]]
            self.fix_list = []
            self.fix_point = [935, 504]
            self.fix_list.append(self.fix_point)
            self.prev_H = [[0, 0, 0], [0, 0, 0]]
            # for i in self.prev_keypoints:
            #     temp = i[0]
            #     # if temp[0] <= 1920:
            #     #     if temp[1] <= 1080:
            #     point.append(temp)
            self.prev_keypoints = point
            # Initialize data
            self.prev_detections = detections
            # Initialization done
            self.initializedFirstFrame = True

            return H, curr
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
        print([prev_angle_x, prev_angle_y, prev_angle_z])
        print([angle_x, angle_y, angle_z])
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)
        prevFrameMatrix_r = [[prev_camera_parameter[0][0], prev_camera_parameter[0][1], prev_camera_parameter[0][2]],[prev_camera_parameter[1][0], prev_camera_parameter[1][1], prev_camera_parameter[1][2]],[prev_camera_parameter[2][0], prev_camera_parameter[2][1], prev_camera_parameter[2][2]]]
        # prevFrameMatrix_r = [[prev_camera_parameter[0][0], prev_camera_parameter[0][1]],[prev_camera_parameter[1][0], prev_camera_parameter[1][1]],[prev_camera_parameter[2][0], prev_camera_parameter[2][1]]]
        # prevFrameMatrix_r = [[math.cos(prev_angle_z), -math.sin(prev_angle_z)], [math.sin(prev_angle_z), math.cos(prev_angle_z)]]
        # prevFrameMatrix_t = [prev_camera_parameter[0][3], prev_camera_parameter[1][3]]
        prevFrameMatrix_t = [prev_camera_parameter[0][3], prev_camera_parameter[1][3], prev_camera_parameter[2][3]]
        frameMatrix_r = [[camera_parameter[0][0], camera_parameter[0][1], camera_parameter[0][2]],[camera_parameter[1][0], camera_parameter[1][1], camera_parameter[1][2]],[camera_parameter[2][0], camera_parameter[2][1], camera_parameter[2][2]]]
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
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            # frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        # 前後幀矩陣變化
        
        # rotate_c = [[rotate_c[0][0], rotate_c[0][1]],[rotate_c[1][0], rotate_c[1][1]],[rotate_c[2][0], rotate_c[2][1]]]
        
        # 1
        # compensation_matrix = [focal_matrix[0][0]*math.sin((angle_y - prev_angle_y)/2), focal_matrix[1][1]*math.sin((angle_x - prev_angle_x)/2), 0]
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
        # 2
        prevFrameMatrix = [[prevFrameMatrix_r[0][0], prevFrameMatrix_r[0][1], prevFrameMatrix_r[0][2], prevFrameMatrix_t[0]], [prevFrameMatrix_r[1][0], prevFrameMatrix_r[1][1], prevFrameMatrix_r[1][2], prevFrameMatrix_t[1]], [prevFrameMatrix_r[2][0], prevFrameMatrix_r[2][1], prevFrameMatrix_r[2][2], prevFrameMatrix_t[2]], [0, 0, 0, 1]]
        prev_focal_matrix = [[prev_focal_matrix[0][0], prev_focal_matrix[0][1], prev_focal_matrix[0][2], 0], [prev_focal_matrix[1][0], prev_focal_matrix[1][1], prev_focal_matrix[1][2], 0], [prev_focal_matrix[2][0], prev_focal_matrix[2][1], prev_focal_matrix[2][2], 0]]
        prevTrans = np.dot(prev_focal_matrix, prevFrameMatrix)
        prevPoints = []
        world_point = []
        for i in self.prev_keypoints:
            temp = np.dot(np.linalg.pinv(prevTrans), [i[0], i[1], 1])
            temp = [temp[0]/temp[3], temp[1]/temp[3], temp[2]/temp[3], temp[3]/temp[3]]
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
        # 2
        focal_matrix = [[focal_matrix[0][0], focal_matrix[0][1], focal_matrix[0][2], 0], [focal_matrix[1][0], focal_matrix[1][1], focal_matrix[1][2], 0] ,[focal_matrix[2][0], focal_matrix[2][1], focal_matrix[2][2], 0]]
        currFrameMatrix = [[frameMatrix_r[0][0], frameMatrix_r[0][1], frameMatrix_r[0][2], frameMatrix_t[0]], [frameMatrix_r[1][0], frameMatrix_r[1][1], frameMatrix_r[1][2], frameMatrix_t[1]], [frameMatrix_r[2][0], frameMatrix_r[2][1], frameMatrix_r[2][2], frameMatrix_t[2]], [0, 0, 0, 1]]
        currTrans = np.dot(focal_matrix, currFrameMatrix)
        currPoints = []
        point_index = 0
        for i in world_point:
            # zc_prev = i[0] * prevFrameMatrix_r[2][0] + i[1] * prevFrameMatrix_r[2][1] + i[2] * prevFrameMatrix_r[2][2] + prevFrameMatrix_t[2]
            # zc_now = i[0] * frameMatrix_r[2][0] + i[1] * frameMatrix_r[2][1] + i[2] * frameMatrix_r[2][2] + frameMatrix_t[2]
            temp = np.dot(currTrans, i)
            # temp[0] = temp[0] * (zc_now/zc_prev)
            # temp[1] = temp[1] * (zc_now/zc_prev)
            # temp[2] = temp[2] * (zc_now/zc_prev)
            # temp2 = [(temp[0]/temp[2]), self.prev_keypoints[point_index][1] - ((temp[1]/temp[2]) - self.prev_keypoints[point_index][1])]
            temp2 = [temp[0]/temp[2], temp[1]/temp[2]]
            currPoints.append([temp2[0], temp2[1]])
            point_index += 1
            
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
        print(self.prev_keypoints[0])
        # print(prevPoints[0])
        print(currPoints[0])
        
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
        
        
        H, inliesrs = cv2.estimateAffinePartial2D(self.prev_keypoints.astype(np.float32), currPoints.astype(np.float32), cv2.RANSAC)
        # 2
        # Handle downscale
        # if self.downscale > 1.0:
        #     H[0, 2] *= self.downscale
        #     H[1, 2] *= self.downscale
        
        # H[1][2] = -(H[1][2])
        print(H[0][0:2])
        print(H[1][0:2])
        print([H[0][2], H[1][2]])
        self.prev_H = H
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
        # self.fix_point = [H[0][0] * self.fix_point[0] + H[0][1] * self.fix_point[1] + H[0][2], H[1][0] * self.fix_point[0] + H[1][1] * self.fix_point[1] + H[1][2]]
        # self.fix_list.append(self.fix_point)
        # self.prev_keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        point = []
        for i in detections:
            point.append([i[0], i[1]])
            point.append([i[0], (i[3] - i[1])/2])
            point.append([i[2], (i[3] - i[1])/2])
            point.append([i[0], i[3]])
            point.append([i[2], i[1]])
            point.append([(i[2] - i[0])/2, i[1]])
            point.append([(i[2] - i[0])/2, i[3]])
            point.append([i[2], i[3]])
        # for i in self.prev_keypoints:
        #     temp = i[0]
        #     point.append(temp)
        self.prev_keypoints = point
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

        return H, curr
    