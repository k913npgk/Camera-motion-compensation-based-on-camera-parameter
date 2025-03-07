import cv2

# raw_frame = cv2.imread('C:/Users/CSY/SMILEtrack-main/BoT-SORT/datasets/sports/test/img1/000001.png')
# raw_prevFrame = cv2.imread('C:/Users/CSY/SMILEtrack-main/BoT-SORT/datasets/sports/test/img1/000002.png')

raw_frame = cv2.imread('C:/Users/CSY/SMILEtrack-main/BoT-SORT/datasets/sports/test/cp_speed_1/000002.jpg')
raw_prevFrame = cv2.imread('C:/Users/CSY/SMILEtrack-main/BoT-SORT/datasets/sports/test/cp_speed_1/000001.jpg')

feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)

frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
prevFrame = cv2.cvtColor(raw_prevFrame, cv2.COLOR_BGR2GRAY)
keypoints = cv2.goodFeaturesToTrack(prevFrame, mask=None, **feature_params)

matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(prevFrame, frame, keypoints, None)

for i, p in enumerate(keypoints):
    if int(p[0][0]) < 860 and int(p[0][0]) > 770 and int(p[0][1]) < 380 and int(p[0][1]) > 217:
        cv2.circle(raw_prevFrame, (int(p[0][0]), int(p[0][1])), 3, (0, 0, 255), -1)
        cv2.line(raw_prevFrame, (int(p[0][0]), int(p[0][1])), (int(matchedKeypoints[i][0][0]), int(matchedKeypoints[i][0][1])), (0, 255, 255), 2)

# for i in matchedKeypoints:
#     cv2.circle(raw_prevFrame, (int(i[0][0]), int(i[0][1])), 3, (0, 255, 255), -1)
    
cv2.namedWindow("image")
cv2.imshow('image', raw_prevFrame)
cv2.waitKey (0)
cv2.destroyAllWindows()

# cv2.namedWindow("image")
# cv2.imshow('image', raw_frame)
# cv2.waitKey (0)
# cv2.destroyAllWindows()