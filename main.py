import cv2 as cv
import numpy as np
from aux_ import FeatureExtraction, feature_matching

img0 = cv.imread("lasmeninas0.jpg", cv.COLOR_BGR2RGBA)
img1 = cv.imread("lasmeninas1.jpg", cv.COLOR_BGR2RGBA)
features0 = FeatureExtraction(img0)
features1 = FeatureExtraction(img1)

matches = feature_matching(features0, features1)
# matched_image = cv.drawMatches(img0, features0.kps,img1, features1.kps, matches, None, flags=2)

H, _ = cv.findHomography( features0.matched_pts,features1.matched_pts, cv.RANSAC, 5.0)

h, w, c = img1.shape
warped = cv.warpPerspective(img0, H, (w, h),borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

output = np.zeros((h, w, 3), np.uint8)
alpha = warped[:, :, 3] / 255.0
output[:, :, 0] = (1. - alpha) * img1[:, :, 0] + alpha * warped[:, :, 0]
output[:, :, 1] = (1. - alpha) * img1[:, :, 1] + alpha * warped[:, :, 1]
output[:, :, 2] = (1. - alpha) * img1[:, :, 2] + alpha * warped[:, :, 2]