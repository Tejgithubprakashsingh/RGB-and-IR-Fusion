# SIFT(Scale invariant feature Transform)
# Program 1
img = cv2.imread("0.4.png",cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img,None)
img = cv2.drawKeypoints(img, keypoints, None)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
