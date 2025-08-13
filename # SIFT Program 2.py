# Program 2
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
uploaded = files.upload()
image1_filename = list(uploaded.keys())[0]
image2_filename = list(uploaded.keys())[1]
# Load images
img1 = cv2.imread(image1_filename, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image2_filename, cv2.IMREAD_GRAYSCALE)
# Check if images are loaded successfully
if img1 is None:
    print(f"Error: Could not load image 1: {image1_filename}")
elif img2 is None:
    print(f"Error: Could not load image 2: {image2_filename}")
else:
    # Detect SIFT features
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    # Check if descriptors were found
    if descriptors1 is None or descriptors2 is None:
        print("Error: Could not find descriptors in one or both images.")
        cv2_imshow(img1)
        cv2_imshow(img2)
    else:
        # Match features using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Check if enough matches are found
        MIN_MATCH_COUNT = 10
        if len(matches) > MIN_MATCH_COUNT:
            # Extract location of good matches
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate transformation matrix
            matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

            # Draw matches for visualization
            img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2_imshow(img_matches)

         
   # Warp image
            height, width = img2.shape
            registered_img = cv2.warpPerspective(img1, matrix, (width, height))

            # Show results using cv2_imshow
            cv2_imshow(registered_img)
        else:
            print("Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT))
            cv2_imshow(img1)
            cv2_imshow(img2)
