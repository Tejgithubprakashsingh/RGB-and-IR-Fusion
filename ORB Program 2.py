# Program 2:
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

if 'img1' not in locals() or 'img2' not in locals() or img1 is None or img2 is None:
    print("Images img1 or img2 are not loaded. Please run the image loading cell first.")
else:
    print("Images found. Proceeding with ORB feature matching and registration.")

    # Create ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Check if descriptors were found
    if descriptors1 is None or descriptors2 is None:
        print("Error: Could not find ORB descriptors in one or both images. Displaying original images.")
        cv2_imshow(img1)
        cv2_imshow(img2)
    else:
        # Create BFMatcher object with NORM_HAMMING for ORB descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 50 matches for visualization
        img_matches_orb = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        print("ORB Matches (first 50):")
        cv2_imshow(img_matches_orb)

        # Check if enough matches are found for homography
        MIN_MATCH_COUNT_ORB = 10 # Increased minimum matches for more reliable homography
        if len(matches) > MIN_MATCH_COUNT_ORB:
            # Extract location of good matches
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate transformation matrix using RANSAC
            matrix_orb, mask_orb = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

            if matrix_orb is not None:
                # Warp image 1 to align with image 2 using the homography matrix
                height, width = img2.shape
                registered_img_orb = cv2.warpPerspective(img1, matrix_orb, (width, height))

                print("Registered Image (img1 aligned to img2 using ORB):")
                cv2_imshow(registered_img_orb)

                registered_img_orb_color = cv2.cvtColor(registered_img_orb, cv2.COLOR_GRAY2BGR)
                img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) # Convert img2 to BGR for blending

                alpha = 0.5 # Adjust alpha for blending
                blended_img_orb = cv2.addWeighted(registered_img_orb_color, alpha, img2_color, 1 - alpha, 0)
                print("Blended Image (Registered img1 + img2 using ORB):")
                cv2_imshow(blended_img_orb)

            else:
                print("Could not find a valid homography matrix with ORB matches.")
                print("Displaying original images.")
                cv2_imshow(img1)
                cv2_imshow(img2)

        else:
            print(f"Not enough ORB matches are found - {len(matches)}/{MIN_MATCH_COUNT_ORB}")
            print("Displaying original images due to insufficient matches.")
            cv2_imshow(img1)
            cv2_imshow(img2)
