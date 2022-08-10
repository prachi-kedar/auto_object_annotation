# Import the library OpenCV
import cv2
import numpy as np

cap = cv2.VideoCapture("/home/neosoft/Desktop/projects/automatic_object_annotation/edited_vdo/edited_vdo1.mp4")

# Used as counter variable
count = 0

# checks whether frames were extracted
success = 1

# Loop until the end of the video
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert image to image gray
    tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = 255 * (tmp < 128).astype(np.uint8)
    gray_new = 255 * (gray < 128).astype(np.uint8)

    coords = cv2.findNonZero(gray_new)
    # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)

    rows, cols = tmp.shape

    # Calculating average pixel
    pixl_list = []
    for i in range(rows):
        for j in range(cols):
            pixl_list.append(tmp[i, j])
    avg_pix_val = (sum(pixl_list) / len(pixl_list))

    # Applying thresholding technique
    _, alpha = cv2.threshold(tmp, avg_pix_val, 255, cv2.THRESH_BINARY)

    # Using cv2.split() to split channels
    # of coloured image
    b, g, r = cv2.split(frame)

    # Making list of Red, Green, Blue
    # Channels and alpha
    rgba = [b, g, r, alpha]

    # Using cv2.merge() to merge rgba
    # into a coloured/multi-channeled image
    dst = cv2.merge(rgba, 4)
    rect = dst[y:y + h, x:x + w]

    # Writing and saving to a new image
    # Saves the frames with frame-count
    cv2.imwrite("/home/neosoft/Desktop/projects/automatic_object_annotation/output/frame%d.png" % count, rect)

    count += 1
