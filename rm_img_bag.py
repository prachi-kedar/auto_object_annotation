# Import the library OpenCV
import cv2

# Import the image
file_name = "test.png"

# Read the image
src = cv2.imread(file_name, 1)
# Convert image to image gray
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

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
b, g, r = cv2.split(src)

# Making list of Red, Green, Blue
# Channels and alpha
rgba = [b, g, r, alpha]

# Using cv2.merge() to merge rgba
# into a coloured/multi-channeled image
dst = cv2.merge(rgba, 4)

# Writing and saving to a new image
# Saves the frames with frame-count
cv2.imwrite("/home/neosoft/Desktop/projects/automatic_object_annotation/framed.png", dst)
