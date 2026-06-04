import cv2
import numpy as np

img = cv2.imread("Dataset/2000_dataset/2000_s1.jpg")
img_res = cv2.resize(img, (1165, 455))

# Let's save a visualization or scan column by column.
# Let's find columns where there is a distinct change in G vs R or a local color change.
# Let's print the mean R, G, B for each 10-pixel column in the middle part of the note (e.g. from x=400 to x=800)
for x in range(400, 800, 10):
    col_strip = img_res[100:350, x:x+10]
    avg_b = np.mean(col_strip[:, :, 0])
    avg_g = np.mean(col_strip[:, :, 1])
    avg_r = np.mean(col_strip[:, :, 2])
    print(f"Col {x:3d}: B={avg_b:5.1f}, G={avg_g:5.1f}, R={avg_r:5.1f}")
