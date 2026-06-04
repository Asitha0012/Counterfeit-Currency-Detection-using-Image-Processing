import cv2
import numpy as np

img = cv2.imread("Dataset/2000_dataset/2000_s1.jpg")
img_res = cv2.resize(img, (1165, 455))
thread_crop = img_res[100:350, 560:630]
# Let's print some pixel colors in the thread crop
print("Thread crop shape:", thread_crop.shape)
# Print average channels
print("Average B:", np.mean(thread_crop[:, :, 0]))
print("Average G:", np.mean(thread_crop[:, :, 1]))
print("Average R:", np.mean(thread_crop[:, :, 2]))

# Let's find any pixels where G > R
g_gt_r = thread_crop[:, :, 1] > thread_crop[:, :, 2]
print("Count of G > R pixels:", np.sum(g_gt_r))

# Let's find any pixels where R > G (since 2000 note is pinkish/reddish, the thread might be reddish/pinkish or yellow/green)
r_gt_g = thread_crop[:, :, 2] > thread_crop[:, :, 1]
print("Count of R > G pixels:", np.sum(r_gt_g))

# Let's print some coordinate lines in the thread region to see where the thread is actually located in 2000_s1.jpg
# The thread is a vertical line. Let's search across the whole note width for a vertical green/blue line, or let's see where the thread is!
# We can find the thread by looking at the column-wise standard deviation or BGR values.
for col in range(0, img_res.shape[1], 10):
    col_strip = img_res[:, col:col+10]
    # Check green-to-red ratio or color shift
    g_mean = np.mean(col_strip[:, :, 1])
    r_mean = np.mean(col_strip[:, :, 2])
    b_mean = np.mean(col_strip[:, :, 0])
    if g_mean > r_mean + 5:
        print(f"Col {col}-{col+10}: G={g_mean:.1f}, R={r_mean:.1f}, B={b_mean:.1f} (Greenish)")
