import cv2
import numpy as np

img = cv2.imread("Fake Notes/500/sim_nothread_500_s1.jpg")
img = cv2.resize(img, (1167, 519))
thread_crop = img[100:400, 560:630]
gray_strip = cv2.cvtColor(thread_crop, cv2.COLOR_BGR2GRAY)

green_mask = (thread_crop[:, :, 1].astype(np.int16) > thread_crop[:, :, 2].astype(np.int16) + 3) & (gray_strip < 210)

# Print columns and how many green pixels they have
for col in range(green_mask.shape[1]):
    count = np.sum(green_mask[:, col])
    if count > 0:
        print(f"Col {560+col}: count={count} ({count/green_mask.shape[0]:.4f})")
