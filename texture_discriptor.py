import numpy as np
import cv2
import matplotlib.pyplot as plt

# این کد روش استخراج ویژگی در یکی از مقاله ها را مشخص می کند. عنوان مقاله در زیر آمده است.
# A locally based feature descriptor for abnormalities detection
def texture_discriptor(img_gray=None, P=8, R=1):
    '''
    :param img_gray: and gray image.
    :param P: P must be like a kernel {0-8-center=5}=3*3 or {0-24-center=13}=5*5 or {0-49-center=25}=7*7 or ...
    :param R: R related to P. {P=8 -- R=1} or {P=25 -- R=2} or {P=49 -- R=3} or ...
    :return:
    '''
    r, c = img_gray.shape
    img_gray = img_gray.astype(np.float64)
    img_gray_temp = np.zeros((r + (2 * R), c + (2 * R)), np.float64)
    img_gray_temp[R:R + r, R:R + c] = img_gray.copy()
    img_gray = img_gray_temp.copy()
    r, c = img_gray.shape
    F1 = np.zeros((r, c), dtype=np.float64)
    k_num = R
    for i in range(R, r - R):
        for j in range(R, c - R):
            temp_img = img_gray[i - k_num:i + k_num + 1, j - k_num:j + k_num + 1].copy()
            temp_img = temp_img.flatten()
            center_idx = int(len(temp_img) / 2)
            g_c = temp_img[center_idx]
            T1 = g_c - ((np.sum(temp_img) - g_c) / P)
            for p in range(0, P):
                g_p = temp_img[p]
                if np.abs(g_c - g_p) < T1:
                    F1[i, j] += 1
    F1 = F1[R:r - R, R:c - R]
    return F1


directory = 'images/'
image_BGR = cv2.imread(directory+'1.jpg')
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)
cv2.imwrite(directory+'1_gray.jpg', image_gray)

# you can change these two parameters base on article
P_param = 8
R_param = 1

# get texture of RGB image
R, G, B = cv2.split(image_RGB)
texture_R = texture_discriptor(img_gray=R, P=P_param, R=R_param)
texture_G = texture_discriptor(img_gray=G, P=P_param, R=R_param)
texture_B = texture_discriptor(img_gray=B, P=P_param, R=R_param)
texture_RGB = cv2.merge((texture_R, texture_G, texture_B))

# get texture of gray image
texture_gray = texture_discriptor(img_gray=image_gray, P=P_param, R=R_param)

cv2.imwrite(directory+'texture_R.jpg', texture_R)
cv2.imwrite(directory+'texture_G.jpg', texture_G)
cv2.imwrite(directory+'texture_B.jpg', texture_B)
cv2.imwrite(directory+'texture_RGB.jpg', texture_RGB)
cv2.imwrite(directory+'texture_gray.jpg', texture_gray)

plt.subplot(2, 5, 1), plt.axis("off"), plt.imshow(image_RGB, cmap='gray'), plt.title('RGB')
plt.subplot(2, 5, 2), plt.axis("off"), plt.imshow(image_gray, cmap='gray'), plt.title('gray')
plt.subplot(2, 5, 3), plt.axis("off"), plt.imshow(R, cmap='gray'), plt.title('R')
plt.subplot(2, 5, 4), plt.axis("off"), plt.imshow(G, cmap='gray'), plt.title('G')
plt.subplot(2, 5, 5), plt.axis("off"), plt.imshow(B, cmap='gray'), plt.title('B')
plt.subplot(2, 5, 6), plt.axis("off"), plt.imshow(texture_RGB, cmap='gray'), plt.title('texture_RGB')
plt.subplot(2, 5, 7), plt.axis("off"), plt.imshow(texture_gray, cmap='gray'), plt.title('texture_gray')
plt.subplot(2, 5, 8), plt.axis("off"), plt.imshow(texture_R, cmap='gray'), plt.title('texture_R')
plt.subplot(2, 5, 9), plt.axis("off"), plt.imshow(texture_G, cmap='gray'), plt.title('texture_G')
plt.subplot(2, 5, 10), plt.axis("off"), plt.imshow(texture_B, cmap='gray'), plt.title('texture_B')
plt.subplots_adjust(left=0.001, bottom=0.03, right=0.999, top=0.930, wspace=0.001, hspace=0.15)
plt.show()