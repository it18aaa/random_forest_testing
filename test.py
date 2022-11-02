import numpy as np
import cv2 as cv2
import pandas as pd
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import sobel
from skimage.filters import scharr
from scipy import ndimage as nd

img = cv2.imread('img/3.jpg')

# resize it so we can play with the image
scale_percent = 15
dim = int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# opencv is BGR ?
img_red = img[:,:,2]
img_blue = img[:,:,1]
img_green = img[:,:,0]

# entropy filter only works on grayscale?
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_entropy = entropy(img_gray, disk(1))

img_gaussian = nd.gaussian_filter(img_gray, sigma=3)

img_entropy2 = entropy(img_gaussian, disk(2))
img_sobel = sobel(img_gray)
img_scharr = scharr(img_gray )

cv2.imshow('original', img)
cv2.imshow('img_red', img_red)
cv2.imshow('img_green', img_green)
cv2.imshow('img_blue', img_blue)
cv2.imshow('img_gray', img_gray)
cv2.imshow('entropy', img_entropy)
cv2.imshow('gaussian', img_gaussian)
cv2.imshow('sobel', img_sobel)
cv2.imshow('scharr', img_scharr)
cv2.imshow('entropy2', img_entropy2)

# flatten the images, so they're flat arrays of pixels
# these columns would be 'features'

# prepare dataframe for machine learning activity

df = pd.DataFrame()
df['Red'] = img_red.reshape(-1)
df['Green'] = img_green.reshape(-1)
df['Blue'] = img_blue.reshape(-1)
df['Gray'] = img_gray.reshape(-1)
df['Entropy'] = img_entropy.reshape(-1)
df['Scharr'] = img_scharr.reshape(-1)
df['Sobel'] = img_sobel.reshape(-1)

# print first and last  10
#print(df.head(10))
#print(df.tail(10))

print(df)
cv2.waitKey()
cv2.destroyAllWindows()