import cv2
import matplotlib.pyplot as plt
import numpy as np

def AHE(image):
    print(image.dtype) ### this must be uint8 or uint16!
    ## if the image is in 0..1, then:   image = (image*255.0).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) # 20.0 good, surprisingly!

    l_channel = clahe.apply(l_channel)

    merged_channels = cv2.merge((l_channel, a_channel, b_channel))

    image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)

    return l_channel






img = cv2.imread("huretina.jpg")

img = AHE(img)

plt.imshow(img, cmap='gray')
plt.show()

