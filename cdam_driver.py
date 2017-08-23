from cdam import extract_cracks
import glob
import matplotlib.pyplot as plt
import cv2

train_images_orig = "NineSigma\\images-for-training\\05 original.jpg"
train_images_anal = "NineSigma\\images-for-training\\05 analyzed_pixated.jpg"


imgo = plt.imread(train_images_orig)
imga = plt.imread(train_images_anal)

# plt.imshow(imga)
# plt.show()

ksize = 50

h, w, nc = imgo.shape

nx = int(h / ksize)
ny = int(w / ksize)

for i in range(nx):
    xbeg = i * ksize
    xend = xbeg + ksize
    for j in range(ny):
        ybeg = j * ksize
        yend = ybeg + ksize

        sub_img = imga[xbeg:xend, ybeg:yend, :]
        crack = extract_cracks(sub_img)

        # cv2.imshow('sub',sub_img[:,:,0])
        # cv2.waitKey(1)

        if crack > 0.:
            print("Detected {}: From {} {} to {} {}".format(crack, xbeg, ybeg, xend, yend))
            imgo = cv2.rectangle(imgo, (ybeg, xbeg), (yend, xend), color=(255, 0, 0), thickness=2)

plt.imshow(imgo)
plt.show()