# This is the main script that will run the DamageDetector

from ddam import *
import glob

train_images_orig = glob.glob("NineSigma\\images-for-training\\*original.jpg")
train_images_anal = glob.glob("NineSigma\\images-for-training\\*analyzed_pixated.jpg")

test_images = glob.glob("NineSigma\\images-to-be-analyzed\\*.jpg")

D = DamageDetector()
D.train(train_images_orig, train_images_anal)

# for img in test_images:
#     D.test(img)

