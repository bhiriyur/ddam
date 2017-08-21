# This is the main script that will run the DamageDetector

from ddam import *
import glob

train_images_orig = glob.glob("NineSigma\\images-for-training\\*original.jpg")
train_images_anal = glob.glob("NineSigma\\images-for-training\\*analyzed_pixated.jpg")

test_images = glob.glob("NineSigma\\images-to-be-analyzed\\Image2.jpg")

D = DamageDetector("ddam.pkl")
# D.train(train_images_orig, train_images_anal)

for img_file in test_images:
    img = plt.imread(img_file)
    D.test(img)

