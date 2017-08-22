# This is the main script that will run the DamageDetector

from ddam_nb import *
import glob

train_images_orig = glob.glob("NineSigma\\images-for-training\\*original.jpg")
train_images_anal = glob.glob("NineSigma\\images-for-training\\*analyzed_pixated.jpg")

test_images = glob.glob("NineSigma\\images-to-be-analyzed\\Image2.jpg")

D = DamageDetector()
D.classifier_type = 4
D.train(train_images_orig, train_images_anal, saved_features='features.pkl')

for img_file in test_images:
    img = plt.imread(img_file)
    D.test(img)

