# This is the main script that will run the DamageDetector

from ddam_vec import *
import glob
import matplotlib.pyplot as plt
import pickle

train_images_orig = glob.glob("NineSigma\\images-for-training\\*original.jpg")
train_images_anal = glob.glob("NineSigma\\images-for-training\\*analyzed_pixated.jpg")

test_images = glob.glob("NineSigma\\images-to-be-analyzed\\Image2.jpg")

# D = DamageDetector()
# D.classifier_type = 3
# D.train(train_images_orig, train_images_anal)
#
# for img_file in test_images:
#     img = plt.imread(img_file)
#     D.test(img)

with open('features.pkl', 'rb') as f:
    x, y = pickle.load(f)

x, y = downsample(x, y, threshold=0.5)
plot_features(x, y, 1, 7)
plt.legend(loc='best')
plt.show()