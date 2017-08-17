import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pickle
# import cv2


def extract_features(img, pixel, kernel_size=5):
    """Static function that returns a feature vector based on a given image (numpy array) and pixel (i, j)"""
    features = []

    nx, ny, nc = img.shape
    px, py = pixel
    dk = kernel_size >> 1    # This is the same as int(kernel_size/2)

    # Add Pixel based features
    for i in range(nc):
        features.append(img[px, py, i])     # Add basic RGB of pixel

    # Add Kernel based features as average
    rgb_average = [0]*nc
    n_average = 0
    for ii in range(px - dk, px + dk):
        for jj in range(py - dk, py + dk):
            if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                pass
            else:
                for kk in range(nc):
                    rgb_average[kk] += img[ii, jj, kk]
                n_average += 1

    for kk in range(nc):
        features.append(rgb_average[kk]/n_average)

    return features


def extract_label(analyzed, pixel):
    """Returns a label (integer) given an analyzed image and a pixel (i, j)
        analyzed image is assumed to be of shape (m x n x 3) containing RGB channels (uint8)
        Labels:
            0 : [Undamaged]     White   [R > 200 && G > 200 && B > 200]
            1 : [Crack]         Red     [R > 200 && G < 50  && B < 50 ]
            2 : [Rebar]         Green   [R < 50  && G > 200 && B < 50 ]
            3 : [Spall]         Blue    [R < 20  && G < 50  && B > 200]
    """
    r, g, b = analyzed[pixel[0], pixel[1], :]
    max_threshold = 200
    min_threshold = 50

    if r > max_threshold and g > max_threshold and b > max_threshold:
        return 0    # undamaged

    if r > max_threshold and g < min_threshold and b < min_threshold:
        return 1    # crack

    if r < min_threshold and g > max_threshold and b < min_threshold:
        return 2    # rebar

    if r < min_threshold and g < min_threshold and b > max_threshold:
        return 3    # spall

    return 0    # DEFAULT


########################################################################################################################
# ----------   D A M A G E    D E T E C T O R     C L A S S -------------
########################################################################################################################
class DamageDetector(object):
    def __init__(self, picklefile=None):
        """Damage detector class"""
        self.trained = False
        self.classifier = None
        self.C = 0.0001
        self.x_scaler = None

        # Load pickle file with training data if available
        if picklefile is not None:
            try:
                with open(picklefile, 'rb') as f:
                    data = pickle.load(f)
                self.classifier = data[0]
                self.C = data[1]
                self.x_scaler = data[2]
                self.trained = True
            except Exception as e:
                print("Error loading stored classifier from {}: {}".format(picklefile, e.message))

    def train(self, imgs_original, imgs_analyzed, picklefile="ddam.pkl"):
        """Trains the classifier and saves to picklefile"""

        # Make sure we have the same number of images
        n = len(imgs_original)
        assert(n == len(imgs_analyzed))

        all_features = []
        all_labels = []

        print("** TRAINING WITH SVM **")
        for i in range(n):
            original = plt.imread(imgs_original[i])
            analyzed = plt.imread(imgs_analyzed[i])
            print("{:3} Extracting features from {} ...".format(i, imgs_original[i]))

            # Make sure they are both of the same shape
            h1, w1, _ = original.shape
            h2, w2, _ = analyzed.shape
            assert (h1 == h2)
            assert (w1 == w2)

            for i in range(h1):
                for j in range(w1):
                    features = extract_features(original, (i, j))
                    label = extract_label(analyzed, (i, j))

                    all_features.append(features)
                    all_labels.append(label)

        # Store the features in Numpy array
        x = np.array(all_features, dtype=np.float64)
        y = np.array(all_labels, dtype=np.int)

        # Fit a per-column scaler and apply
        x_scaler = StandardScaler().fit(x)
        scaled_x = x_scaler.transform(x)

        # Split to training /validation sets
        rand_state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_x, y, test_size=0.2, random_state=rand_state)

        # Get lengths
        n_train = len(x_train)
        n_features = len(x_train[0])
        n_test = len(x_test)
        print("Num features         = {}".format(n_features))
        print("Num training samples = {}".format(n_train))
        print("Num test samples     = {}".format(n_test))

        # Set up the Support Vector Classifier
        self.classifier = LinearSVC(C=self.C)

        # Check the training time for SVC
        t = time.time()
        self.classifier.fit(x_train, y_train)
        duration = round(time.time() - t, 2)
        print("Training time        = {} seconds".format(duration))

        # Check performance on test set (Validation)
        accuracy = round(self.classifier.score(x_test, y_test), 4)
        print("Validation accuracy  = {} with C = {} ".format(accuracy, self.C))

        # Save to pickle file
        with open(picklefile, 'wb') as f:
            pickle.dump([self.classifier, self.C, self.x_scaler], f)
        print("Saved SVC to {}".format(picklefile))

        self.trained = True

        return

    def test(self, img):
        """Applies the classifier on the image provided. Returns an analyzed image of same
        dimensions as original with color labels described in function extract_label
        """

        # Make sure training has happened
        if not self.trained:
            raise ValueError("Classifier not yet trained! Do that first and then come back.")

        # Initialize analyzed
        h, w, nc = img.shape
        analyzed = 255*np.ones((h, w, 3), dtype='int8')      # Start as undamaged

        # Sweep through image
        t0 = time.time()
        n, n1, n2, n3, n4 = 0, 0, 0, 0, 0
        for i in range(h):
            for j in range(w):
                features = extract_features(img, (i, j))
                test_features = self.x_scaler.transform(features)
                prediction = self.classifier.predict(test_features)
                n += 1
                if prediction == 0:
                    analyzed[i, j, :] = [255, 255, 255]
                    n1 += 1

                elif prediction == 1:
                    analyzed[i, j, :] = [255, 0, 0]
                    n2 += 1

                elif prediction == 2:
                    analyzed[i, j, :] = [0, 255, 0]
                    n3 += 1

                elif prediction == 3:
                    analyzed[i, j, :] = [0, 255, 0]
                    n4 += 1

        duration = time.time() - t0

        # Print to screen
        print("RESULTS: ({:9.0f} secs".format(duration))
        print("TOTAL     = {}".format(n))
        print("UNDAMAGED = {}".format(n1))
        print("CRACK     = {}".format(n2))
        print("REBAR     = {}".format(n3))
        print("SPALL     = {}".format(n4))

        # Show test results
        plt.figure("Result")
        plt.subplot(1, 2, 1)
        plt.plot(img)
        plt.subplot(1, 2, 2)
        plt.plot(analyzed)
        plt.show()

        return


