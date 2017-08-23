import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pickle
import cv2

DEBUG = True

def plot_features(x, y, feat1, feat2):
    i0 = np.where(y == 0)
    i1 = np.where(y == 1)
    i2 = np.where(y == 2)
    i3 = np.where(y == 3)

    idxs = [i0, i3, i1, i2]
    cols = 'mrgb'
    markers = '.^o.'
    labels = ['Undamaged', 'Spall', 'Rebar', 'Crack']
    feats = ['Y', 'Cr', 'Cb', 'Ravg-9','Gavg-9','Bavg-9', 'Gray', 'Laplacian', 'SobelX', 'SobelY']

    for i, idx in enumerate(idxs):
        if len(idx[0]) == 0:
            continue

        xdata = x[idx[0], feat1]
        ydata = x[idx[0], feat2]
        plt.plot(xdata, ydata, cols[i]+markers[i], label=labels[i])
        plt.xlabel(feats[feat1])
        plt.ylabel(feats[feat2])



def downsample(x, y, threshold=0.7):
    """Returns features and labels after sampling down labels of class 0"""
    i0 = np.where(y == 0)
    i1 = np.where(y == 1)
    i2 = np.where(y == 2)
    i3 = np.where(y == 3)
    n = len(i0[0])
    n_sub = int(threshold*n)
    print("Subsampling {} of {}".format(n_sub, n))
    sub = resample(np.arange(n), replace=False, n_samples=n_sub)

    inew = np.hstack((i0[0][sub], i1[0], i2[0], i3[0]))
    print (inew.shape)
    xnew = x[inew, :]
    ynew = y[inew]

    return xnew, ynew


def extract_features_vectorized(img, kernel_size=11):
    """Vectorized implementation that provides the entire feature vector
    for a whole image
    """

    img_csp = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # Kernel Averaging Filter
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size*kernel_size)
    img_blur = cv2.filter2D(img, -1, kernel)

    # Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Laplacian/Sobel
    img_laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    img_sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    img_sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Stack all of the layers
    img_feat = np.dstack((img_csp, img_blur, img_gray, img_laplacian, img_sobelx, img_sobely))

    # Reshape
    n_features = img_feat.shape[2]
    features = img_feat.T.reshape(n_features, -1)

    return features.T


def extract_labels_vectorized(img):
    """Vectorized implementation of label extraction
            0 : [Undamaged]     White   [R > 200 && G > 200 && B > 200]
            1 : [Rebar]         Red     [R > 200 && G < 50  && B < 50 ]
            2 : [Crack]         Green   [R < 50  && G > 200 && B < 50 ]
            3 : [Spall]         Blue    [R < 20  && G < 50  && B > 200]
    """

    r = img[:, :, 0].reshape(1, -1)
    g = img[:, :, 1].reshape(1, -1)
    b = img[:, :, 2].reshape(1, -1)

    max_threshold = 200

    labels = np.zeros(r.shape, dtype=np.int)

    labels[r > max_threshold] = 1
    labels[g > max_threshold] = 2
    labels[b > max_threshold] = 3

    labels[(r > max_threshold) & (g > max_threshold) & (b > max_threshold)] = 0

    if DEBUG:
        n1 = np.count_nonzero(labels == 0)
        n2 = np.count_nonzero(labels == 1)
        n3 = np.count_nonzero(labels == 2)
        n4 = np.count_nonzero(labels == 3)
        print("Number of undamaged = ", n1)
        print("Number of rebar     = ", n2)
        print("Number of crack     = ", n3)
        print("Number of spall     = ", n4)

    return labels.T


def extract_features(img, pixel, kernel_size=11):
    """Static function that returns a feature vector based on a given image (numpy array) and pixel (i, j)"""
    features = []

    nx, ny, nc = img.shape
    px, py = pixel
    dk = kernel_size >> 1    # This is the same as int(kernel_size/2)

    # Add Pixel based features
    for i in range(nc):
        features.append(img[px, py, i])     # Add basic RGB of pixel

    # Add kernel based average to features
    rgb_average = [0]*nc
    n_average = 0
    # rgb_var = np.empty([kernel_size, kernel_size, nc])
    redarr, grnarr, bluarr = [], [], []
    for ii in range(px - dk, px + dk):
        for jj in range(py - dk, py + dk):
            if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                pass
            else:
                for kk in range(nc):
                    rgb_average[kk] += img[ii, jj, kk]

                redarr.append(img[ii, jj, 0])
                grnarr.append(img[ii, jj, 1])
                bluarr.append(img[ii, jj, 2])
                n_average += 1

    for kk in range(nc):
        features.append(rgb_average[kk]/n_average)

    # Add intensity/average_intensity to features
    for kk in range(nc):
        features.append(img[px, py, kk]/rgb_average[kk]*n_average)

    # Add kernel based variance to features
    red_var = np.var(redarr)
    grn_var = np.var(grnarr)
    blu_var = np.var(grnarr)

    features.extend([red_var, grn_var, blu_var])

    return features


def extract_label(analyzed, pixel):
    """Returns a label (integer) given an analyzed image and a pixel (i, j)
        analyzed image is assumed to be of shape (m x n x 3) containing RGB channels (uint8)
        Labels:
            0 : [Undamaged]     White   [R > 200 && G > 200 && B > 200]
            1 : [Rebar]         Red     [R > 200 && G < 50  && B < 50 ]
            2 : [Crack]         Green   [R < 50  && G > 200 && B < 50 ]
            3 : [Spall]         Blue    [R < 20  && G < 50  && B > 200]
    """
    r, g, b = analyzed[pixel[0], pixel[1], :]
    max_threshold = 200
    min_threshold = 50

    if r > max_threshold and g > max_threshold and b > max_threshold:
        return 0    # undamaged

    if r > max_threshold and g < min_threshold and b < min_threshold:
        return 1    # rebar

    if r < min_threshold and g > max_threshold and b < min_threshold:
        return 2    # crack

    if r < min_threshold and g < min_threshold and b > max_threshold:
        return 3    # spall

    return 0    # DEFAULT


########################################################################################################################
# ----------   D A M A G E    D E T E C T O R     C L A S S -------------
########################################################################################################################
class DamageDetector(object):
    def __init__(self, picklefile=None):
        """Damage detector class"""
        self.classifier_type = 1   # 0=GaussianNaiveBayes, 1=LinearSVM, 2=SVC(rbf), 3=DecisionTree, 4=RandomForest
        self.trained = False
        self.classifier = None
        self.C = 1.0
        self.gamma = 'auto'
        self.n_estimators = 10
        self.x_scaler = None
        self.threshold = 0.7

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

    def train(self, imgs_original, imgs_analyzed, picklefile="ddam.pkl", saved_features=None):
        """Trains the classifier and saves to picklefile"""

        if saved_features is None:
            # Make sure we have the same number of images
            n = len(imgs_original)
            assert(n == len(imgs_analyzed))

            all_features = []
            all_labels = []

            print("** TRAINING CLASSIFIER **")
            for ii in range(n):
                original = plt.imread(imgs_original[ii])
                analyzed = plt.imread(imgs_analyzed[ii])
                print("{:3} Extracting features from {} ...".format(ii, imgs_original[ii]))

                # Make sure they are both of the same shape
                h1, w1, _ = original.shape
                h2, w2, _ = analyzed.shape
                assert (h1 == h2)
                assert (w1 == w2)

                features = extract_features_vectorized(original, kernel_size=9)
                labels = extract_labels_vectorized(analyzed)
                features, labels = downsample(features, labels, self.threshold)
                all_features.append(features)
                all_labels.append(labels)

            features = np.vstack(all_features)
            labels = np.vstack(all_labels).ravel()

            # Store the features in Numpy array
            x = np.array(features, dtype=np.float64)
            y = np.array(labels, dtype=np.int)
            print("Features array shape = ", x.shape)
            print("Labels   array shape = ", y.shape)
            print("Writing features to features.pkl...")
            with open('features.pkl', 'wb') as f:
                pickle.dump([x, y], f)

        else:
            with open(saved_features, 'rb') as f:
                data = pickle.load(f)
                x = data[0]
                y = data[1]

        # Fit a per-column scaler and apply
        print("Normalizing features using StandardScaler...")
        x_scaler = StandardScaler().fit(x)
        self.x_scaler = x_scaler
        scaled_x = x_scaler.transform(x)

        # Split to training /validation sets
        print("Splitting to training/validation sets...")
        rand_state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_x, y, test_size=0.2, random_state=rand_state)

        # Get lengths
        n_train = len(x_train)
        n_features = len(x_train[0])
        n_test = len(x_test)
        print("Num features               = {}".format(n_features))
        print("Num training samples       = {}".format(n_train))
        print("Num validation samples     = {}".format(n_test))

        # Set up the Support Vector Classifier
        if self.classifier_type == 0:
            self.classifier = GaussianNB()
            print("Gaussian Naive Bayes")
        elif self.classifier_type == 1:
            self.classifier = LinearSVC(C=self.C)
            print("Linear SVC")
        elif self.classifier_type == 2:
            self.classifier = SVC(C=self.C, gamma=self.gamma)
            print("Nonlinear (RBF) SVC")
        elif self.classifier_type == 3:
            self.classifier = DecisionTreeClassifier()
            print("Decision Tree")
        elif self.classifier_type == 4:
            self.classifier = RandomForestClassifier(n_estimators=self.n_estimators)
            print("Random Forest")

        # Check the training time for SVC
        print("Began training...")
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
        print("Saved Classifier to {}".format(picklefile))

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
        w, h, nc = img.shape
        analyzed = 255*np.ones((w, h, 3), dtype='uint8')      # Start as undamaged

        # Sweep through image
        t0 = time.time()

        features = extract_features_vectorized(img, kernel_size=9)
        test_features = self.x_scaler.transform(features)
        predictions = self.classifier.predict(test_features).reshape((h, w)).transpose()

        i1, j1 = np.where(predictions == 1)
        analyzed[i1, j1, 1] = 0
        analyzed[i1, j1, 2] = 0

        i2, j2 = np.where(predictions == 2)
        analyzed[i2, j2, 0] = 0
        analyzed[i2, j2, 2] = 0

        i3, j3 = np.where(predictions == 3)
        analyzed[i3, j3, 0] = 0
        analyzed[i3, j3, 1] = 0

        n, n2, n3, n4 = h*w, i1.shape[0], i2.shape[0], i3.shape[0]
        n1 = n - n2 - n3 - n4

        duration = int(time.time() - t0)

        # Print to screen
        print("RESULTS: {:} secs".format(duration))
        print("TOTAL     = {}".format(n))
        print("UNDAMAGED = {}".format(n1))
        print("REBAR     = {}".format(n2))
        print("CRACK     = {}".format(n3))
        print("SPALL     = {}".format(n4))

        # Show test results
        plt.figure("Result")
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(analyzed)
        plt.show()

        return
