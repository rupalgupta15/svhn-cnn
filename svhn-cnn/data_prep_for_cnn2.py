import numpy as np
import cv2
import h5py
from helpers import read_bounding_box_data

FULL_TRAIN_FILE = "train_full_data"
FULL_TEST_FILE = "test_full_data"


# to normalize the images
def normalize_images(images):  # images are the xtrain kind of values and labels are y values
    def rgb2gray(images):
        grayscale = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray[..., np.newaxis]
            grayscale.append(gray)
        return np.array(grayscale)
    gray_images = rgb2gray(images)
    # images_mean = np.mean(gray_images, axis=(1, 2))
    # images_std = np.std(gray_images, axis=(1, 2))
    images_mean = np.mean(gray_images, axis=0)
    images_std = np.std(gray_images, axis=0)
    save_mean_std(images_mean, 'training_mean_std_info/cnn2_train_img_mean.h5')
    save_mean_std(images_std, 'training_mean_std_info/cnn2_train_img_std.h5')
    norm_images = (gray_images - images_mean) /(images_std)
    return norm_images


def test_normalize_images(images):  # images are the xtrain kind of values and labels are y values
    def rgb2gray(images):
        grayscale = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray[..., np.newaxis]
            grayscale.append(gray)
        return np.array(grayscale)
    gray_images = rgb2gray(images)
    # images_mean = np.mean(gray_images, axis=(1, 2))
    # images_std = np.std(gray_images, axis=(1, 2))
    images_mean = load_mean()
    images_std = load_std()
    norm_images = (gray_images - images_mean) /(images_std)
    return norm_images


# method to create new y labels
def create_y_labels(num_y_entries, Yorig):  # Yorig are the labels coming from actual data - no negative images here
    # developing Ytrain:
    new_y_labels = []
    y_neg = [0, 10, 10, 10, 10, 10]
    # fill all y with this value
    for x in range(0, num_y_entries):
        final_y = np.copy(np.array(y_neg))
        new_y_labels.append(final_y)
    new_y_labels = np.array(new_y_labels)
    final_Y_labels = []
    for x in range(0, len(Yorig)):
        new_label = np.copy(new_y_labels[x])
        num_digits = len(Yorig[x])
        new_label[0] = num_digits
        for j in range(0, num_digits):
            non_neg_label = Yorig[x][j]
            new_label[j + 1] = non_neg_label
        generated_label = np.asarray(new_label)
        final_Y_labels.append(generated_label)
    final_Y_labels = np.array(final_Y_labels)
    new_y_labels[:len(Yorig)] = final_Y_labels
    new_y_labels = np.array(new_y_labels)
    return new_y_labels


def load_more_negative_images_64x64():
    f = h5py.File("more_negative_images_64x64.h5", 'r')
    key = list(f.keys())[0]
    # print(f.keys(), f[key], f[key].shape)
    return f[key]


def get_negative_data():
    negatives = load_more_negative_images_64x64()  # load more negative data
    return negatives


def neg_split(negatives):
    train_neg_idx = int(0.8 * len(negatives))
    return train_neg_idx


# method to train model
def data_prep():  # only for training
    print('------------------ Loading Data ------------------')
    negatives = get_negative_data()
    train_neg_idx = neg_split(negatives)
    # negatives = load_more_negative_images_64x64()  # load more negative data
    Xtrain, Ytrain = read_bounding_box_data('train_data.json', FULL_TRAIN_FILE, 64)  # changed size in helpers
    # print('neg shape', negatives.shape)  # (34176, 64, 64, 3) or 34176
    # print('Xtrain and Ytrain initial shape', Xtrain.shape, Ytrain.shape)  # (33401, 32, 32, 3) (33401,)
    # train_neg_idx = int(0.8*len(negatives))  # no of negatives to be added to training data
    # train_neg_idx = np.random.randint(0, negatives.shape[0], train_neg_no)
    Xtrain = np.append(Xtrain, negatives[:train_neg_idx], axis=0)  # adding negative images to data
    new_neg_y_idx = negatives[:train_neg_idx].shape[0]
    total_y_idx = len(Ytrain) + new_neg_y_idx
    new_y_labels = create_y_labels(total_y_idx, Ytrain)
    Ydata_norm = new_y_labels
    # normalize X and Y at the end
    print('************* returning normalized data ***************')
    Xdata_norm = normalize_images(Xtrain)
    # train_model(Xdata_norm, Ydata_norm)  # only uncomment if you need to train the model
    return Xdata_norm, Ydata_norm


def save_mean_std(data, filename):
    file = h5py.File(filename, 'w')
    file.create_dataset('dataset_name', data=data)
    print('Data saved mean and std.h5')


def load_mean():
    f = h5py.File("training_mean_std_info/train_img_mean.h5", 'r')
    key = list(f.keys())[0]
    # print(f.keys(), f[key], f[key].shape)
    return f[key]


def load_std():
    f = h5py.File("training_mean_std_info/train_img_std.h5", 'r')
    key = list(f.keys())[0]
    print(f.keys(), f[key], f[key].shape)
    return f[key]


def generate_test_data(negatives, train_idx):
    Xtest, Ytest = read_bounding_box_data('test_data.json', FULL_TEST_FILE, 64)
    Xtest = np.append(Xtest, negatives[train_idx:], axis=0)  # adding negative images to data
    test_neg_y_idx = negatives[train_idx:].shape[0]
    total_test_y_idx = len(Ytest) + test_neg_y_idx
    new_test_y_labels = create_y_labels(total_test_y_idx, Ytest)
    test_Ydata_norm = new_test_y_labels
    test_Xdata_norm = test_normalize_images(Xtest)
    return test_Xdata_norm, test_Ydata_norm


def fetch_data():
    negatives = get_negative_data()
    train_neg_idx = neg_split(negatives)
    # Xdata_norm, Ydata_norm = generate_training_data(negatives, train_idx)
    # creating Xtest and Ytest in a similar manner
    test_Xdata_norm, test_Ydata_norm = generate_test_data(negatives, train_neg_idx)  # don't read test data now
    return test_Xdata_norm, test_Ydata_norm
