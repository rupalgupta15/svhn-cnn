import cv2
import numpy as np
import scipy.io as scio
import h5py
import image_pyramid as ip
import os
import json

# For reproducability :
# np.random.seed(23)


TRAIN_FILE = "train_32x32.mat"
TEST_FILE = "test_32x32.mat"
OUTPUT_DIR = "output"
FULL_TRAIN_FILE = "train_full_data"
FULL_TEST_FILE = "test_full_data"


# Note: this method only works for arrays of arrays like Xtrain, Xtest
def normalize_images(images, labels):  # images are the xtrain kind of values and labels are y values
    def rgb2gray(images):
        grayscale = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray[..., np.newaxis]
            grayscale.append(gray)
        return np.array(grayscale)
    gray_images = rgb2gray(images)
    images_mean = np.mean(gray_images, axis=(1, 2))
    images_std = np.std(gray_images, axis=(1,2))
    def norm(images, img_mean, img_std, labels):
        normalized = []
        for i, img in enumerate(images):
            if img_std[i] == 0:  # if std of image is 0, skip it  -> only 1 such image
                if labels.ndim > 1 and labels.shape[1] > 1:
                    labels = np.delete(labels, i, axis=0)
                else:
                    labels = np.delete(labels, i)
                continue
            normalized.append((img - img_mean[i]) / img_std[i])
        return np.array(normalized), labels
    final_norm_gray_img, labels = norm(gray_images, images_mean, images_std, labels)
    return final_norm_gray_img, labels


def normalize_color_images(images, labels):  # images are the xtrain kind of values and labels are y values
    # Compute channel wise mean and std dev
    red_mean, green_mean, blue_mean = [], [], []
    red_std, green_std, blue_std = [], [], []
    for image in images:
        red_val = np.reshape(image[:,:,0], -1)
        green_val = np.reshape(image[:,:,1], -1)
        blue_val = np.reshape(image[:,:,2], -1)
        red_mean.append(np.mean(red_val))
        green_mean.append(np.mean(green_val))
        blue_mean.append(np.mean(blue_val))
        red_std.append(np.std(red_val))
        green_std.append(np.std(green_val))
        blue_std.append(np.std(blue_val))
    mean_arr = np.stack((red_mean, green_mean, blue_mean), axis=1)
    # print('mean_arr', len(mean_arr), mean_arr, mean_arr.shape)
    std_arr = np.stack((red_std, green_std, blue_std), axis=1)
    # images_mean = np.mean(gray_images, axis=(1, 2))
    # images_std = np.std(gray_images, axis=(1,2))
    def norm(images, img_mean, img_std, labels):
        normalized = []
        for i, img in enumerate(images):
            if img_std[i].all() == 0:  # if std of image is 0, skip it  -> only 1 such image
                if labels.shape[1] > 1:
                    labels = np.delete(labels, i, axis=0)
                else:
                    labels = np.delete(labels, i)
                continue
            normalized.append((img - img_mean[i]) / img_std[i])
        return np.array(normalized), labels
    final_norm_gray_img, labels = norm(images, mean_arr, std_arr, labels)
    return final_norm_gray_img, labels


def normalize_single_test_img(image):
    def rgb2gray(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray[..., np.newaxis]
        return np.array(gray)
    gray_image = rgb2gray(image)
    image_mean = np.mean(gray_image)
    image_std = np.std(gray_image)
    if image_std == 0:
        image_std = 0.001
    normalized = ((gray_image - image_mean) / image_std)
    return normalized


def load_more_negative_images_64x64():
    f = h5py.File("all_negatives_64x64.h5", 'r')
    key = list(f.keys())[0]
    # print(f.keys(), f[key], f[key].shape)
    return f[key]


# No need to use this method when training and testing on full data:
# this method will have only 32x32 images
# loads data from mat file and also normalizes it
def prepare_data_from_mat():  # Not used anymore
    svhn_train_data = scio.loadmat(TRAIN_FILE)  # for reading mat data
    svhn_test_data = scio.loadmat(TEST_FILE)  # for reading mat data
    #  gives data in form of dictionary
    print('Keys of dict', svhn_train_data.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
    Xtrain = svhn_train_data['X']
    Ytrain = svhn_train_data['y']
    Ytrain[Ytrain == 10] = 0

    Xtest = svhn_test_data['X']
    Ytest = svhn_test_data['y']
    Ytest[Ytest == 10] = 0
    # print('Shape of Xtrain', Xtrain.shape)  # (32, 32, 3, 26032)  height, width, channels, no of images
    # print('Shape of Xtrain[0]',type(Xtrain), Xtrain[0].shape, Xtrain[0])
    # print('Shape of Ytrain', Ytrain)
    # For this, we need to reshape Xtrain such that its shape is (26032, 32, 32, 3) without affecting contents
    Xtrain = Xtrain.transpose(3, 0, 1, 2)  # np.transpose(x, (1, 0, 2)).shape
    Ytrain = Ytrain.ravel()
    Xtest = Xtest.transpose(3, 0, 1, 2)  # np.transpose(x, (1, 0, 2)).shape
    Ytest = Ytest.ravel()
    # Append negative images using more data:
    negatives = load_more_negative_images_64x64()

    # To be able to send this data to cropped images recognizer, reshape to 32x32
    if negatives.shape[1]!=32 or negatives.shape[2]!=32:
        new_negs = []
        for img in negatives:
            resized = cv2.resize(img, (32, 32))
            new_negs.append(resized)
        negatives = np.asarray(new_negs)
    train_idx = int(0.8*len(negatives))
    Xtrain = np.append(Xtrain, negatives[:train_idx], axis=0)  # adding negative images to data
    Xtest = np.append(Xtest, negatives[train_idx:], axis=0)
    Ytrain_neg = np.ones(negatives.shape[0], dtype=int)
    Ytrain_neg.fill(10)
    Ytrain = np.append(Ytrain, Ytrain_neg[:train_idx], axis=0)
    Ytest = np.append(Ytest, Ytrain_neg[train_idx:], axis=0)
    train_gray_norm, Ytrain = normalize_images(Xtrain, Ytrain)
    test_gray_norm, Ytest = normalize_images(Xtest, Ytest)
    # finally xtrain and ytrain are normalized images:
    Xtrain = train_gray_norm
    Xtest = test_gray_norm
    return Xtrain, Ytrain, Xtest, Ytest


def sliding_window(image, steps, win_size):
    for y in range(0, image.shape[0], steps):
        for x in range(0, image.shape[1], steps):
            yield (x, y, image[y:y + win_size[1], x:x + win_size[0]])


def read_test_real_data():
    Xdata = []
    Ydata = []
    with open('test_data.json') as f:
        data = json.load(f)
        data = data['results']
        for d in data:
            filename = d['filename']
            # if len(d['boxes']) != 1:
            #     continue
            if filename == '29930.png':  # has 6 digits - ignore
                continue
            image = cv2.imread(os.path.join(FULL_TEST_FILE, filename))
            h, w, _ = image.shape
            Xdata.append(image)
            bbox = d['boxes'][0]
            Ydata.append(bbox['label'])
    Xdata = np.array(Xdata)
    Ydata = np.array(Ydata)

    return Xdata, Ydata


def read_bounding_box_data(file, img_dir, size):  # file should be in form of json, img_dir should be FULL DIR
    Xdata = []
    Ydata = []
    # bbox_data = []  # contains one bbox for all digits
    with open(file) as f:  # 'test_data.json'
        data = json.load(f)
        data = data['results']
        # print(len(data))  # 33402
        for d in data:
            filename = d['filename']
            # if len(d['boxes']) != 1:
            #     continue
            if filename == '29930.png':  # has 6 digits - ignore 1 file
                continue
            bbox = d['boxes']
            top = [d['top'] for d in bbox]  # y
            left = [d['left'] for d in bbox]  # x
            width = [d['width'] for d in bbox]  # w
            height = [d['height'] for d in bbox]  # h
            labels = [int(d['label']) if int(d['label']) != 10 else 0 for d in bbox]  # replacing 10 with 0
            final_top, final_left = int(np.min(top)), np.max([0, int(np.min(left))])
            final_height, final_width = np.max(height), np.sum(width)  # np.max(0, np.int(np.sum(left)))
            max_bot = int(final_top + final_height)  # rightmost bottom point y
            max_right = int(final_left + final_width)
            image = cv2.imread(os.path.join(img_dir, filename))  # FULL_TEST_FILE
            h, w, _ = image.shape
            final_img = image[final_top:max_bot, final_left:max_right, :]
            final_img = cv2.resize(final_img, (size, size))
            Xdata.append(final_img)
            Ydata.append(labels)
    Xdata = np.array(Xdata)
    Ydata = np.array(Ydata)
    return Xdata, Ydata


def load_mean():
    # f = h5py.File("train_img_mean.h5", 'r')
    f = h5py.File("training_mean_std_info/cnn1_train_img_mean.h5", 'r')
    key = list(f.keys())[0]
    # print(f.keys(), f[key], f[key].shape)
    return f[key]


def load_std():
    # f = h5py.File("train_img_std.h5", 'r')
    f = h5py.File("training_mean_std_info/cnn1_train_img_std.h5", 'r')
    key = list(f.keys())[0]
    # print(f.keys(), f[key], f[key].shape)
    return f[key]


def normalize_single_img(image):
    def rgb2gray(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray[..., np.newaxis]
        return np.array(gray)
    gray_image = rgb2gray(image)
    image_mean = np.mean(gray_image)
    image_std = np.std(gray_image)
    if image_std == 0:
        image_std = 0.001
    normalized = ((gray_image - image_mean) / image_std)
    return normalized


def normalize_real_image(image):  # images are the xtrain kind of values and labels are y values
    def rgb2gray(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray[..., np.newaxis]
        return np.array(gray)
    gray_images = rgb2gray(image)
    images_mean = load_mean()
    images_std = load_std()
    norm_image = (gray_images - images_mean) / (images_std)
    return norm_image


# Reference: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def nms(boxes, thresh):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype("float")
    final_box = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    index = np.argsort(y2)
    while len(index) > 0:
        last_box = len(index) - 1
        i = index[last_box]
        final_box.append(i)
        x1_p, y1_p, x2_p, y2_p = x1[index[:last_box]], y1[index[:last_box]], x2[index[:last_box]], y2[index[:last_box]]
        main_x1, main_y1 = np.maximum(x1[i], x1_p), np.maximum(y1[i], y1_p)
        main_x2, main_y2 = np.minimum(x2[i], x2_p), np.minimum(y2[i], y2_p)
        w, h = np.maximum(0, main_x2 - main_x1 + 1), np.maximum(0, main_y2 - main_y1 + 1)
        common = (w * h) / area[index[:last_box]]
        indices = np.where(common > thresh)[0]
        index = np.delete(index, np.concatenate(([last_box], indices)))
    return boxes[final_box].astype("int")


# only called in CNN1
def generate_window(img_name, dig_recogizer=False, filename=None):
    if dig_recogizer:  # If called for the digit recognizer cnn
        win_w, win_h = (32, 32)
    else:
        win_w, win_h = (100, 64)  # (74, 64), (128, 64)
    if filename is not None:
        incoming_img = cv2.imread(os.path.join(filename, img_name))
    else:
        # # Xdata, Ydata = read_test_real_data()
        # # gray_norm_data, label = normalize_images(Xdata, Ydata)
        incoming_img = img_name
    c = 0
    multiple_windows = []
    bbox = []
    orig = incoming_img.copy()
    pyr_lvl = 0
    num_pyramids = []
    for img in ip.pyramid(incoming_img):
        # earlier steps = 20
        steps = 12  # steps = 12 , 6 # for 4: 6
        # if pyr_lvl > 0:  # >=3
        #     steps = 8  # steps = 6  # for 4: 8
        if pyr_lvl > 2:  # >=3
            steps = 6  # steps = 6  # for 4: 4
        if pyr_lvl > 5:
            steps = 4
        for (x, y, window) in sliding_window(img, steps=steps, win_size=(win_w, win_h)):  # 4 - 4199, 12 steps - c = 448
            if window.shape[0] != win_h or window.shape[1]!=win_w:
                continue
            img_copy = img.copy()
            cropped = img_copy[y:y+win_h, x:x+win_w]
            box = np.array([x, y, x+win_w, y+win_h])
            cropped = cv2.resize(cropped, (64, 64))
            bbox.append(box)
            normalized = normalize_real_image(cropped)
            num_pyramids.append(pyr_lvl)
            multiple_windows.append(normalized)
            c += 1
        pyr_lvl += 1
    # print('sliding window generated', c)
    return multiple_windows, bbox, incoming_img, num_pyramids


# Reusing code from PS3
def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None
