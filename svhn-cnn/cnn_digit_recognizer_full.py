# Detects if there is a digit in image:

import tensorflow as tf
import cv2
import numpy as np

# File to train the network for CNN
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from helpers import generate_window, nms
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from data_prep_cnn1 import data_prep, fetch_data
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score

# Using only 32x32 cropped images
# TRAIN_FILE = "train_32x32.mat"
# TEST_FILE = "test_32x32.mat"

FULL_TEST_FILE = "test_full_data"
FULL_TRAIN_FILE = "train_full_data"
SCALE = 1.75  # 1.5


def build_model(load_weights_file=None):
    input = Input(shape=(64, 64, 1))

    model = Conv2D(16, (3, 3), padding='same', activation=tf.nn.relu)(input)
    model = Conv2D(16, (3, 3), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    model = Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu)(model)
    model = Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    model = Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu)(model)
    model = Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    model = Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu)(model)
    model = Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    # old model:
    # model = Conv2D(32, 3, padding='same', activation=tf.nn.relu)(input)
    # model = BatchNormalization()(model)
    # model = MaxPooling2D()(model)
    # model = Conv2D(64, 3, padding='same', activation=tf.nn.relu)(model)
    # model = BatchNormalization()(model)
    # model = MaxPooling2D()(model)
    # model = Conv2D(128, 3, padding='same', activation=tf.nn.relu)(model)
    # model = BatchNormalization()(model)
    # model = MaxPooling2D()(model)
    model = Flatten()(model)
    # model = Dense(64, activation=tf.nn.relu)(model)
    # # model = Dense(128, activation=tf.nn.relu)(model) # Extra for v2
    # output = Dense(2, activation=tf.nn.softmax)(model)

    model = Dense(1024, activation=tf.nn.relu)(model)
    model = Dense(512, activation=tf.nn.relu)(model)
    model = Dropout(0.5)(model)

    output = Dense(2, activation=tf.nn.softmax)(model)
    model = Model(inputs=input, outputs=output)
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu, input_shape=(32,32,1)))
    # # # model.add(tf.keras.layers.MaxPooling2D())
    # model.add(tf.keras.layers.Flatten())  # added for CNN
    # model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))  # 11 to include 10

    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # binary_entropy - dense 1
    if load_weights_file is not None:
        print("Loading Weights")
        model.load_weights(load_weights_file)
    model.summary()
    return model


def train_model(Xtrain, Ytrain, Xtest, Ytest):
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                rotation_range=30, width_shift_range=0.5, height_shift_range=0.5)
    datagen.fit(Xtrain)
    model = build_model()
    checkpoint = ModelCheckpoint(filepath='models/digit_recognizer_full-{epoch:03d}.h5', monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True, verbose=2)
    early_stopping = EarlyStopping(monitor= 'loss',  min_delta=0.000001, patience=5, verbose=1, mode='auto')
    callbacks = [checkpoint, early_stopping]
    # cnn_digit_history = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=15\
    #                               , verbose=2, callbacks=callbacks)
    model.fit_generator(datagen.flow(Xtrain, Ytrain, batch_size=64), steps_per_epoch=len(Xtrain) / 64,
                                     epochs=50, verbose=1, validation_data=(Xtest, Ytest), callbacks=callbacks)
    #  steps_per_epoch=len(Xtrain) / 32,
    build_confusion_matrix(Xtest, Ytest, model)


def training():
    Xdata_norm, Ydata_norm = data_prep()
    Xtest_norm, Ytest_norm = fetch_data()
    train_model(Xdata_norm, Ydata_norm, Xtest_norm, Ytest_norm)


def build_confusion_matrix(Xtest, Ytest, model):
    actual_labels = np.argmax(model.predict(Xtest), axis=1)
    expected_labels = Ytest
    c = confusion_matrix(actual_labels, expected_labels)
    print('confusion', c)


def test_model_real_images(all_windows, model=None):
    # building model and loading it:
    if model:
        model = model
    else:
        model = build_model("./models/detection/digit_recognizer_full-025.h5")
        print("Model Loaded...")
    # print('img shape', test_Xdata_norm[192,:,:,:].shape)  # 192 is for 193.png
    # print('CNN 1 all windows shape', len(all_windows), all_windows.shape)  # (156, 64, 64, 1)

    if len(all_windows.shape) == 4:
        if all_windows.shape[1] != 64 or all_windows.shape[2] != 64:
            new_window = []
            for img in all_windows:
                new = cv2.resize(img, (64, 64))
                new_window.append(new[..., np.newaxis])
            all_windows = np.asarray(new_window)

    prediction = model.predict(all_windows)  # [192:193,:,:,:]
    num_dig = np.argmax(prediction, axis=1)  # num digits [147  78 191  32   4  50]
    num_digits_non_zero = num_dig[np.where(num_dig>0)]
    non_zero_idx = np.where(num_dig>0)  # replaced 0 with 2 to check if it is even able to detect
    counts = np.bincount(num_digits_non_zero)
    # print('counts', counts)

    # prediction at the non zero indexes:
    non_zero_pred = prediction[non_zero_idx,:]
    cross_thresh = np.where(non_zero_pred>0.88)  # 0.6
    # print('cross_thresh', len(cross_thresh), cross_thresh)  # cross_thresh[1] has the main values
    # print('non zero and cross thresh', non_zero_idx, cross_thresh[1])
    cross_thresh_new = np.asarray(cross_thresh[1])
    final_non_zero_idx = np.take(non_zero_idx, cross_thresh_new)
    # print('final non zero ids', np.take(non_zero_idx, cross_thresh_new))
    return final_non_zero_idx, prediction  # no thresholding only returning non zero index


def final_method_cnn(img_to_test, video=False, model_cnn_1=None):
    if video == False:
        filename = 'input_images'  # 'my_images/my' , my_images
    else:
        filename = None
    # checking sliding window
    # all_windows, bboxes = generate_window('crop_scn2.png', dig_recogizer=True, filename=filename)
    all_windows, bboxes, incoming_img, num_pyramids = generate_window(img_to_test, dig_recogizer=False, filename=filename)
    all_windows = np.asarray(all_windows)
    bboxes = np.asarray(bboxes)
    indices, prediction = test_model_real_images(all_windows, model_cnn_1)
    final_box = bboxes[indices, :]
    num_pyramids = np.asarray(num_pyramids)
    valid_pyr = num_pyramids[indices]
    all_boxes = []
    for i, item in enumerate(final_box):
        if valid_pyr[i] != 0:
            factor = valid_pyr[i]*SCALE
        else:
            factor = 1
        all_boxes.append(final_box*factor)
    all_boxes = np.asarray(all_boxes)
    nms_output = nms(final_box, 0.3)  # 0.1
    img_copy = incoming_img.copy()
    return all_windows[indices,:], final_box, incoming_img, valid_pyr, nms_output  # indices[0]


# if __name__ == "__main__":
    # training()  # training the model
    # Comment above after training the model
