# CNN 1 for Format 2 images basic and complex
# File to train the network for CNN
# cnn_digit_classification.py

"""Please Note that the 5 images generated in graded_images folder takes around 30 seconds per image
Therefore all 5 images will be generated in 2.5 minutes"""


import tensorflow as tf
import os
import numpy as np
# import matplotlib.pyplot as plt
import cv2

from tensorflow.python.keras.layers import Dense, Input, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score
from helpers import normalize_images, read_bounding_box_data
from helpers import normalize_single_test_img
import cnn_digit_recognizer_full
from data_prep_for_cnn2 import data_prep, fetch_data

# For reproducability:
np.random.seed(23)

SCALE = 1.75  # 1.5
epochs = 30
# INPUT_DIR = "train"
TRAIN_FILE = "train_32x32.mat"
TEST_FILE = "test_32x32.mat"
OUTPUT_DIR = "output"
FULL_TRAIN_FILE = "train_full_data"
FULL_TEST_FILE = "test_full_data"
VID_DIR = "my_videos"
VID_OUT_DIR = "out_videos"


# added axis = -1 in batch normalization, axis that should be normalized (typically the features axis)
# added reduceLronplateau
# best architecture: 8 hidden conv layers, 1 locally connected hidden layer and 2 densely connected hidden layers
def build_model(load_weights_file=None):
    input = Input(shape=(64, 64,1))
    # Changed all kernels to 5x5 from 3x3  : 5x5; Removed 32 made it 48
    model = Conv2D(48, (5, 5), padding='same', activation=tf.nn.relu)(input)
    model = Conv2D(48, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    model = Conv2D(64, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = Conv2D(64, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    model = Conv2D(128, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = Conv2D(128, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    # 192 for locally connected layers
    model = Conv2D(256, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = Conv2D(256, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    model = Conv2D(512, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = Conv2D(512, (5, 5), padding='same', activation=tf.nn.relu)(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D()(model)

    # model = Conv2D(256, 3, padding='same', activation=tf.nn.relu)(model)
    # model = BatchNormalization()(model)
    # model = MaxPooling2D()(model)
    # model = Conv2D(512, 3, padding='same', activation=tf.nn.relu)(model)
    # model = BatchNormalization()(model)
    # model = MaxPooling2D()(model)

    model = Flatten(name='flatten')(model)
    model = Dense(3072, activation=tf.nn.relu, name='fc1')(model)
    model = Dense(3072, activation=tf.nn.relu, name='fc2')(model)
    model = Dropout(0.5)(model)

    # model = Dense(128, activation=tf.nn.relu)(model) # Extra for v2
    num_digits = Dense(6, activation=tf.nn.softmax)(model)  # none=10, 1, 2, 3, 4, 5
    # output from multi digit CNN: no of digits, predicting digits till 4 places
    digit1 = Dense(11, activation=tf.nn.softmax)(model)
    digit2 = Dense(11, activation=tf.nn.softmax)(model)
    digit3 = Dense(11, activation=tf.nn.softmax)(model)
    digit4 = Dense(11, activation=tf.nn.softmax)(model)
    digit5 = Dense(11, activation=tf.nn.softmax)(model)

    model = Model(inputs=input, outputs=[num_digits, digit1, digit2, digit3, digit4, digit5])
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if load_weights_file is not None:
        print("Loading Weights")
        model.load_weights(load_weights_file)
    model.summary()
    return model


def train_model(Xtrain, Ytrain):
    # datagen = ImageDataGenerator(
    #             featurewise_center=True, featurewise_std_normalization=True,
    #             rotation_range = 45, width_shift_range=0.6, height_shift_range=0.6,
    #             fill_mode='nearest')  # what about brightness
    # datagen.fit(Xtrain)
    model = build_model()
    # Adam rate default : 0.001
    checkpoint = ModelCheckpoint(filepath='models/classification/cnn2_complex-{epoch:03d}.h5', monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True, verbose=2)
    decrease_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, cooldown=1, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor= 'loss',  min_delta=0.000001, patience=5, verbose=1, mode='auto')
    callbacks = [checkpoint, decrease_lr, early_stopping]
    indices = Xtrain.shape[0]
    arr = np.arange(indices)
    np.random.shuffle(arr)  # shuffling for validation- in place
    Xtrain = Xtrain[arr,:,:,:]
    Ytrain = Ytrain[arr,:]
    Y_num_dig, Y_dig1, Y_dig2, Y_dig3, Y_dig4, Y_dig5 = Ytrain[:,0], Ytrain[:,1], Ytrain[:,2], Ytrain[:,3], Ytrain[:,4], Ytrain[:,5]
    model.fit(Xtrain, [Y_num_dig, Y_dig1, Y_dig2, Y_dig3, Y_dig4, Y_dig5],
              validation_split=0.2, epochs=epochs, verbose=2, callbacks=callbacks)

    # If generating plots, comment line above and uncomment below
    # class_cnn_hist_train = model.fit(Xtrain, [Y_num_dig, Y_dig1, Y_dig2, Y_dig3, Y_dig4, Y_dig5],
    #           validation_split=0.2, epochs=epochs, verbose=2, callbacks=callbacks)
    #
    # generate_acc_plots(class_cnn_hist_train, 'dense_acc', 'val_dense_acc', "num_dig_acc", 'Number of Digits Accuracy')
    # generate_acc_plots(class_cnn_hist_train, 'dense_1_acc', 'val_dense_1_acc', 'digit_1_acc', "Digit 1 Accuracy")
    # generate_acc_plots(class_cnn_hist_train, 'dense_2_acc', 'val_dense_2_acc', 'digit_2_acc', "Digit 2 Accuracy")
    # generate_acc_plots(class_cnn_hist_train, 'dense_3_acc', 'val_dense_3_acc', 'digit_3_acc', "Digit 3 Accuracy")
    # generate_acc_plots(class_cnn_hist_train, 'dense_4_acc', 'val_dense_4_acc', 'digit_4_acc', "Digit 4 Accuracy")
    # generate_acc_plots(class_cnn_hist_train, 'dense_5_acc', 'val_dense_5_acc', 'digit_5_acc', "Digit 5 Accuracy")
    #
    # generate_loss_plots(class_cnn_hist_train, 'dense_loss', 'val_dense_loss', 'num_dig_loss', "Number of Digits Loss")
    # generate_loss_plots(class_cnn_hist_train, 'dense_1_loss', 'val_dense_1_loss', 'digit_1_loss', "Digit 1 Loss")
    # generate_loss_plots(class_cnn_hist_train, 'dense_2_loss', 'val_dense_2_loss', 'digit_2_loss', "Digit 2 Loss")
    # generate_loss_plots(class_cnn_hist_train, 'dense_3_loss', 'val_dense_3_loss', 'digit_3_loss', "Digit 3 Loss")
    # generate_loss_plots(class_cnn_hist_train, 'dense_4_loss', 'val_dense_4_loss', 'digit_4_loss', "Digit 4 Loss")
    # generate_loss_plots(class_cnn_hist_train, 'dense_5_loss', 'val_dense_5_loss', 'digit_5_loss', "Digit 5 Loss")
    #
    # generate_model_loss_plot(class_cnn_hist_train, 'loss', 'val_loss', 'model_loss', "Model Loss")
    #
    # generate_err_plots(class_cnn_hist_train, 'dense_acc', 'val_dense_acc', "num_dig_error", "Number of Digits Error")
    # generate_err_plots(class_cnn_hist_train, 'dense_1_acc', 'val_dense_1_acc', 'digit_1_error', "Digit 1 Error")
    # generate_err_plots(class_cnn_hist_train, 'dense_2_acc', 'val_dense_2_acc', 'digit_2_error', "Digit 2 Error")
    # generate_err_plots(class_cnn_hist_train, 'dense_3_acc', 'val_dense_3_acc', 'digit_3_error', "Digit 3 Error")
    # generate_err_plots(class_cnn_hist_train, 'dense_4_acc', 'val_dense_4_acc', 'digit_4_error', "Digit 4 Error")
    # generate_err_plots(class_cnn_hist_train, 'dense_5_acc', 'val_dense_5_acc', 'digit_5_error', "Digit 5 Error")


# def generate_acc_plots(model_hist, line1, line2, filename, plot_title):
#     print("Plotting {}".format(plot_title))
#     fig = plt.figure(figsize=(7, 6))
#     plt.style.use('seaborn-bright')
#     plt.grid(b=None, which='major', axis='both', c='#D3D3D3')
#     plt.plot(model_hist.history[line1], c='#003EFF', label='Training')
#     plt.plot(model_hist.history[line2], c='#FF4500', label='Validation')
#     ax = plt.gca()
#     ax.set_xlim(ax.get_xlim())
#     plt.title(plot_title)
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylim((0, 1.1))
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     plt.legend(loc='lower right', frameon=True)
#     fig.savefig('plots/classification_cnn/' + filename + '.png')
#     plt.close()
#
#
# def generate_loss_plots(model_hist, line1, line2, filename, plot_title):
#     print("Plotting {}".format(plot_title))
#     fig = plt.figure(figsize=(7, 6))
#     plt.style.use('seaborn-bright')
#     plt.grid(b=None, which='major', axis='both', c='#D3D3D3')
#     plt.plot(model_hist.history[line1], c='#003EFF', label='Training')
#     plt.plot(model_hist.history[line2], c='#FF4500', label='Validation')
#     ax = plt.gca()
#     ax.set_xlim(ax.get_xlim())
#     plt.title(plot_title)
#     plt.ylabel('Loss')
#     plt.xlabel('Epochs')
#     plt.ylim((-0.1, 1.1))
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     plt.legend(loc='upper right', frameon=True)
#     fig.savefig('plots/classification_cnn/' + filename + '.png')
#     plt.close()
#
#
# def generate_model_loss_plot(model_hist, line1, line2, filename, plot_title):
#     print("Plotting {}".format(plot_title))
#     fig = plt.figure(figsize=(7, 6))
#     plt.style.use('seaborn-bright')
#     plt.grid(b=None, which='major', axis='both', c='#D3D3D3')
#     plt.plot(model_hist.history[line1], c='#003EFF', label='Training')
#     plt.plot(model_hist.history[line2], c='#FF4500', label='Validation')
#     ax = plt.gca()
#     ax.set_xlim(ax.get_xlim())
#     plt.title(plot_title)
#     plt.ylabel('Loss')
#     plt.xlabel('Epochs')
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     plt.legend(loc='upper right', frameon=True)
#     fig.savefig('plots/classification_cnn/' + filename + '.png')
#     plt.close()
#
#
# def generate_err_plots(model_hist, line1, line2, filename, plot_title):
#     train = [1.0 - i for i in model_hist.history[line1]]
#     val = [1.0 - i for i in model_hist.history[line2]]
#     print("Plotting {}".format(plot_title))
#     fig = plt.figure(figsize=(7, 6))
#     plt.style.use('seaborn-bright')
#     plt.grid(b=None, which='major', axis='both', c='#D3D3D3')
#     plt.plot(train, c='#003EFF', label='Training')
#     plt.plot(val, c='#FF4500', label='Validation')
#     ax = plt.gca()
#     ax.set_xlim(ax.get_xlim())
#     plt.title(plot_title)
#     plt.ylabel('Error')
#     plt.xlabel('Epochs')
#     plt.ylim((-0.1, 1.1))
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     plt.legend(loc='upper right', frameon=True)
#     fig.savefig('plots/classification_cnn/' + filename + '.png')
#     plt.close()


def build_confusion_matrix(Xtest, Ytest):
    model = build_model("./models/classification/cnn2_complex-010.h5")
    print("Model Loaded...")
    print(model.evaluate(Xtest, Ytest))
    actual_labels = np.argmax(model.predict(Xtest), axis=1)
    print(actual_labels)
    expected_labels = Ytest
    c = confusion_matrix(actual_labels, expected_labels)
    print('confusion', c)


def evaluate_model(x):
    model = build_model("./models/cnn2_complex-010.h5")
    y_prob = model.predict(x)
    y_classes = y_prob.argmax(axis=-1)
    print('predicted', y_prob, y_classes)


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


def generate_training_data(negatives, train_idx):
    Xtrain, Ytrain = read_bounding_box_data('train_data.json', FULL_TRAIN_FILE, 64)  # changed size in helpers
    Xtrain = np.append(Xtrain, negatives[:train_idx], axis=0)  # adding negative images to data
    new_neg_y_idx = negatives[:train_idx].shape[0]
    total_y_idx = len(Ytrain) + new_neg_y_idx
    new_y_labels = create_y_labels(total_y_idx, Ytrain)
    Xdata_norm, Ydata_norm = normalize_images(Xtrain, new_y_labels)
    return Xdata_norm, Ydata_norm


def test_model(test_Xdata_norm, test_Ydata_norm):
    # building model and loading it:
    model = build_model("./models/classification/cnn2_complex-010.h5")
    print("Model Loaded...")
    Y_num_dig, Y_dig1, Y_dig2, Y_dig3, Y_dig4, Y_dig5 = test_Ydata_norm[:, 0], test_Ydata_norm[:, 1], \
                                                        test_Ydata_norm[:, 2], test_Ydata_norm[:, 3], \
                                                        test_Ydata_norm[:, 4], test_Ydata_norm[:,5]

    # print('model evaluation', model.evaluate(test_Xdata_norm, [Y_num_dig, Y_dig1, Y_dig2, Y_dig3, Y_dig4, Y_dig5]))
    prediction = model.predict(test_Xdata_norm)  # earlier = 192: 195 [379:391,:,:,:]
    # print('prediction[0]', prediction[0], prediction[0].shape)
    conf_idx = np.where(prediction[0] > 0.7)
    new_pred_num_digit = []
    # c = 0
    # for i in range(test_Xdata_norm[100:195,:,:,:].shape[0]):
    for i in range(test_Xdata_norm.shape[0]):
        real_idx = np.where(conf_idx[0] == i)[0]
        if i in conf_idx[0]:
            new_pred_num_digit.append(conf_idx[1][real_idx][0])
        elif i not in conf_idx[0]:
            new_pred_num_digit.append(0)
            # c+=1
    new_pred_num_digit = np.asarray(new_pred_num_digit)
    pred_num_digit = new_pred_num_digit  # np.argmax(prediction[0], axis=1)
    pred_num_digit = np.argmax(prediction[0], axis=1)  # only use when doing per image
    pred_digit_1 = np.argmax(prediction[1], axis=1)
    pred_digit_2 = np.argmax(prediction[2], axis=1)
    pred_digit_3 = np.argmax(prediction[3], axis=1)
    pred_digit_4 = np.argmax(prediction[4], axis=1)
    pred_digit_5 = np.argmax(prediction[5], axis=1)
    my_pred = np.vstack((pred_num_digit, pred_digit_1, pred_digit_2, pred_digit_3, pred_digit_4, pred_digit_5))
    final_pred = my_pred.T
    expected_labels = test_Ydata_norm  # [353:360,:] # works fine: 236:243, 353:360  [379:391,:]
    # print('expected_labels', expected_labels, expected_labels.shape)
    num_correct = np.count_nonzero(final_pred == expected_labels)
    all_equal = np.all(final_pred == expected_labels, axis=1)
    correct = np.count_nonzero(all_equal)
    # print('num_correct', num_correct, expected_labels.shape[0])
    acc = (correct/expected_labels.shape[0])*100
    # acc = accuracy_score(final_pred, expected_labels, normalize=False)
    print('acc', acc)


def final_pred_post_processing(final_pred):
    max_num_dig = np.max(final_pred[:,0])
    idx = np.where(final_pred[:, 0] == max_num_dig)
    all_digits = final_pred[idx]
    lkp = {}
    for x in all_digits:
        tup = tuple(x)
        if tup not in lkp:
            lkp[tup] = 1
        else:
            lkp[tup] += 1
    d = max(lkp, key=lkp.get)
    final = list(d)
    num_dig = final[0]
    val = []
    for item in final[1:num_dig+1]:
        val.append(item)
    value = ''.join(map(str, val))
    return value


# all windows should be normalized before calling this method
def test_model_real_images(all_windows, model=None):
    # building model and loading it:
    if model:
        model = model
    else:
        model = build_model("./models/classification/cnn2_complex-010.h5")
        print("CNN2 Model Loaded...")
    prediction = model.predict(all_windows)  # [192:193,:,:,:]
    # print('prediction shape', len(prediction))
    # method 1: take argmax:
    num_dig = np.argmax(prediction[0], axis=1)  # num digits [147  78 191  32   4  50]
    # above line tells which cropped window has how many digits
    # we want to fetch num of digits which are maximum in the above list - 2 is the maximum no of digits predicted
    num_digits_non_zero = num_dig[np.where(num_dig>0)]
    non_zero_idx = np.where(num_dig>0)
    non_zero_pred = prediction[0][non_zero_idx,:]
    cross_thresh = np.where(non_zero_pred>0.98)  # 0.85, 0.98
    cross_thresh_new = np.asarray(cross_thresh[1])
    final_non_zero_idx = np.take(non_zero_idx, cross_thresh_new)
    pred_num_digit = np.argmax(prediction[0][final_non_zero_idx], axis=1)
    pred_digit_1 = np.argmax(prediction[1][final_non_zero_idx], axis=1)
    pred_digit_2 = np.argmax(prediction[2][final_non_zero_idx], axis=1)
    pred_digit_3 = np.argmax(prediction[3][final_non_zero_idx], axis=1)
    pred_digit_4 = np.argmax(prediction[4][final_non_zero_idx], axis=1)
    pred_digit_5 = np.argmax(prediction[5][final_non_zero_idx], axis=1)
    my_pred = np.vstack((pred_num_digit, pred_digit_1, pred_digit_2, pred_digit_3, pred_digit_4, pred_digit_5))
    final_pred = my_pred.T
    most_vote = final_pred_post_processing(final_pred)
    return final_non_zero_idx, most_vote


def train_test_model():
    # Xtest_norm, Ytest_norm = fetch_data()
    # test_model(Xtest_norm, Ytest_norm)  # can be used to find overall accuracy for test dataset
    # to check accuracy of the train sequence:
    Xdata_norm, Ydata_norm = data_prep()
    test_model(Xdata_norm, Ydata_norm)


def test_one_image():  # just to check the performance of CNN on a cropped perfect image:
    # filename = 'my_images'
    # incoming_img = cv2.imread(os.path.join(filename, 'bw_scn2.png'))
    test_file = 'test_full_data'
    incoming_img = cv2.imread(os.path.join(test_file, '1976.png'))
    model = build_model("./models/classification/cnn2_complex-010.h5")
    shapes = incoming_img.shape
    img_list = []
    if len(shapes) == 3:
        gray = cv2.resize(incoming_img, (64, 64))
        gray = normalize_single_test_img(gray)
        img_list.append(gray) # gray[..., np.newaxis]
    img_arr = np.asarray(img_list)
    prediction = model.predict(img_arr)
    # print('prediction', prediction, prediction[0])  # len of output:
    pred_num_digit = np.argmax(prediction[0])
    pred_digit_1 = np.argmax(prediction[1])
    pred_digit_2 = np.argmax(prediction[2])
    pred_digit_3 = np.argmax(prediction[3])
    pred_digit_4 = np.argmax(prediction[4])
    pred_digit_5 = np.argmax(prediction[5])
    predicted_label = [pred_num_digit, pred_digit_1, pred_digit_2, pred_digit_3, pred_digit_4, pred_digit_5]
    print('predicted_label', predicted_label)
    # print('expected_labels', expected_labels)


def train_main():  # only for training
    Xdata_norm, Ydata_norm = data_prep()
    train_model(Xdata_norm, Ydata_norm)  # only uncomment if you need to train the model


# to be called for cnn_digit_recognizer
def cnn_main(img, write=True, video=False):  # this is to call after the original cnn has been called once
    img_to_test = img
    print('Processing input image')
    # print('img_to_test', img_to_test[6])
    # actual_window, selected_box, incoming_img, pyr_num, cnn1_nms = cnn_digit_recognizer_full.final_method_cnn(img_to_test + '.png', video)
    actual_window, selected_box, incoming_img, pyr_num, cnn1_nms = cnn_digit_recognizer_full.final_method_cnn(img_to_test, video)
    actual_indices, most_voted = test_model_real_images(actual_window)
    indices = actual_indices[int(len(actual_indices)/2)]  # selected box  [0]
    final_box = actual_window[indices,:]  # cropped_img
    img_copy = incoming_img.copy()
    selected_bbox = selected_box[indices,:]
    selected_pyr = pyr_num[indices]
    if selected_pyr == 0:
        f = 1
    else:
        f = selected_pyr*SCALE
    scaled_bbox = selected_bbox*f
    x1 = int(scaled_bbox[0])
    y1 = int(scaled_bbox[1])
    x2 = int(scaled_bbox[2])
    y2 = int(scaled_bbox[3])
    cv2.rectangle(img_copy, (int(0.98*x1), int(0.98*y1)), (int(1.05*(x2)), int(1.05*(y2))), (0,255,0), 2)
    cv2.putText(img_copy, most_voted, (x1+50, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)

    if len(final_box.shape) == 3:  # only one box:
        if write == True:
            cv2.imwrite('graded_images/' + img_to_test[6:], img_copy)
        else:
            return img_copy
    else:
        for item in final_box:
            # x, y, maxx, maxy = item[0], item[1], item[2], item[3]
            # cropped = incoming_img[y:maxy, x:maxx]
            print('shape of box', img_copy.shape)  # cropped
            # cv2.rectangle(img_copy, (x,y), (maxx, maxy), (255,0,0), 2)


# to be called for cnn_digit_recognizer
def cnn_video(img, model_cnn_1, model_cnn_2, video=True):  # this is to call after the original cnn has been called once
    img_to_test = img
    actual_window, selected_box, incoming_img, pyr_num, cnn1_nms = cnn_digit_recognizer_full.final_method_cnn(img_to_test, video=video, model_cnn_1=model_cnn_1)
    actual_indices, most_voted = test_model_real_images(actual_window, model_cnn_2)
    indices = actual_indices[int(len(actual_indices)/2)]  # selected box  [0]
    final_box = actual_window[indices,:]  # cropped_img
    img_copy = incoming_img.copy()
    selected_bbox = selected_box[indices,:]
    selected_pyr = pyr_num[indices]
    if selected_pyr == 0:
        f = 1
    else:
        f = selected_pyr*SCALE
    scaled_bbox = selected_bbox*f
    x1 = int(scaled_bbox[0])
    y1 = int(scaled_bbox[1])
    x2 = int(scaled_bbox[2])
    y2 = int(scaled_bbox[3])
    cv2.rectangle(img_copy, (int(0.98*x1), int(0.98*y1)), (int(1.05*(x2)), int(1.05*(y2))), (0,255,0), 2)
    cv2.putText(img_copy, most_voted, (x1+50, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)
    if len(final_box.shape) == 3:  # only one box:
        return img_copy
    else:
        for item in final_box:
            # x, y, maxx, maxy = item[0], item[1], item[2], item[3]
            # cropped = incoming_img[y:maxy, x:maxx]
            print('shape of box', img_copy.shape)
            # cv2.rectangle(img_copy, (x,y), (maxx, maxy), (255,0,0), 2)

# Reusing code from PS3
# def mp4_video_writer(filename, frame_size, fps=20):
#     print('writing the video to output')
#     filename = filename.replace("mp4", "avi")
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def my_read_video(video_name, fps):
    filename = os.path.join(VID_DIR, video_name)
    frame_num = 1
    model_cnn_1 = cnn_digit_recognizer_full.build_model("./models/detection/digit_recognizer_full-025.h5")
    # print("CNN 1 Model Loaded for Video...")
    model_cnn_2 = build_model("./models/classification/cnn2_complex-010.h5")
    # print("CNN 2 Model Loaded for Video...")
    video = cv2.VideoCapture(filename)
    out_path = "out_videos/output-{}".format(video_name)
    out_path = out_path.replace("mp4", "avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video_out = mp4_video_writer(out_path, (630, 1120), fps)
    video_out = cv2.VideoWriter(out_path, fourcc, fps, (1120, 630), True)
    print('video.isOpened()', video.isOpened())
    while video.isOpened():
        ret, frame = video.read()
        if ret==True:
            img = np.copy(frame)
            print("Processing frame {}".format(frame_num))
            frame_num = frame_num + 1
        else:
            img = None
            break
        # if frame_num > 3:
        #     break
        final = cnn_video(img, model_cnn_1, model_cnn_2, video=True)
        video_out.write(final)
    video.release()
    video_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # train_main()  # step1 - to train the model - no need to do again once fully trained
    # train_test_model()  # step2 - to check test images accuracy
    # main()  # has the code for test accuracy calculation in 1st line only
    # my_read_video('cnn_video.mp4', 25)  # step 4: video
    # step 3: generate images:
    # Following calls generate 5 images for graded_images:
    cnn_main('input_1.png')
    cnn_main('input_2.png')
    cnn_main('input_3.png')
    cnn_main('input_4.png')
    cnn_main('input_5.png')

# not working images are: output_no_* .png
# The output images are generated under: under graded folder
