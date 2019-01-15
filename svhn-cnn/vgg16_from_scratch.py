"""VGG 16 model pre-trained weights - using complete images
Trained on 64x64 images from full training and test set
"""

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import h5py

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Adam
from helpers import read_bounding_box_data
from data_prep_for_cnn2 import load_more_negative_images_64x64

# Doing Method 2: Fetching data from the train folder
# For reproducability :
# np.random.seed(23)

TRAIN_FILE = "train_32x32.mat"
TEST_FILE = "test_32x32.mat"
OUTPUT_DIR = "output"
FULL_TRAIN_FILE = "train_full_data"
FULL_TEST_FILE = "test_full_data"
epochs = 30


def vgg_build_model(load_weights_file=None):  # Xtrain, Ytrain, Xtest, Ytest, options
    model = VGG16(weights=None, include_top=False)
    # lets fine tune the model
    # for layer in model.layers:
    #     layer.trainable = True
    # row, col, ch = Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3]  # 64, 64, 1 incoming
    row, col, ch = 64, 64, 3
    input_layer = Input(shape=(row, col, ch), name='image_input')  # earlier 32x32x3
    # Use the generated model
    model = model(input_layer)

    model = Flatten(name='flatten')(model)
    model = Dense(3072, activation=tf.nn.relu, name='fc1')(model)
    model = Dense(3072, activation=tf.nn.relu, name='fc2')(model)   #2048
    model = Dropout(0.5)(model)

    # model = Dense(128, activation=tf.nn.relu)(model) # Extra for v2
    num_digits = Dense(6, activation=tf.nn.softmax)(model)  # none=10, 1, 2, 3, 4, 5
    # output from multi digit CNN: no of digits, predicting digits till 5 places
    digit1 = Dense(11, activation=tf.nn.softmax)(model)
    digit2 = Dense(11, activation=tf.nn.softmax)(model)
    digit3 = Dense(11, activation=tf.nn.softmax)(model)
    digit4 = Dense(11, activation=tf.nn.softmax)(model)
    digit5 = Dense(11, activation=tf.nn.softmax)(model)
    model = Model(inputs=[input_layer], outputs=[num_digits, digit1, digit2, digit3, digit4, digit5])

    # my_model = Model(inputs=[input_layer], outputs=[x])
    # my_model.load_weights()
    # In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    # model.add(my_model)  # added
    model.compile(Adam(amsgrad=True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if load_weights_file is not None:
        model.load_weights(load_weights_file)
    return model


def vgg_train_model(Xtrain, Ytrain):  # options,
    print('************* training VGG 16 from scratch ***************')
    model = vgg_build_model()  # options
    indices = Xtrain.shape[0]
    arr = np.arange(indices)
    np.random.shuffle(arr)
    Xtrain = Xtrain[arr, :, :, :]
    Ytrain = Ytrain[arr, :]

    Y_num_dig, Y_dig1, Y_dig2, Y_dig3, Y_dig4, Y_dig5 = Ytrain[:, 0], Ytrain[:, 1], \
                                                        Ytrain[:, 2], Ytrain[:, 3], \
                                                        Ytrain[:, 4], Ytrain[:, 5]
    checkpoint = ModelCheckpoint(filepath='models/vgg/vgg_scratch-{epoch:03d}.h5', monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True, verbose=2)
    decrease_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, cooldown=0, min_lr=0.000001)
    early_stopping = EarlyStopping(monitor='val_loss',  min_delta=0.0000001, patience=5, verbose=1, mode='auto')
    callbacks = [checkpoint, decrease_lr, early_stopping]
    model.summary()
    model.fit(Xtrain, [Y_num_dig, Y_dig1, Y_dig2, Y_dig3, Y_dig4, Y_dig5], batch_size=64,
              validation_split=0.2, epochs=epochs, verbose=2, callbacks=callbacks)

    # If want to generate plot comment above line and uncomment below lines

    # The fit() method on a Keras Model returns a History object.
    vgg_hist_train = model.fit(Xtrain, [Y_num_dig, Y_dig1, Y_dig2, Y_dig3, Y_dig4, Y_dig5], batch_size=64,
                                   validation_split=0.2, epochs=epochs, verbose=2, callbacks=callbacks)

    # The History.history attribute is a dictionary recording training loss values and metrics values at
    # successive epochs, as well as validation loss values and validation metrics values (if applicable).
    # plots to create for single digit accuracy:
    # generate_acc_plots(vgg_hist_train, 'dense_acc', 'val_dense_acc', "num_dig_acc",'Number of Digits Accuracy')
    # generate_acc_plots(vgg_hist_train, 'dense_1_acc', 'val_dense_1_acc', 'digit_1_acc', "Digit 1 Accuracy")
    # generate_acc_plots(vgg_hist_train, 'dense_2_acc', 'val_dense_2_acc', 'digit_2_acc', "Digit 2 Accuracy")
    # generate_acc_plots(vgg_hist_train, 'dense_3_acc', 'val_dense_3_acc', 'digit_3_acc', "Digit 3 Accuracy")
    # generate_acc_plots(vgg_hist_train, 'dense_4_acc', 'val_dense_4_acc', 'digit_4_acc', "Digit 4 Accuracy")
    # generate_acc_plots(vgg_hist_train, 'dense_5_acc', 'val_dense_5_acc', 'digit_5_acc', "Digit 5 Accuracy")
    #
    # generate_loss_plots(vgg_hist_train, 'dense_loss', 'val_dense_loss', 'num_dig_loss', "Number of Digits Loss")
    # generate_loss_plots(vgg_hist_train, 'dense_1_loss', 'val_dense_1_loss', 'digit_1_loss', "Digit 1 Loss")
    # generate_loss_plots(vgg_hist_train, 'dense_2_loss', 'val_dense_2_loss', 'digit_2_loss', "Digit 2 Loss")
    # generate_loss_plots(vgg_hist_train, 'dense_3_loss', 'val_dense_3_loss', 'digit_3_loss', "Digit 3 Loss")
    # generate_loss_plots(vgg_hist_train, 'dense_4_loss', 'val_dense_4_loss', 'digit_4_loss', "Digit 4 Loss")
    # generate_loss_plots(vgg_hist_train, 'dense_5_loss', 'val_dense_5_loss', 'digit_5_loss', "Digit 5 Loss")
    #
    # generate_model_loss_plot(vgg_hist_train, 'loss', 'val_loss', 'model_loss', "Model Loss")
    #
    # generate_err_plots(vgg_hist_train, 'dense_acc', 'val_dense_acc', "num_dig_error","Number of Digits Error")
    # generate_err_plots(vgg_hist_train, 'dense_1_acc', 'val_dense_1_acc', 'digit_1_error', "Digit 1 Error")
    # generate_err_plots(vgg_hist_train, 'dense_2_acc', 'val_dense_2_acc', 'digit_2_error', "Digit 2 Error")
    # generate_err_plots(vgg_hist_train, 'dense_3_acc', 'val_dense_3_acc', 'digit_3_error', "Digit 3 Error")
    # generate_err_plots(vgg_hist_train, 'dense_4_acc', 'val_dense_4_acc', 'digit_4_error', "Digit 4 Error")
    # generate_err_plots(vgg_hist_train, 'dense_5_acc', 'val_dense_5_acc', 'digit_5_error', "Digit 5 Error")


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
#     fig.savefig('plots/vgg_from_scratch/'+ filename+ '.png')
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
#     fig.savefig('plots/vgg_from_scratch/'+ filename+ '.png')
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
#     fig.savefig('plots/vgg_from_scratch/'+ filename+ '.png')
#     plt.close()
#
#
# def generate_err_plots(model_hist, line1, line2, filename, plot_title):
#     train = [1.0-i for i in model_hist.history[line1]]
#     val = [1.0-i for i in model_hist.history[line2]]
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
#     fig.savefig('plots/vgg_from_scratch/' + filename + '.png')
#     plt.close()


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


def save_mean_std(data, filename):
    file = h5py.File(filename, 'w')
    file.create_dataset('dataset_name', data=data)
    print('Data saved mean and std.h5')


def load_mean():
    f = h5py.File("training_mean_std_info/vgg_train_img_mean.h5", 'r')
    key = list(f.keys())[0]
    # print(f.keys(), f[key], f[key].shape)
    return f[key]


def load_std():
    f = h5py.File("training_mean_std_info/vgg_train_img_std.h5", 'r')
    key = list(f.keys())[0]
    # print(f.keys(), f[key], f[key].shape)
    return f[key]


def normalize_test_images(images):  # images are the xtrain kind of values
    print('images',images.shape)
    # Compute channel wise mean and std dev
    images_mean = load_mean()
    images_std = load_std()
    images = images - images_mean
    images = images/images_std
    return images


def normalize_color_images(images):  # images are the xtrain kind of values
    # Compute channel wise mean and std dev
    img_mean = np.mean(images, axis=0)
    images = images - img_mean
    img_std = np.std(images, axis=0)
    images = images/img_std
    save_mean_std(img_mean, 'training_mean_std_info/vgg_train_img_mean.h5')
    save_mean_std(img_std, 'training_mean_std_info/vgg_train_img_std.h5')
    return images


def generate_training_data(negatives, train_idx):
    Xtrain, Ytrain = read_bounding_box_data('train_data.json', FULL_TRAIN_FILE, 64)  # changed size in helpers
    Xtrain = np.append(Xtrain, negatives[:train_idx], axis=0)  # adding negative images to data
    new_neg_y_idx = negatives[:train_idx].shape[0]
    total_y_idx = len(Ytrain) + new_neg_y_idx
    new_y_labels = create_y_labels(total_y_idx, Ytrain)
    Ydata_norm = new_y_labels
    Xdata_norm = normalize_color_images(Xtrain)
    return Xdata_norm, Ydata_norm


def generate_test_data(negatives, train_idx):
    Xtest, Ytest = read_bounding_box_data('test_data.json', FULL_TEST_FILE, 64)
    Xtest = np.append(Xtest, negatives[train_idx:], axis=0)  # adding negative images to data
    test_neg_y_idx = negatives[train_idx:].shape[0]
    total_test_y_idx = len(Ytest) + test_neg_y_idx
    new_test_y_labels = create_y_labels(total_test_y_idx, Ytest)
    test_Ydata_norm = new_test_y_labels
    test_Xdata_norm = normalize_test_images(Xtest)
    return test_Xdata_norm, test_Ydata_norm


def test_model(Xtest, Ytest):
    model = vgg_build_model("./models/vgg/vgg_scratch-014.h5")
    print("Model Loaded...")
    prediction = model.predict(Xtest)  # [192:193, :, :, :]
    pred_num_digit = np.argmax(prediction[0], axis=1)
    pred_digit_1 = np.argmax(prediction[1], axis=1)
    pred_digit_2 = np.argmax(prediction[2], axis=1)
    pred_digit_3 = np.argmax(prediction[3], axis=1)
    pred_digit_4 = np.argmax(prediction[4], axis=1)
    pred_digit_5 = np.argmax(prediction[5], axis=1)
    predicted_label = [pred_num_digit, pred_digit_1, pred_digit_2, pred_digit_3, pred_digit_4, pred_digit_5]
    my_pred = np.vstack((pred_num_digit, pred_digit_1, pred_digit_2, pred_digit_3, pred_digit_4, pred_digit_5))
    final_pred = my_pred.T
    # actual_labels = np.argmax(model.predict(test_Xdata_norm), axis=1)
    expected_labels = Ytest   # [192:193,:]   # [11:12] will get prediction for 12th image
    num_correct = np.count_nonzero(final_pred == expected_labels)
    all_equal = np.all(final_pred == expected_labels, axis=1)
    correct = np.count_nonzero(all_equal)
    # print('num_correct', correct, expected_labels.shape[0])
    acc = (correct / expected_labels.shape[0]) * 100
    # acc = accuracy_score(final_pred, expected_labels, normalize=False)
    print('acc', acc)


def fetch_test_data():
    negatives = load_more_negative_images_64x64()  # 64x64 images only
    train_idx = int(0.8 * len(negatives))  # no of negatives to be added to training data
    Xdata_norm, Ydata_norm = generate_test_data(negatives, train_idx)
    # creating Xtest and Ytest in a similar manner
    # test_Xdata_norm, test_Ydata_norm = generate_training_data(negatives, train_idx) # don't read test data now
    return Xdata_norm, Ydata_norm  # , test_Xdata_norm, test_Ydata_norm


def fetch_data():
    negatives = load_more_negative_images_64x64()  # 64x64 images only
    train_idx = int(0.8 * len(negatives))  # no of negatives to be added to training data
    Xdata_norm, Ydata_norm = generate_training_data(negatives, train_idx)
    # creating Xtest and Ytest in a similar manner
    # test_Xdata_norm, test_Ydata_norm = generate_training_data(negatives, train_idx) # don't read test data now
    return Xdata_norm, Ydata_norm  # , test_Xdata_norm, test_Ydata_norm


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=5)
    # parser.add_argument('--fc', type=int, default=4096)
    # options = parser.parse_args()

    # step 1 : train model
    # Xdata_norm, Ydata_norm = fetch_data()  # this fetches the training data only
    # vgg_train_model(Xdata_norm, Ydata_norm)

    # step 2 : test model
    # test model on the test images:
    # X_testdata_norm, Y_testdata_norm = fetch_test_data()
    # test_model(X_testdata_norm, Y_testdata_norm)  # to check test sequence accuracy
    Xdata_norm, Ydata_norm = fetch_data()
    test_model(Xdata_norm, Ydata_norm)  # to check train sequence accuracy


if __name__ == "__main__":
    main()
