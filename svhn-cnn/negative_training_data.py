import cv2
import os
import numpy as np
import json
import glob
import h5py


FULL_TRAIN_FILE = "train_full_data"
EXTRA = "negatives"


def read_data():
    positives = []
    boxes = []
    positive_labels = []
    with open('train_data.json') as f:
        data = json.load(f)
        data = data['results']
        len_data = len(data)
        for d in data:
            filename = d['filename']
            if filename == '29930.png':  # ignoring the file with 6 digits (there is only 1 file)
                continue
            image = cv2.imread(os.path.join(FULL_TRAIN_FILE, filename))
            box = d['boxes']  # box is a list
            boxes.append(box)
            label = box[0]['label']
            positives.append(image)
            positive_labels.append(label)
    positives = np.array(positives)
    positive_labels = np.array(positive_labels)
    return positives, len_data, boxes, positive_labels


def save_images(data):
    # file = h5py.File('negative images 64x64.h5', 'w')
    file = h5py.File('all_negatives_64x64.h5', 'w')
    file.create_dataset('dataset_name', data=data)
    print('Data saved in all_negatives_64x64.h5')


def create_negatives():
    positives, len_data, boxes, positive_labels = read_data()
    negatives = []
    c = 0
    for i in range(0, len_data - 1):
        new_size = (64, 64)  # new_size = (32, 32)
        b = boxes[i]
        top = [d['top'] for d in b]  # y
        left = [d['left'] for d in b]  # x
        # if i==6876:
        #     print('6876',left, b)
        width = [d['width'] for d in b]  # w
        height = [d['height'] for d in b]  # h
        final_top, final_left = np.min(top), np.min(left)
        final_height, final_width = np.max(height), np.sum(width)  # np.max(0, np.int(np.sum(left)))
        dist_height = final_height * 0.2
        dist_width = final_width * 0.2
        min_top = int(final_top - dist_height)  # leftmost top point y
        min_left = int(final_left - dist_width)  # leftmost top point x
        max_bot = int(final_top + final_height - dist_height)  # rightmost bottom point y
        max_right = int(final_left + final_width - dist_width)  # rightmost bottom point x
        img = positives[i]
        img_h, img_w, _ = img.shape
        if min_top > 20 and min_left > 20:
            cropped = cv2.resize(img[0: min_top - 1, 0:min_left - 1, :], new_size)
            negatives.append(cropped)
            c += 1
            if (max_bot - min_top) > 20 and min_left > 20:
                cropped = cv2.resize(img[min_top:max_bot, 0:min_left - 1, :], new_size)
                negatives.append(cropped)
                c += 1
            if min_top > 20 and (max_right - min_left) > 20:
                cropped = cv2.resize(img[0: min_top - 1, min_left:max_right, :], new_size)
                negatives.append(cropped)
                c += 1
        if (img_h - max_bot) > 20 and (img_w - max_right) > 20:
            cropped = cv2.resize(img[max_bot + 1: img_h, max_right + 1:img_w, :], new_size)
            negatives.append(cropped)
            c += 1
            if min_top > 0 and (max_bot - min_top) > 20 and (img_w - max_right) > 20:
                cropped = cv2.resize(img[min_top: max_bot, max_right + 1:img_w, :], new_size)
                negatives.append(cropped)
                c += 1
            if max_bot > 0 and min_left > 0 and (img_h - max_bot) > 20 and (max_right - min_left) > 20:
                cropped = cv2.resize(img[max_bot + 1: img_h, min_left:max_right, :], new_size)
                negatives.append(cropped)
                c += 1
    # print('count', c)
    return negatives


def additional_neg():
    dx = dy = 100
    negs = []
    all_images = [cv2.imread(file) for file in glob.glob("negatives/*.jpg")]
    for i in range(len(all_images)):
        img = all_images[i]
        h, w, _ = img.shape
        # print('w - dx - 1', w - dx - 1, h - dy - 1)
        x = [np.random.randint(dx, w - dx - 1, size=50)][0]
        y = [np.random.randint(dy, h - dy - 1, size=50)][0]
        result = list(zip(x, y))
        for i in range(len(result)):
            val = result[i]
            x1 = val[0]
            y1 = val[1]
            x2 = val[0]+64
            y2 = val[1]+64

            if x2<w and y2<h:
                crop = img[y1:y2, x1:x2, :]
                negs.append(crop)
    # print('final_negs', len(negs))
    return negs


if __name__ == "__main__":
    negatives = create_negatives()
    add_neg = additional_neg()
    final_negatives = negatives + add_neg
    save_images(final_negatives)

# earlier negative training data -> 12833
# now: 34176 -> more_negative_images_64x64.h5
