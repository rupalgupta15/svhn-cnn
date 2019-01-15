# References most of the code from PS4:
import cv2
import numpy as np


def reduce_image(image):
    kernel = np.array((1.0, 4.0, 6.0, 4.0, 1.0))/16.0
    # five_tap = np.outer(kernel, kernel)
    filtered = cv2.sepFilter2D(image, -1, kernel, kernel)
    return filtered[0::2, 0::2]


def gaussian_pyramid(image, levels):
    img_in = image.copy()
    pyramids = []
    pyramids.append(img_in)
    img = img_in
    for i in range(levels-1):
        reduced = reduce_image(img)
        # norm_img = normalize_and_scale(reduced)
        pyramids.append(reduced)
        img = reduced
    return pyramids


def pyramid(img, scale=1.75):  # 1.5, 1.75. 2
    yield img
    size = (64,64)
    while True:
        w = int(img.shape[1]/scale)
        h = int(img.shape[0]/scale)
        img = cv2.resize(img, (w, h))
        if img.shape[0] < size[1] or img.shape[1] < size[0]:
            break
        yield img


def normalize_and_scale(image_in, scale_range=(0, 255)):
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)
    return image_out


def create_combined_img(img_list):
    size = len(img_list)
    h = []
    w = []
    for i in range(size):
        img_out = img_list[i]
        h.append(img_out.shape[0])
        w.append(img_out.shape[1])
    max_h = max(h)
    combined = np.zeros((max_h, np.sum(w)))
    x = 0
    for i in range(size):
        height = h[i]  # odd dimensions
        new_img = img_list[i]
        norm_img = normalize_and_scale(new_img)
        combined[:height, x:x+w[i]] = norm_img
        x += w[i]
    return combined


