## Deep Active Lesion Segmention (DALS), Code by Ali Hatamizadeh ( http://web.cs.ucla.edu/~ahatamiz/ )

import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2 as cv

def create_contour_mask(contours, size):
    contour_mask = np.zeros((size, size))
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            x_pos = contours[i][j][0][0]
            y_pos = contours[i][j][0][1]
            contour_mask[x_pos][y_pos] = 1

    return np.array(contour_mask)


def contoured_image(pred,img):
    thresh_pred_acm = pred.astype(np.uint8)
    contours, hierarchy = cv.findContours(thresh_pred_acm, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    filename = './tmp1.jpg'
    cv.imwrite(filename, img)
    img = cv.imread(filename)
    img = cv.drawContours(img, contours, -1, (255, 255, 0), 1)

    return img


def load_image(path, batch_size,label=False):
    image = np.load(path)
    image = image.astype('float32')
    if label:
        image *= 1.0 / image.max()
    else:
        image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
    image = np.asarray([image] * batch_size)
    image = image[:, :, :, np.newaxis]

    return image


def resolve_status(train_status):
    if train_status == 1:
        restore = False
        is_training = True
    if train_status == 2:
        restore = True
        is_training = True
    if train_status == 3:
        restore = True
        is_training = False

    return restore, is_training

def dice_hard(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice, name='hard_dice')

    return hard_dice

def dice_soft(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, name='dice_coe')

    return dice


def my_func(mask):
    epsilon = 0
    def bwdist(im): return distance_transform_edt(np.logical_not(im))
    bw = mask
    signed_dist = bwdist(bw) - bwdist(1 - bw)
    d = signed_dist.astype(np.float32)
    d += epsilon
    while np.count_nonzero(d < 0) < 5:
        d -= 1

    return d