#!/usr/bin/python3

import cv2
import numpy as np
import tensorflow as tf
loaded = tf.saved_model.load('./pspnet50_ade20k_saved')
infer = loaded.signatures['serving_default']

lut = cv2.imread('ade20k.png')

img_raw = cv2.imread('test.jpg')
h = img_raw.shape[0]
w = img_raw.shape[1]

img = cv2.resize(img_raw, (473, 473))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = tf.dtypes.cast(img, tf.float32)

pred = infer(img)['out'].numpy()

pred = np.squeeze(pred)
pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
pred = np.uint8(pred)
pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
pred = np.uint8(pred)
print(pred.shape)

pred = cv2.LUT(pred, lut)

overlay = cv2.addWeighted(pred, 0.8, img_raw, 0.2, 0)

cv2.imshow('classes', overlay)
cv2.waitKey(0)
