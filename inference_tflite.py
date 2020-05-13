#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
import cv2
import sys
import numpy as np

if len(sys.argv) < 4:
	print('Usage: ' + sys.argv[0] + ' model_path image_path colormap')
	quit()

interpreter = tf.lite.Interpreter(model_path = sys.argv[1])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

lut = cv2.imread(sys.argv[3])
img_raw = cv2.imread('test.jpg')
h = img_raw.shape[0]
w = img_raw.shape[1]
# Preprocess image
img = cv2.resize(img_raw, (input_shape[0], input_shape[1]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = tf.dtypes.cast(img, tf.float32)

# Set input
interpreter.set_tensor(input_details[0]['index'], img)

# Run inference with TFLite
interpreter.invoke()
# Get output
pred = interpreter.get_tensor(output_details[0]['index'])
# Postprocess image
pred = np.squeeze(pred)
pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
pred = np.uint8(pred)
pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
pred = np.uint8(pred)
pred = cv2.LUT(pred, lut)

overlay = cv2.addWeighted(pred, 0.8, img_raw, 0.2, 0)

cv2.imshow('classes', overlay)
cv2.waitKey(0)

