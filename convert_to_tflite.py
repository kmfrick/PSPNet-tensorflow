#!/usr/bin/env python3

# Run on TF 2.x!

import tensorflow as tf

export_dir="./pspnet50_ade20k_saved"

model = tf.saved_model.load(export_dir)
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([473, 473, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

tflite_model = converter.convert()
open("pspnet50_ade20k_f32.tflite", "wb").write(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("pspnet50_ade20k_dyr.tflite", "wb").write(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
open("pspnet50_ade20k_f16.tflite", "wb").write(tflite_model)
