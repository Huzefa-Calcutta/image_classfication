#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script for doing inference
"""
import os
import datetime
import configparser
import cv2
from model import *
import tensorflow as tf
import numpy as np
import json


if __name__ == '__main__':
    # loading the config file with info about location of test data and model
    cfgParse = configparser.ConfigParser()
    cfgParse.read("model_inference.cfg")

    # location of model storage
    model_loc = cfgParse.get("data", "model_filepath")

    # location of test data
    test_img_dir = cfgParse.get("data", "test_data")
    class_int_json = cfgParse.get("data", "class_int_dict_json")
    with open(class_int_json, 'r') as f:
        class_int_dict = json.load(f)
    int_class_dict = {v:k for k,v in class_int_dict.items()}

    # directory where predictions have to be stored
    prediction_loc = cfgParse.get('data', 'predicted_loc')
    if not os.path.isdir(os.path.split(prediction_loc)[0]):
        if os.path.split(prediction_loc)[0] != '':
            os.makedirs(prediction_loc)

    prediction_time_st = datetime.datetime.now()
    # creating classifier instance
    focal_loss_fixed = focal_loss(0.4, 2)
    model = tf.keras.models.load_model(model_loc, custom_objects={'focal_loss_fixed':focal_loss_fixed})
    test_data = {}
    for image in os.listdir(test_img_dir):
        image_arr = np.array([cv2.imread(os.path.join(test_img_dir, image))], dtype=np.float32)
        test_data[image] = int_class_dict[np.argmax(model.predict(image_arr))]
    prediction_time_end = datetime.datetime.now()
    prediction_time = (prediction_time_end - prediction_time_st).total_seconds() / 60.0
    print("Time required for prediction for random forest model is %.3f minutes" % prediction_time)
