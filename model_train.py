from tensorflow.keras.applications import MobileNetV2, Xception, NASNetMobile
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, GlobalAveragePooling2D, Flatten, Dense, Input, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers, optimizers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import tensorflow as tf
import configparser
import sys
import os
import cv2
import math
import json
import numpy as np
from model import *


def focal_loss(alpha, gamma):
    def focal_loss_fixed(y_true, y_pred):
        cross_entropy = tf.multiply(y_true, -tf.math.log(y_pred))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, cross_entropy))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


def image_data_gen(batch_size, image_dir, label_img_name_dict, img_type =".jpeg"):
    img_batch_collection = []
    label_batch_collection = []
    image_name_list = [img for img in os.listdir(image_dir) if img.endswith(img_type)]
    img_batch = [cv2.imread(os.path.join(image_dir, image_name_list[0]))]
    label_batch = [int(label_img_name_dict[image_name_list[0]])]
    prev_image_shape = img_batch[0].shape
    for i in range(len(image_name_list)):
        curr_image_arr = cv2.imread(os.path.join(image_dir, image_name_list[i]))
        if len(img_batch) < batch_size and curr_image_arr.shape == prev_image_shape:
            img_batch.append(curr_image_arr)
            label_batch.append(int(label_img_name_dict[image_name_list[i]]))
        else:
            img_batch_collection.append(img_batch)
            label_batch_collection.append(label_batch)
            img_batch = [curr_image_arr]
            label_batch = [int(label_img_name_dict[image_name_list[i]])]
        prev_image_shape = curr_image_arr.shape

    while True:
        for i in range(len(img_batch_collection)):
            batch_x = np.array(img_batch_collection[i])
            batch_y = to_categorical(label_batch_collection[i], num_classes=4)
            yield batch_x, batch_y


config_file = "model_train.cfg"
cfg_parser = configparser.ConfigParser()
cfg_parser.read(config_file)

train_img_folder = cfg_parser.get('Input', 'train_img_folder')
img_type = cfg_parser.get("Input", "img_type")
val_img_folder = cfg_parser.get('Input', 'val_img_folder')
train_label_image_json = cfg_parser.get('Input', 'train_label_image_json')
val_label_image_json = cfg_parser.get('Input', 'val_label_image_json')

model_save_dir = cfg_parser.get('output', 'save_dir')
model_weights_file_name = cfg_parser.get('output', 'model_weights')
model_arch_file_name = cfg_parser.get('output', 'model_arch')

with open(train_label_image_json, 'r') as inp:
    train_labels = json.load(inp)

with open(val_label_image_json, 'r') as inp:
    val_labels = json.load(inp)

batch_size = int(cfg_parser.get('model_specs', 'batch_size'))
num_epochs = int(cfg_parser.get("model_specs", "epochs"))
optimizer = cfg_parser.get("model_specs", "optimizer")
learning_rate = float(cfg_parser.get("model_specs", "learning_rate"))
gpu = cfg_parser.get("model_specs", "gpu")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

image_classifier = Image_Classifier(0, [], [], 'mobilenet_v2', optimizer, learning_rate)

number_of_train_images = len([img for img in os.listdir(train_img_folder) if img.endswith(img_type)])
number_of_test_images = len([img for img in os.listdir(val_img_folder) if img.endswith(img_type)])

image_classifier.train_model(img_type, train_img_folder, train_labels, val_img_folder, val_labels, model_save_dir, model_weights_file_name, batch_size, num_epochs)
