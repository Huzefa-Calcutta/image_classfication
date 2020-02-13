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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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


config_file = sys.argv[1]
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

number_of_train_images = len([img for img in os.listdir(train_img_folder) if img.endswith(img_type)])
number_of_test_images = len([img for img in os.listdir(val_img_folder) if img.endswith(img_type)])

x = Input(shape=(None, None, 3))
feat_extractor = MobileNetV2(include_top=False, input_tensor=x)
x = feat_extractor.output
for layer in feat_extractor.layers[:-10]:
    layer.trainable = False

x = GlobalAveragePooling2D()(x)
x = Dense(4, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.02))(x)
pred_output = Activation('softmax')(x)
classifier = Model(inputs=feat_extractor.input, outputs=pred_output)
classifier.compile(optimizers.Adagrad(lr=0.0001), loss=focal_loss(0.4, 2.0))

for layer in classifier.layers:
    if layer.trainable:
        print(layer)

earlystop = EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)  # ensures that we have model weights corresponding to the best value of the metric at the end of

# make tensorbaord log_dir
if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

tensorboard = TensorBoard(log_dir='./logs', write_graph=True, update_freq='epoch')

# Saving model_checkpoint
filepath = os.path.join(model_save_dir, model_weights_file_name + "-{epoch:02d}-{val_loss:.2f}.h5")
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True,
                             mode='min', save_weights_only=True)

history = classifier.fit_generator(image_data_gen(batch_size, train_img_folder, train_labels, img_type), epochs=num_epochs,
                                           steps_per_epoch=math.ceil(number_of_train_images/num_epochs),
                                           validation_data=image_data_gen(batch_size, val_img_folder, val_labels, img_type),
                                           validation_steps=math.ceil(number_of_test_images/batch_size),
                                           callbacks=[checkpoint, earlystop, tensorboard], verbose=2,
                                           shuffle=False,
                                           use_multiprocessing=True, workers=8)
