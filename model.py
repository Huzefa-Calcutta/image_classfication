from tensorflow.keras.applications import MobileNetV2, Xception, NASNetMobile
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, GlobalAveragePooling2D, Dense, Input, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers, optimizers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import tensorflow as tf
import os
import cv2
import math
import numpy as np
import inspect


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


class Image_Classifier():
    def __init__(self, num_conv_layers, num_conv_filter, conv_filter_size, feat_extractor="mobilenet_v2", optimiser='adagrad', learning_rate=0.0001, train_feat_extractor=False):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        if isinstance(self.conv_filter_size, int):
            self.conv_filter_size_list = [self.conv_filter_size] * self.num_conv_layers
        if isinstance(self.conv_filter_size, (list, tuple, np.ndarray, set)):
            if len(self.conv_filter_size) < self.num_conv_layers:
                raise ValueError("The filter size for all convolution layers have not been specified. Please ensure the conv_filter_size has same numebr of elements as num_conv_layers")
            self.conv_filter_size_list = self.conv_filter_size

        if isinstance(self.num_conv_filter, int):
            self.conv_filter_num_list = [self.num_conv_filter] * self.num_conv_layers
        if isinstance(self.num_conv_filter, (list, tuple, np.ndarray, set)):
            if len(self.num_conv_filter) < self.num_conv_layers:
                raise ValueError("The number of filters for all convolution layers have not been specified. Please ensure the conv_filter_size has same numeber of elements as num_conv_layers")
            self.conv_filter_num_list = self.num_conv_filter
        if self.feat_extractor == 'mobilenet_v2':
            self.feat_extractor = MobileNetV2(include_top=False)
        else:
            self.feat_extractor = Xception(include_top=False)
        self.model = self.build_model()

    def build_model(self):
        x = Input(shape=(None, None, 3))
        feat_extractor = MobileNetV2(include_top=False, input_tensor=x)
        for layer in feat_extractor.layers:
            layer.trainable = self.train_feat_extractor
        x = feat_extractor.output

        for i in range(self.num_conv_layers):
            conv_layer = Conv2D(filters=self.num_conv_filter[i], kernel_size=5, activation=None, kernel_initializer='glorot_uniform', name="conv2_%d" % i)
            x = conv_layer(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(4, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                  kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.02))(x)
        pred_output = Activation('softmax')(x)
        classifier = Model(inputs=feat_extractor.input, outputs=pred_output)
        opt = tf.keras.optimizers.get(self.optimiser)
        opt.learning_rate = self.learning_rate

        classifier.compile(optimizer=opt, loss=focal_loss(0.4, 2.0))
        return classifier

    def train_model(self, img_type, train_img_folder, train_labels, val_img_folder, val_labels, model_save_dir, model_weights_file_name, batch_size, num_epochs):
        number_of_train_images = len([img for img in os.listdir(train_img_folder) if img.endswith(img_type)])
        number_of_test_images = len([img for img in os.listdir(val_img_folder) if img.endswith(img_type)])
        earlystop = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                                  restore_best_weights=True)  # ensures that we have model weights corresponding to the best value of the metric at the end of

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
                                     mode='min', save_weights_only=False)
        history = self.model.fit_generator(image_data_gen(batch_size, train_img_folder, train_labels, img_type),
                                           epochs=num_epochs,
                                           steps_per_epoch=math.ceil(number_of_train_images / num_epochs),
                                           validation_data=image_data_gen(batch_size, val_img_folder, val_labels,
                                                                          img_type),
                                           validation_steps=math.ceil(number_of_test_images / batch_size),
                                           callbacks=[checkpoint, earlystop, tensorboard], verbose=2,
                                           shuffle=False,
                                           use_multiprocessing=True, workers=8)
