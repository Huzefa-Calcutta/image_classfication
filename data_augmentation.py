from imgaug import augmenters as iaa
import os
import cv2
import sys
import configparser
import json

config_file = sys.argv[1]
cfg_parser = configparser.ConfigParser()
cfg_parser.read(config_file)

train_img_dir = cfg_parser.get("images", "original_train")
valid_img_dir = cfg_parser.get("images", "original_valid")
save_dir = cfg_parser.get("images", "aug_train")
label_path = cfg_parser.get("labels", "image_labels_csv")
label_encoding_file = cfg_parser.get("labels", "label_encoding_file")
train_label_output = cfg_parser.get("labels", "label_train_file")
valid_label_output = cfg_parser.get("labels", "label_test_file")

# dictionary to map image to corresponding label encoding
train_label = {}
test_label = {}

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(label_encoding_file, 'r') as f:
    label_encoding_dict = json.load(f)

with open(label_path, 'r') as label_input:
    for line in label_input:
        tokens = line.strip().split(',')
        if tokens[2] == 'train':
            train_label[tokens[0]] = label_encoding_dict[tokens[1]]
        else:
            test_label[tokens[0]] = label_encoding_dict[tokens[1]]

sharp = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        img = cv2.imread(os.path.join(train_img_dir, image))
        img_aug = sharp(image=img)
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(image)[0] + "_sharp.jpeg"), img_aug)
        train_label[os.path.splitext(image)[0] + "_sharp.jpeg"] = train_label[image]

sig_contrast = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        img = cv2.imread(os.path.join(train_img_dir, image))
        img_aug = sig_contrast(image=img)
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(image)[0] + "_sig_contrast.jpeg"), img_aug)
        train_label[os.path.splitext(image)[0] + "_sig_contrast.jpeg"] = train_label[image]

gammma_contrast = iaa.GammaContrast((1.1, 2.0))
for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        img = cv2.imread(os.path.join(train_img_dir, image))
        img_aug = gammma_contrast(image=img)
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(image)[0] + "_gamma_contrast.jpeg"), img_aug)
        train_label[os.path.splitext(image)[0] + "_gamma_contrast.jpeg"] = train_label[image]

log_contrast = iaa.LogContrast(gain=(0.6, 1.4))
for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        img = cv2.imread(os.path.join(train_img_dir, image))
        img_aug = log_contrast(image=img)
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(image)[0] + "_log_contrast.jpeg"), img_aug)
        train_label[os.path.splitext(image)[0] + "_log_contrast.jpeg"] = train_label[image]

linear_contrast = iaa.LinearContrast((1.1, 1.6))
for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        img = cv2.imread(os.path.join(train_img_dir, image))
        img_aug = linear_contrast(image=img)
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(image)[0] + "_linear_contrast.jpeg"), img_aug)
        train_label[os.path.splitext(image)[0] + "_linear_contrast.jpeg"] = train_label[image]

flip_hori = iaa.Fliplr(1.0)
for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        img = cv2.imread(os.path.join(train_img_dir, image))
        img_aug = flip_hori(image=img)
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(image)[0] + "_flip_horizontal.jpeg"), img_aug)
        train_label[os.path.splitext(image)[0] + "_flip_horizontal.jpeg"] = train_label[image]

flip_vert = iaa.Flipud(1.0)
for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        img = cv2.imread(os.path.join(train_img_dir, image))
        img_aug = flip_vert(image=img)
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(image)[0] + "_flip_vertical.jpeg"), img_aug)
        train_label[os.path.splitext(image)[0] + "_flip_vertical.jpeg"] = train_label[image]

channel_flip = iaa.ChannelShuffle(1.0)
for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        img = cv2.imread(os.path.join(train_img_dir, image))
        img_aug = channel_flip(image=img)
        cv2.imwrite(os.path.join(save_dir, os.path.splitext(image)[0] + "_channel_flip.jpeg"), img_aug)
        train_label[os.path.splitext(image)[0] + "_channel_flip.jpeg"] = train_label[image]

for image in os.listdir(train_img_dir):
    if image.endswith(".jpeg"):
        os.rename(os.path.join(train_img_dir, image), os.path.join(save_dir, image))

with open(train_label_output, 'w') as out:
    out.write(json.dumps(train_label))

with open(valid_label_output, 'w') as out:
    out.write(json.dumps(test_label))
