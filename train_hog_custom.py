import json
import os
import cv2
import numpy as np
import pandas as pd
import argparse
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained HoG Person Detector')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--train', type=str, default="./PennFudanPed/PennFudanPed_train.json")
    parser.add_argument('--val', type=str, default="./PennFudanPed/PennFudanPed_val.json")
    parser.add_argument('--save_model', type=str, default="./PennFudanPed/hog_custom")
    args = parser.parse_args()
    return args

def get_patch(image, bb):
    x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]) # w is along x (right), h is along y (down)
    return image[y:y+h+1, x:x+w+1, :]

    
def get_neg_patch(image, bb):
    x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    temp = image[y:y+h+1, x:x+w+1, :]
    template = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    source = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(source,template,cv2.TM_CCOEFF_NORMED)
    threshold = 1
    loc = np.where( res < threshold)
    for pt in zip(*loc[::-1]):
        returnable = image[pt[1]:pt[1]+h+1,pt[0]:pt[0]+w+1,:]
    return returnable

def extract_patches(img, img_id, bboxes, positive_dir, patch_size=(64,128)):
    for idx, bbox in enumerate(bboxes):
        patch = get_patch(img, bbox)
        patch = cv2.resize(patch, patch_size)
        save_patch = os.path.join(positive_dir, str(img_id)+"_%u.jpg"%(idx))
        cv2.imwrite(save_patch, patch)

def extract_neg_patches(img,img_id,bboxes,negative_dir, patch_size=(64,128)):
    for idx,bbox in enumerate(bboxes):
        patch = get_neg_patch(img,bbox)
        patch = cv2.resize(patch, patch_size)
        save_patch = os.path.join(negative_dir, str(img_id)+"_%u.jpg"%(idx))
        cv2.imwrite(save_patch, patch)

def create_positive_samples(root, train_json):
    # extract patches containing pedestrians using given annotations, resize them, and save inside "root/PennFudanPed/Positive"
    positive_dir = os.path.join(os.path.join(root, "PennFudanPed"), "Positive")
    if os.path.exists(positive_dir) == False:
        os.mkdir(positive_dir)

    img_dicts = train_json['images']
    annotations = train_json['annotations']
    annotations = pd.json_normalize(annotations)

    for img_dict in img_dicts:
        img = cv2.imread(os.path.join(root,img_dict['file_name']))
        img_id = img_dict['id']
        bboxes = list(annotations.loc[annotations['image_id'] == img_id]['bbox'])

        extract_patches(img, img_id, bboxes, positive_dir)

def create_negative_samples(root, train_json):
    # extract patches that don't contain pedestrians, resize them, and save inside "root/PennFudanPed/Negative"
    negative_dir = os.path.join(os.path.join(root, "PennFudanPed"), "Negative")
    if os.path.exists(negative_dir) == False:
        os.mkdir(negative_dir)
    img_dicts = train_json['images']
    annotations = train_json['annotations']
    annotations = pd.json_normalize(annotations)

    for img_dict in img_dicts:
        img = cv2.imread(os.path.join(root,img_dict['file_name']))
        img_id = img_dict['id']
        bboxes = list(annotations.loc[annotations['image_id'] == img_id]['bbox'])

        extract_neg_patches(img, img_id, bboxes, negative_dir)

    

def main(root, train_json, val_json, save_model):
    train_json = json.loads(open(train_json,'r').read())
    val_json = json.loads(open(val_json,'r').read())

    # create training data ie positive and negative samples for SVM using train_json
    create_positive_samples(root, train_json)
    create_negative_samples(root, train_json)

if __name__ == "__main__":
    args = parse_args()
    main(args.root, args.train, args.val, args.save_model)