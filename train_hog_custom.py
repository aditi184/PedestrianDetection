import json
import os
import cv2
import numpy as np
import pandas as pd
import argparse
from utils import *
import random
# random.seed(25)
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

def iOU(x,y,bb,img):
    x1, y1, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    patch_dim = [[x,y],[x+64,y],[x+64,y+128],[x,y+128]]
    original_box = [[x1,y1],[x1+w,y1],[x1+w,y1+h],[x1,y1+h]]
    pred_box = [x,y,x+64,y+128]
    true_box = [x1,y1,x1+w,y1+h]

    px1 = max(true_box[0], pred_box[0])
    py1 = max(true_box[1], pred_box[1])
    px2 = min(true_box[2], pred_box[2])
    py2 = min(true_box[3], pred_box[3])
    
    intersection = max(0, px2 - px1 + 1) * max(0, py2 - py1 + 1)
    area1 = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
    area2 = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    union = area1 + area2 - intersection
    iou = intersection / float(union)


    # bg1 = np.zeros(img.shape).astype(img.dtype)
    # mask1 = [np.array(patch_dim)]
    # color = [255, 255, 255]
    # cv2.fillPoly(bg1, mask1, color)
    # result1 = cv2.bitwise_and(img, bg1)
    # bg2 = np.zeros(img.shape).astype(img.dtype)
    # mask2 = [np.array(original_box)]
    # cv2.fillPoly(bg2, mask2, color)
    # result2 = cv2.bitwise_and(img, mask2)
    # intersection = np.logical_and(result1, result2)
    # union = np.logical_or(result1, result2)
    return iou

def get_neg_patch(image,bb):
    y = random.randint(1,image.shape[0]-128)
    x = random.randint(1,image.shape[1]-64)
    if(iOU(x,y,bb,image)<0.1):
        patch = image[y:y+128,x:x+64,:]
        return patch
    else:
        get_neg_patch(image,bb)

'''
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
'''

def extract_patches(img, img_id, bboxes, positive_dir, patch_size=(64,128)):
    for idx, bbox in enumerate(bboxes):
        patch = get_patch(img, bbox)
        patch = cv2.resize(patch, patch_size)
        save_patch = os.path.join(positive_dir, str(img_id)+"_%u.jpg"%(idx))
        cv2.imwrite(save_patch, patch)
'''
def extract_neg_patches(img,img_id,bboxes,negative_dir, patch_size=(64,128)):
    for idx,bbox in enumerate(bboxes):
        patch = get_neg_patch(img,bbox)
        patch = cv2.resize(patch, patch_size)
        save_patch = os.path.join(negative_dir, str(img_id)+"_%u.jpg"%(idx))
        cv2.imwrite(save_patch, patch)
'''
def extract_neg_patches(img,img_id,bboxes,negative_dir, patch_size=(64,128)):
    for idx,bbox in enumerate(bboxes):
        patch = get_neg_patch(img,bbox)
        if patch is None:
            continue
        else:
            try:
                # print("PATCH" , idx , "   " , patch)
                patch = cv2.resize(patch, patch_size)

            except Exception as e:
                print(str(e))
            
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
'''
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

'''




def create_negative_samples(root,train_json):
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
        extract_neg_patches(img,img_id,bboxes,negative_dir)

def main(root, train_json, val_json, save_model):
    train_json = json.loads(open(train_json,'r').read())
    val_json = json.loads(open(val_json,'r').read())

    # create training data ie positive and negative samples for SVM using train_json
    create_positive_samples(root, train_json)
    create_negative_samples(root, train_json)

if __name__ == "__main__":
    args = parse_args()
    main(args.root, args.train, args.val, args.save_model)