import json
import os
import cv2
import numpy as np
import argparse
import ipdb
import pandas as pd
from tqdm import tqdm
from utils import *
import torch
from ipdb import set_trace

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained HoG Person Detector')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--test', type=str, default="PennFudanPed_val.json")
    parser.add_argument('--out', type=str, default="PennFudanPed_predict.json")
    args = parser.parse_args()
    return args

def show_hog_params(hog):
    print("pretrained hog parameters...")
    print("hog winSize", hog.winSize)
    print("hog blockSize", hog.blockSize)
    print("hog blockStride", hog.blockStride)
    print("hog cellSize", hog.cellSize)
    print("hog nbins", hog.nbins)
    print("hog histogramNormType", hog.histogramNormType)

def main(root, test_json, output_json):
    # pretrained hog model
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(hog.getDefaultPeopleDetector())
    show_hog_params(hog)
    sigmoid = torch.nn.Sigmoid() # use sigmoid to normalize svm scores

    # predictions will be saved iteratively
    predictions = []
    no_pred_count = 0
    nms_count = 0

    # for saving images with predicted bboxes, and comparing them with annotations
    annotations = test_json['annotations'] # this is ONLY used for comparison of predicted bboxes
    annotations = pd.json_normalize(annotations)
    save_preds_dir = os.path.join(args.root, "predictions_hog_pretrained")
    if os.path.exists(save_preds_dir) == False:
        os.mkdir(save_preds_dir)

    # read the images using the file name in the json file
    print("\nstarting inference over given test.json")
    img_dicts = test_json['images']

    for img_dict in tqdm(img_dicts):
        img = cv2.imread(os.path.join(root, img_dict['file_name']))
        img_id = img_dict['id']

        # predict the bboxes using pretrained HoG
        bboxes, scores = hog.detectMultiScale(img, winStride=(2, 2), padding=(10, 10), scale=1.02) # bboxes.dtype is int, scores.dtype is float
        scores = scores.reshape(-1)
        
        if len(scores) != 0:
            # do NMS and append the predictions in COCO format
            init = len(scores)
            bboxes, scores = do_NMS(bboxes, scores, overlapThresh=0.8) # bboxes.dtype is int, scores.dtype is float
            final = len(scores)
            nms_count += (init-final)
        
        if len(scores) == 0:
            # no predictions
            # print("no prediction encountered")
            no_pred_count+=1
            continue

        for bb, score in zip(bboxes, scores):
            pred = {}
            pred["image_id"] = img_id
            pred["score"] = sigmoid(torch.tensor(float(score))).item()
            pred["category_id"] = 1
            pred["bbox"] = bb.astype(float).tolist()
            predictions.append(pred)
        
        # for visualization of bboxes and comparison with annotations
        save_img_with_pred(img, img_id, bboxes, scores, list(annotations.loc[annotations['image_id'] == img_id]['bbox']), save_preds_dir)
    
    print("no predictions for %u images out of %u"%(no_pred_count, len(img_dicts)))
    with open(output_json, "w") as f:
        json.dump(predictions, f)

    print("Non-Maximal Suppression reduced %u Bounding Boxes"%(nms_count))

if __name__ == "__main__":
    args = parse_args()
    test_json = json.loads(open(args.test,'r').read())
    main(args.root, test_json, args.out)