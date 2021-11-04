import pandas as pd
import json
import os
import cv2
import numpy as np
import argparse
from imutils.object_detection import non_max_suppression

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained HoG Person Detector')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--test', type=str, default="./PennFudanPed/PennFudanPed_val.json")
    parser.add_argument('--out', type=str, default="./PennFudanPed/PennFudanPed_prediction.json")
    args = parser.parse_args()
    return args

def draw_rectangles(img, bboxes, scores):
    for idx, (x, y, w, h) in enumerate(bboxes):
        cv2.rectange(img, (x,y), (x+w,y+h), (0,255,0), 2)
    return img

def do_NMS(bboxes, scores):
    # changes x,y,w,h to x,y,x2,y2
    for idx in range(bboxes.shape[0]):
        bboxes[idx, 2] += bboxes[idx, 0]
        bboxes[idx, 3] += bboxes[idx, 1]
    
    bboxes_nms = non_max_suppression(bboxes, probs=None, overlapThresh=0.65)
    
    # get scores for these bounding boxes
    scores_nms = []
    for bb in bboxes_nms:
        scores_nms.append(scores[(bb == bboxes).mean(axis=1) == 1][0,0])
    scores_nms = np.array(scores_nms).reshape(-1,1)

    # changes x,y,x2,y2 to x,y,w,h
    for idx in range(bboxes_nms.shape[0]):
        bboxes_nms[idx, 2] = bboxes_nms[idx, 2] - bboxes_nms[idx, 0] + 1
        bboxes_nms[idx, 3] = bboxes_nms[idx, 3] - bboxes_nms[idx, 1] + 1

    return bboxes_nms, scores_nms

def main(root, test_json, output_json):
    # pretrained hog model
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(hog.getDefaultPeopleDetector())

    # predictions will be saved iteratively
    predictions = []

    # read the images using the file name in the json file
    test_json = json.loads(open(test_json,'r').read())
    img_dicts = test_json['images']
    for img_dict in img_dicts:
        img = cv2.imread(os.path.join(root, img_dict['file_name']))
        img_id = img_dict['id']

        # predict the bboxes using pretrained HoG
        bboxes, scores = hog.detectMultiScale(img)

        if len(scores) != 0:
            # do NMS and append the predictions in COCO format
            bboxes, scores = do_NMS(bboxes, scores)
        else:
            # no predictions
            bboxes, scores = np.array([[0,0,0,0]]), np.array([[0]])
            print("no prediction encountered")

        for bb, score in zip(bboxes, scores):
            pred = {}
            pred["image_id"] = img_id
            pred["score"] = float(score[0])
            pred["category_id"] = 1
            pred["bbox"] = bb.tolist()
            predictions.append(pred)
    
    with open(output_json, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    args = parse_args()
    main(args.root, args.test, args.out)