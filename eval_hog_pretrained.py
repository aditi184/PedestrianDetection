import json
import os
import cv2
import numpy as np
import argparse
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained HoG Person Detector')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--test', type=str, default="./PennFudanPed/PennFudanPed_val.json")
    parser.add_argument('--out', type=str, default="./PennFudanPed/PennFudanPed_prediction.json")
    args = parser.parse_args()
    return args

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