import json
import os
import cv2
import numpy as np
import argparse
import ipdb
import pandas as pd
from tqdm import tqdm
from utils import *
import pickle
from skimage.feature import hog
from sklearn import svm
import imutils
from sklearn.metrics import classification_report,accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained HoG Person Detector')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--test', type=str, default="PennFudanPed_val.json")
    parser.add_argument('--out', type=str, default="PennFudanPed_predict_custom.json")
    parser.add_argument('--model',type=str, default="./PennFudanPed/hog_custom.sav")
    args = parser.parse_args()
    return args



def scaling_image(img,scale):
    print("Control in scaling")
    scaled_imgs = []
    h = img.shape[0]
    w = img.shape[1]
    # while(True):
    #     print("The infinite loop?")
    #     h_scale = int(h/scale)
    #     w_scale = int(w/scale)
    #     img_scale = cv2.resize(img,(w_scale,h_scale))
    #     if(img_scale.shape[0]<128 or img_scale.shape[1]<64):

    #         break
    #     scaled_imgs.append(img_scale)

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
        resized = imutils.resize(img, width = int(img.shape[1] * scale))
        r = img.shape[1] / float(resized.shape[1])
		# if the resized image is smaller than the template, then break
		# from the loop
        if resized.shape[0] < 64 or resized.shape[1] < 32:
            break
        scaled_imgs.append(resized)
    return scaled_imgs

def sliding_windows(img,stride):
    print("Control in sliding")
    windows = []
    for j in range(0,img.shape[0],stride[0]):
        for i in range(0, img.shape[1],stride[1]):
            window = [i,j,img[j:j+128,i:i+64]]
            windows.append(window)
    return windows

def make_predictions(clf,root,test_json, output_json):
    print("control in making preds")
    predictions = []
    # for saving images with predicted bboxes, and comparing them with annotations
    annotations = test_json['annotations'] # this is ONLY used for comparison of predicted bboxes
    annotations = pd.json_normalize(annotations)
    save_preds_dir = os.path.join(args.root, "predictions_hog_pretrained")
    if os.path.exists(save_preds_dir) == False:
        os.mkdir(save_preds_dir)

    # read the images using the file name in the json file
    print("\nstarting inference over given test.json")
    img_dicts = test_json['images']

    for img_dict in img_dicts:
        img = cv2.imread(os.path.join(root,img_dict['file_name']))
        img_id = img_dict['id']
        scale = 1.5
        pred_boxes = []
        pred_scores = []
        scaled_images = scaling_image(img,scale)
        stride = [10,10]
        for im in scaled_images:
            windows = sliding_windows(im,stride)
            for window in windows:
                image = window[2]
                if(image.shape[0] >= 128 and image.shape[1] >= 64):
                    # print("-----")
                    # print(image.shape)
                    feature =  hog(image,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=True)
                    # print(feature.shape)
                    # print("-----")
                    feature = feature.reshape(1,-1)
                    # feature = np.transpose(feature)
                    feature_class = clf.predict(feature)
                    if(feature_class == 1):
                        feature_confidence = clf.decision_function(feature)
                        pred_box = [window[0]*scale,window[1]*scale,64*scale,128*scale]
                        pred_boxes.append(pred_box)
                        pred_scores.append(feature_confidence)
        bboxes,scores = do_NMS(pred_boxes,pred_scores,0.8)
        for bb, score in zip(bboxes, scores):
            pred = {}
            pred["image_id"] = img_id
            pred["score"] = float(score)
            pred["category_id"] = 1
            pred["bbox"] = bb.astype(float).tolist()
            predictions.append(pred)

        save_img_with_pred(img, img_id, bboxes, scores, list(annotations.loc[annotations['image_id'] == img_id]['bbox']), save_preds_dir)
    
    with open(output_json, "w") as f:
        json.dump(predictions, f)

def main(root, test_json, output_json):
    # pretrained hog model
    clf = pickle.load(open(args.model, 'rb'))
    make_predictions(clf,root,test_json, output_json)

if __name__ == "__main__":
    args = parse_args()
    test_json = json.loads(open(args.test,'r').read())
    main(args.root, test_json, args.out)